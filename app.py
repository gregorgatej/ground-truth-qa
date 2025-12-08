# Spletni strežnik (zgrajen z ogrodjem FastAPI), ki prikazuje pare vprašanj
# in odgovorov na podlagi besedilnih odsekov iz PDF dokumentov. Uporabnik
# jih pregleda, lahko popravi in odda povratne informacije o tem ali se mu dani
# par zdi ustrezno zastavljen. Rezultati se za potrebe kasnejše analize
# shranijo v app_data/feedback.json.

# Spletni strežnik, ki skrbi za prikaz spletne strani in sprejemanje/vraćanje
# podatkov.
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
# Delo z datotečnimi potmi.
from pathlib import Path
# Shranjevanje in branje podatkov v JSON formatu.
import json
# Prenos PDFjev iz oddaljenih naslovov.
import requests
# Omogočata odpiranje, prikazovanje in risanje po PDFjih.
import fitz
from PIL import Image, ImageDraw
# Generiranje zgoščenih vrednostih (ang. hashes).
import hashlib
# Prikaz dinamičnih HTML strani prek predlog (ang. templates).
from jinja2 import Template
# Merjenje trenutnega časa od začetka Unix epohe (tj. 1. 1. 1970, 00:00:00 UTC).
# Vrne število sekund (z decimalnim delom) pretečenih od slednje do zdajšnjega
# trenutka.
from time import time
# Delo z datumi in časi.
from datetime import datetime, timedelta
# Beleženje dogodkov in napak.
import logging
# Drugo ...
import os
from minio import Minio
from urllib.parse import quote
import unicodedata
import asyncio
from filelock import FileLock

from dotenv import load_dotenv
load_dotenv()

SESSION_SECRET = os.getenv("SESSION_SECRET")
if not SESSION_SECRET:
    raise RuntimeError("SESSION_SECRET environment variable is not set")

S3_ENDPOINT = "moja.shramba.arnes.si"
S3_BUCKET = "zrsvn-rag-najdbe-vecji"

# MinIO config
s3_client = Minio(
    S3_ENDPOINT,
    access_key=os.getenv("S3_ACCESS_KEY"),
    secret_key=os.getenv("S3_SECRET_ACCESS_KEY"),
    secure=True
)

def get_fresh_presigned_url(file_s3_path: str, hours: int = 1) -> str | None:
    """Vrne presigned URL za dani key ali None ob napaki."""
    try:
        logging.debug("Generating presigned URL for key: %r", file_s3_path)
        return s3_client.presigned_get_object(
            S3_BUCKET,
            file_s3_path,
            expires=timedelta(hours=hours),
        )
    except Exception:
        logging.exception("Error generating presigned URL for %s", file_s3_path)
        return None

# Ustvarimo instanco FastAPI aplikacije.
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)

# Nastavimo, da se vsebina mape "static" streže pod URLjem /static.
app.mount("/static", StaticFiles(directory="static"), name="static")
# Enako nastavimo za mapo "assets" oz. URL /assets.
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# Definiramo pot do datoteke, ki vsebuje pare vprašanj in odgovorov.
qa_data_path = Path("app_data/qa_data.json")
# Pot do datoteke kamor shranjujemo povratne informacije.
feedback_path = Path("app_data/feedback.json")
# Če povratne informacije še ne obstajajo ustvarimo prazno JSON polje.
if not feedback_path.exists():
    feedback_path.write_text("[]", encoding="utf-8")

# Preberemo surove podatke iz qa_data_path in jih pretvorimo v Python objekte.
raw_data = json.loads(qa_data_path.read_text(encoding="utf-8"))
# Ker so podatki gnezdeni jih shranimo v seznam kjer se bodo vsi nahajali
# na isti ravni, tj. vsak element vsebuje par vprašanje-odgovor in dodatne
# metapodatke.
flattened_data = []
for item in raw_data:
    for qa in item["questions_answers"]:
        flattened_data.append({
            "question": qa["question"],
            "answer": qa["answer"],
            "text": item["text"],
            "chunkID": item["chunkID"],
            "fileUrl": item["fileUrl"],
            "fileS3Path": item["fileS3Path"],
            "fileName": item["fileName"],
            "pageNumber": item["pageNumber"],
            "boundingBox": item["boundingBox"],
        })
# Končni seznam parov vprašanj in odgovorov, skupaj z metapodatki.
qa_data = flattened_data
qa_lock = asyncio.Lock()
feedback_lock = FileLock(str(feedback_path) + ".lock")

# Iz diska naložimo HTML predloge, ki so v obliki samostojnih strani:
# Prikaže glavno stran z enim parom vprašanje-odgovor.
index_template = Template(Path("templates/index.html").read_text(encoding="utf-8"))
# Sporoči napako, tj. da ni na voljo nobenih parov vprašanj in odgovorov.
no_qa_template = Template(Path("templates/no_qa.html").read_text(encoding="utf-8"))
# Izpiše zahvalo uporabniku za sodelovanje.
thank_you_template = Template(Path("templates/thank_you.html").read_text(encoding="utf-8"))
login_template = Template(Path("templates/login.html").read_text(encoding="utf-8"))

# Dodatne predloge, ki vsebuje samo del strani:
# Prikaže sliko strani PDF dokumenta, skupaj s parom vprašanje-odgovor in gumbi
# za ocenjevanje.
qa_item_readonly_template = Template(Path("templates/qa_item_readonly.html").read_text(encoding="utf-8"))
# Prikaže sliko strani PDF dokumenta, skupaj s parom vprašanje-odgovor, ki ga
# prikaže v urejevalnih poljih (<textarea>) kjer lahko uporabnik spremeni besedilo.
qa_item_edit_template = Template(Path("templates/qa_item_edit.html").read_text(encoding="utf-8"))

# Nastavimo logiranje (minimalna stopnja sporočil, ki se bodo beležila so tista na
# INFO nivoju).
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Funkcija za prenos PDF datoteke s spletnega naslova.
# - Najprej na podlagi URL-ja izračunamo MD5 zgoščeno vrednost, da dobimo unikatno
#   ime za lokalno kopijo datoteke.
# - Če datoteka še ne obstaja (pdf_path.exists()) jo prenesemo z requests.get.
# - Če je prenos neuspešen zabeležimo napako in vrnemo None.
# - Če je prenos uspešen vrnemo pot (Path) do lokalne PDF datoteke.
def download_pdf(url: str) -> Path:
    pdf_hash = hashlib.md5(url.encode()).hexdigest()
    pdf_path = Path(f"static/{pdf_hash}.pdf")
    if not pdf_path.exists():
        try:
            # HTTP GET zahteva.
            resp = requests.get(url)
            # Sproži izjemo, če koda v odgovoru ni 200.
            resp.raise_for_status()
            pdf_path.write_bytes(resp.content)
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading PDF from {url}: {e}")
            return None
    # Če datoteka že obstaja ali je bila uspešno prenesena vrnemo pot.
    return pdf_path

# Funkcija za prikaz podane strani PDF datoteke.
# Koraki:
# - PDF datoteko odpremo.
# - Dano PDF stran izrišemo kot sliko (tj. jo renderiramo).
# - Nanese se pol prosojna roza plast, glede na podan robni okvir (bounding_box)
#   znotraj podanih koordinat.
# - Slika se shrani kot 'static/rendered.png'.
# - Vrnjena je pot do renderirane slike v obliki niza (String).
# Če pride v postopku do napake vrnemo prazen niz, da s tem označimo neupoštevanje
# danega para vprašanje-odgovor.
def render_pdf_page(pdf_path: Path, page_number: int, bounding_box: dict) -> str:
    if pdf_path is None:
        # Če PDF ni na voljo ga preskočimo.
        return ""
    
    try:
        # PDF dokument odpremo.
        doc = fitz.open(pdf_path)
        # page_number prične z 1, medtem ko fitz prične štetje z 0, zato
        # page_number - 1.
        page = doc.load_page(page_number - 1)
    except Exception as e:
            logging.error(f"Error opening PDF file {pdf_path}: {e}")
            return ""
    # Nastavimo resolucijo za renderiranje.
    render_dpi = 150
    scale_factor = render_dpi / 72
    render_matrix = fitz.Matrix(scale_factor, scale_factor)
    # PDF je vektorji zgrajena datoteka, zato jo je treba za prikaz na 
    # zaslonu ali shranjevanje v slikovnem formatu renderirati v piksle.
    # Stran iz PDF se pretvori (renderira) v raster sliko oz. pixmap objekt,
    # ki vsebuje podatke o pikslih (barva, alfa kanal itd.).
    pix = page.get_pixmap(matrix=render_matrix)
    img_path = Path("static/rendered.png")
    pix.save(img_path)

    img = Image.open(img_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    page_width = page.rect.width
    page_height = page.rect.height
    img_width, img_height = img.size
    x_scale = img_width / page_width
    y_scale = img_height / page_height
    # Mesto okoli robnega okvirja rahlo povečamo.
    PADDING_FACTOR = 0.05
    # Izhodiščne koordinate iz podatkov (v točkah).
    # Levo.
    l = bounding_box["l"]
    # Zgoraj.
    t = bounding_box["t"]
    # Desno.
    r = bounding_box["r"]
    # Spodaj.
    b = bounding_box["b"]
    pad_x = (r - l) * PADDING_FACTOR
    pad_y = (t - b) * PADDING_FACTOR
    # Okvir glede na PADDING_FACTOR razširimo, vendar pazimo, da ne
    # gremo čez rob strani.
    l = max(0, l - pad_x)
    r = min(page_width, r + pad_x)
    t = min(page_height, t + pad_y)
    b = max(0, b - pad_y)

    # Koordinate navpično obrnemo, ker je y-koordinata v koordinatnem sistemu
    # podanega PDFja (kot ga definira PyMuPDF/fitz)
    # izmerjena od spodaj navzgor (tj. ima izhodišče (0,0) v spodnjem
    # levem kotu strani), medtem ko je referenčna točka naših
    # bounding_box koordinat zgornji levi kot.
    t_corrected = page_height - t
    b_corrected = page_height - b

    l_scaled = l * x_scale
    r_scaled = r * x_scale
    t_scaled = t_corrected * y_scale
    b_scaled = b_corrected * y_scale

    # Narišemo pol prosojno roza plast.
    draw.rectangle([l_scaled, t_scaled, r_scaled, b_scaled], fill=(255, 182, 193, 100))

    # Izvirno sliko in roza plast združimo.
    img = Image.alpha_composite(img, overlay)
    img.save(img_path)

    return str(img_path)

@app.get("/login", response_class=HTMLResponse)
def login_page():
    return HTMLResponse(login_template.render())

@app.post("/login", response_class=HTMLResponse)
def login(request: Request, email: str = Form(...)):
    request.session["email"] = email
    return HTMLResponse('<script>window.location.href="/";</script>')

# Definicija glavne, t.i. home HTTP poti (ang. route):
# - Če qa_data ne vsebuje nobenih parov vprašanj in odgovorov, prikažemo stran
#   'no_qa'.
# - V nasprotnem primeru poiščemo prvi par vprašanje-odgovor za katerega funkcija
#   render_qa_partial vrne veljaven HTML. Če ob tem pride to težav izbrani par preskočimo
#   in gremo na naslednjega.
# - Ko imamo veljaven HTML ga vstavimo v glavno predlogo index.html.
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    if "email" not in request.session:
        return HTMLResponse('<script>window.location.href="/login";</script>')

    async with qa_lock:
        if not qa_data:
            return HTMLResponse(no_qa_template.render())

        idx = 0
        partial_html = ""
        while idx < len(qa_data):
            partial_html = render_qa_partial(idx, edit_mode=False)
            if partial_html:
                break
            logging.info(f"Skipping broken QA at index {idx}")
            qa_data.pop(idx)

        if not qa_data:
            return HTMLResponse(no_qa_template.render())

        final_html = index_template.render(qa_content=partial_html)
    return HTMLResponse(final_html)

# HTTP pot, ki prikaže zahvalo, ko uporabnik pregleda vse pare 
# vprašanj in odgovorov.
@app.get("/thank-you", response_class=HTMLResponse)
async def thank_you(request: Request):
    if "email" not in request.session:
        return HTMLResponse('<script>window.location.href="/login";</script>')
    return HTMLResponse(thank_you_template.render())

# Funkcija vrne HTML fragment (brez <html> in <body>), ki vsebuje:
# - Izrisano stran PDFja (skupaj z robnim okvirjem, če je ta podan).
# - Par vprašanje-odgovor.
# - Gumb za pošiljanje povratne informacije (če je edit_mode=True,
#   dodamo še polje za popravek vprašanja in/ali odgovora).
# Če pride do napake (npr. PDF ni dosegljiv, slika se ne more zgenerirati),
# vrnemo prazen niz, da se lahko problematičen par preskoči.
def render_qa_partial(index: int, edit_mode: bool) -> str:
    # Glede na dani indeks pridobimo element iz seznama.
    item = qa_data[index]

    key = item["fileS3Path"]  # uporabi neposredno, brez normalizacije
    fresh_url = get_fresh_presigned_url(key)

    if not fresh_url:
        logging.warning("Cannot generate presigned URL for %s", key)
        return ""

    pdf_path = download_pdf(fresh_url)
    
    # Če prenos ne uspe element preskočimo.
    if pdf_path is None:
        logging.warning(f"Skipping QA item at index {index} due to PDF download failure.")
        return ""

    # Renderiramo sliko strani z robnim okvirjem.
    image_url = render_pdf_page(pdf_path, item["pageNumber"], item["boundingBox"])
    
    # Če renderiranje ne uspe element preskočimo.
    if not image_url:  # If the image couldn't be rendered
        logging.warning(f"Skipping QA item at index {index} due to rendering failure.")
        return ""

    # Dodamo parameter t(čas), da brskalnik vedno zahteva novo kopijo slike
    # in ne uporabi stare iz predpomnilnika. 
    image_url = f"/{image_url}?t={int(time())}"

    if edit_mode:
        # Če imamo podan edit_mode=True vrnemo predlogo z možnostjo vnašanja
        # popravkov.
        return qa_item_edit_template.render(
            index=index,
            question=item["question"],
            answer=item["answer"],
            image_url=image_url
        )
    else:
        # V nasprotnem primeru vrnemo predlogo v načinu zgolj za branje.
        return qa_item_readonly_template.render(
            index=index,
            question=item["question"],
            answer=item["answer"],
            image_url=image_url
        )

# HTTP pot (ang. route), ki prikliče predlogo v načinu urejanja parov vprašanj in
# odgovorov. Prek GET parametra 'index' pridobimo podatek o elementu, ki je
# prikazan. Podobno kot pri home() preskakujemo neveljavne elemente, dokler
# ne najdemo takšnega, ki ga lahko prikažemo. Ko zmanjka elementov, uporabnika
# preusmerimo na thank-you.
@app.get("/edit_qa", response_class=HTMLResponse)
async def edit_qa(request: Request, index: int):
    if "email" not in request.session:
        return HTMLResponse('<script>window.location.href="/login";</script>')
    async with qa_lock:
        while index < len(qa_data):
            partial = render_qa_partial(index, edit_mode=True)
            if partial:
                return HTMLResponse(partial)
            logging.info(f"Skipping broken QA at index {index}")
            qa_data.pop(index)
    return HTMLResponse('<script>window.location.href="/thank-you";</script>')

# Deluje podobno kot pot /edit_qa, le da prikliče predlogo v načinu samo za branje.
@app.get("/display_qa", response_class=HTMLResponse)
async def display_qa(request: Request, index: int):
    if "email" not in request.session:
        return HTMLResponse('<script>window.location.href="/login";</script>')
    async with qa_lock:
        while index < len(qa_data):
            partial = render_qa_partial(index, edit_mode=False)
            if partial:
                return HTMLResponse(partial)
            logging.info(f"Skipping broken QA at index {index}")
            qa_data.pop(index)
    return HTMLResponse('<script>window.location.href="/thank-you";</script>')

# Procesiramo uporabnikovo evaluacijo para vprašanje-odgovor:
# - Če je indeks neveljaven, neposredno preusmerimo na thank-you.
# - Pripravimo objekt 'record', ki vsebuje starejši par vprašanje-odgovor,
#   originalne metapodatke in nove vrednosti (evaluation, popravki).
# - Če evaluation == "skip", označimo record["skipped"] = True.
# - Če evaluation == "adequate" ali "inadequate", ustrezno nastavimo
#   record["evaluation"].
# - Če evaluation == "corrected", shranimo popravljeno vprašanje in odgovor.
# - V record zabeležimo tudi čas vnosa (tj. dodamo timestamp).
# - Zapišemo record v app_data/feedback.json (na način, da ga dodamo
#   obstoječim zapisom (append in ne overwrite).
# - Ko je element obdelan ga iz qa_data odstranimo.
# - Če ni več elementov prikažemo thank-you, sicer poiščemo naslednjega.
@app.post("/evaluate", response_class=HTMLResponse)
async def evaluate(
    request: Request,
    index: int = Form(...),
    evaluation: str = Form(...),
    correctedQuestion: str = Form(None),
    correctedAnswer: str = Form(None)
):
    if "email" not in request.session:
        return HTMLResponse('<script>window.location.href="/login";</script>')

    async with qa_lock:
        if index < 0 or index >= len(qa_data):
            return HTMLResponse('<script>window.location.href="/thank-you";</script>')

        item = qa_data[index]
        current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        record = {
            "question": item["question"],
            "answer": item["answer"],
            "text": item["text"],
            "chunkID": item["chunkID"],
            "fileUrl": item["fileUrl"],
            "fileS3Path": item["fileS3Path"],
            "fileName": item["fileName"],
            "evaluation": None,
            "correctedQuestion": None,
            "correctedAnswer": None,
            "skipped": False,
            "timestamp": current_timestamp,
            "userEmail": request.session.get("email")
        }

        if evaluation == "skip":
            record["evaluation"] = None
            record["skipped"] = True
        elif evaluation == "adequate":
            record["evaluation"] = "adequate"
        elif evaluation == "inadequate":
            record["evaluation"] = "inadequate"
        elif evaluation == "corrected":
            record["evaluation"] = "corrected"
            record["correctedQuestion"] = correctedQuestion
            record["correctedAnswer"] = correctedAnswer

        with feedback_lock:
            existing = json.loads(feedback_path.read_text(encoding="utf-8"))
            existing.append(record)
            feedback_path.write_text(
                json.dumps(existing, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )

        qa_data.pop(index)

        if not qa_data:
            return HTMLResponse('<script>window.location.href="/thank-you";</script>')

        next_idx = index
        while next_idx < len(qa_data):
            partial = render_qa_partial(next_idx, edit_mode=False)
            if partial:
                return HTMLResponse(partial)
            logging.info(f"Skipping broken QA at index {next_idx}")
            qa_data.pop(next_idx)

    return HTMLResponse('<script>window.location.href="/thank-you";</script>')
