# Spletni strežnik (zgrajen z ogrodjem FastAPI), ki prikazuje pare vprašanj
# in odgovorov na podlagi besedilnih odsekov iz PDF dokumentov. Uporabnik
# jih pregleda, lahko popravi in odda povratne informacije o tem ali se mu dani
# par zdi ustrezno zastavljen. Rezultati se za potrebe kasnejše analize
# shranijo v app_data/feedback.json.

# Spletni strežnik, ki skrbi za prikaz spletne strani in sprejemanje/vraćanje
# podatkov.
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
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
from datetime import datetime
# Beleženje dogodkov in napak.
import logging

# Ustvarimo instanco FastAPI aplikacije.
app = FastAPI()

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
            "boundingBox": item["boundingBox"]
        })
# Končni seznam parov vprašanj in odgovorov, skupaj z metapodatki.
qa_data = flattened_data

# Iz diska naložimo HTML predloge, ki so v obliki samostojnih strani:
# Prikaže glavno stran z enim parom vprašanje-odgovor.
index_template = Template(Path("templates/index.html").read_text(encoding="utf-8"))
# Sporoči napako, tj. da ni na voljo nobenih parov vprašanj in odgovorov.
no_qa_template = Template(Path("templates/no_qa.html").read_text(encoding="utf-8"))
# Izpiše zahvalo uporabniku za sodelovanje.
thank_you_template = Template(Path("templates/thank_you.html").read_text(encoding="utf-8"))

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
    """
    Function to download a PDF file from a web URL.
    - First, compute the MD5 hash of the URL to get a unique
      filename for the local copy of the file.
    - If the file does not exist yet (pdf_path.exists()), download it 
      using requests.get.
    - If the download fails, log an error and return None.
    - If the download succeeds, return the path (Path) to the local PDF file.
    """
    pdf_hash = hashlib.md5(url.encode()).hexdigest()
    pdf_path = Path(f"static/{pdf_hash}.pdf")
    if not pdf_path.exists():
        try:
            # HTTP GET zahteva.
            resp = requests.get(url)
            # Sproži izjemo, če koda v odgovoru ni 200.
            resp.raise_for_status()  # Raise an exception for bad responses
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
# - Če je podan robni okvir (bounding_box) se nanese pol prosojna roza plast
#   znotraj podanih koordinat.
# - Slika se shrani kot 'static/rendered.png'.
# - Vrnjena je pot do renderirane slike v obliki niza (String).
# Če pride v postopku do napake vrnemo prazen niz, da s tem označimo neupoštevanje
# danega para vprašanje-odgovor.
def render_pdf_page(pdf_path: Path, page_number: int, bounding_box: list) -> str:
    """
    Function to render the specified page of a PDF file.
    Steps:
    - Open the PDF file.
    - Render the specified PDF page as an image.
    - If a bounding box is provided, apply a semi-transparent pink overlay
      within the given coordinates.
    - Save the image as 'static/rendered.png'.
    - Return the path to the rendered image as a string.
    If an error occurs during the process, return an empty string to indicate
    that the given question-answer pair should be skipped.
    """
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

    # Če je bounding_box prazen ali None preskočimo risanje prekrivne plasti.
    if bounding_box:
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
        box = bounding_box[0]
        # Izhodiščne koordinate iz podatkov (v točkah).
        # Levo.
        l = box["l"]
        # Zgoraj.
        t = box["t"]
        # Desno.
        r = box["r"]
        # Spodaj.
        b = box["b"]
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

# Definicija glavne, t.i. home HTTP poti (ang. route):
# - Če qa_data ne vsebuje nobenih parov vprašanj in odgovorov, prikažemo stran
#   'no_qa'.
# - V nasprotnem primeru poiščemo prvi par vprašanje-odgovor za katerega funkcija
#   render_qa_partial vrne veljaven HTML. Če ob tem pride to težav izbrani par preskočimo
#   in gremo na naslednjega.
# - Ko imamo veljaven HTML ga vstavimo v glavno predlogo index.html.
@app.get("/", response_class=HTMLResponse)
def home():
    """
    Definition of the main, home HTTP route:
    - If qa_data contains no question-answer pairs, display the 'no_qa' page.
    - Otherwise, find the first question-answer pair for which the function
      render_qa_partial returns valid HTML. If there is an issue with the selected pair,
      skip it and move on to the next one.
    - Once valid HTML is obtained, insert it into the main template index.html.
    """
    # Če je seznam prazen takoj prikažemo no_qa.
    if not qa_data:
        return HTMLResponse(no_qa_template.render())

    # Poiščemo prvi element, ki se uspešno izriše.
    idx = 0
    partial_html = ""
    while idx < len(qa_data):
        partial_html = render_qa_partial(idx, edit_mode=False)
        if partial_html:
            break
        # Če renderiranje ni uspelo to zabeležimo in odstranimo izbrani element
        # iz seznama.
        logging.info(f"Skipping broken QA at index {idx}")
        qa_data.pop(idx)

    # Če se seznam izprazni zaradi preskakovanja neveljavnih parov vprašanj in 
    # odgovorov prav tako prikaži no_qa.
    if not qa_data:
        return HTMLResponse(no_qa_template.render())

    # HTML delno predlogo vstavimo v celotno predlogo.
    final_html = index_template.render(qa_content=partial_html)
    return HTMLResponse(final_html)

# TODO Je spodnja route potrebna?
# HTTP pot (ang. route), ki prikliče predlogo no_qa.html.
# @app.get("/no-qa", response_class=HTMLResponse)
# def no_qa():
#     """
#     HTTP route that calls the no_qa.html template.
#     """
#     return HTMLResponse(no_qa_template.render())

# HTTP pot, ki prikaže zahvalo, ko uporabnik pregleda vse pare 
# vprašanj in odgovorov.
@app.get("/thank-you", response_class=HTMLResponse)
def thank_you():
    """
    HTTP route that displays a thank-you page when the user 
    has reviewed all question-answer pairs.
    """
    return HTMLResponse(thank_you_template.render())

# Funkcija vrne HTML fragment (brez <html> in <body>), ki vsebuje:
# - Izrisano stran PDFja (skupaj z robnim okvirjem, če je ta podan).
# - Par vprašanje-odgovor.
# - Gumb za pošiljanje povratne informacije (če je edit_mode=True,
#   dodamo še polje za popravek vprašanja in/ali odgovora).
# Če pride do napake (npr. PDF ni dosegljiv, slika se ne more zgenerirati),
# vrnemo prazen niz, da se lahko problematičen par preskoči.
def render_qa_partial(index: int, edit_mode: bool) -> str:
    """
    The function returns an HTML fragment (without <html> and <body>) that contains:
    - Rendered PDF page (including the bounding box if provided).
    - A question-answer pair.
    - A button to submit feedback (if edit_mode=True,
      also adds fields for correcting the question and/or answer).
    If an error occurs (e.g., the PDF is not accessible, or the image cannot be generated),
    it returns an empty string so that the problematic pair can be skipped.
    """
    # Glede na dani indeks pridobimo element iz seznama.
    item = qa_data[index]
    # Najprej naložimo PDF (brez dela poti, ki označuje stran).
    pdf_path = download_pdf(item["fileUrl"].split("#")[0])
    
    # Če prenos ne uspe element preskočimo.
    if pdf_path is None:
        logging.warning(f"Skipping QA item at index {index} due to PDF download failure.")
        return ""

    # Renderiramo sliko strani z robnim okvirjem.
    image_url = render_pdf_page(pdf_path, item["pageNumber"], item.get("boundingBox", []))
    
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
def edit_qa(index: int):
    """
    HTTP route that calls the template in edit mode for question-answer pairs.
    The GET parameter 'index' specifies which item to display.
    Similar to home(), invalid items are skipped until we find one that can be shown.
    When no items remain, the user is redirected to the thank-you page.
    """
    while index < len(qa_data):
        partial = render_qa_partial(index, edit_mode=True)
        if partial:
            return HTMLResponse(partial)
        logging.info(f"Skipping broken QA at index {index}")
        qa_data.pop(index)

    return HTMLResponse('<script>window.location.href="/thank-you";</script>')

# Deluje podobno kot pot /edit_qa, le da prikliče predlogo v načinu samo za branje.
@app.get("/display_qa", response_class=HTMLResponse)
def display_qa(index: int):
    """
    Works similarly to the route /edit_qa, but calls the template in read-only mode.
    """
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
def evaluate(
    # Indeks para, ki ga ocenjujemo.
    index: int = Form(...),
    # Vrednosti: "skip", "adequate", "inadequate" ali "corrected".
    evaluation: str = Form(...),
    # Popravljeno vprašanje (če evaluation == "corrected").
    correctedQuestion: str = Form(None),
    # Popravljen odgovor (če evaluation == "corrected").
    correctedAnswer: str = Form(None)
):
    """
    Process the user's evaluation of a question-answer pair:
    - If the index is invalid, redirect directly to the thank-you page.
    - Prepare a 'record' object containing the original question-answer pair,
      metadata, and new values (evaluation, corrections).
    - If evaluation == "skip", set record["skipped"] = True.
    - If evaluation == "adequate" or "inadequate", set record["evaluation"] accordingly.
    - If evaluation == "corrected", save the corrected question and answer.
    - Also record the submission time by adding a timestamp to the record.
    - Write the record to app_data/feedback.json by appending it to existing entries (not overwriting).
    - Remove the processed element from qa_data.
    - If no elements remain, show thank-you; otherwise, find the next one.
    """
    # Preverimo ali indeks obstaja.
    if index < 0 or index >= len(qa_data):
        return HTMLResponse('<script>window.location.href="/thank-you";</script>')

    item = qa_data[index]

    # Pripravimo timestamp.
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
        "timestamp": current_timestamp
    }

    # Nastavimo ustrezna polja glede na izbrane parametre.
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

    # Preberemo obstoječo vsebino datoteke feedback.json in vanjo dodamo nov vnos.
    existing = json.loads(feedback_path.read_text(encoding="utf-8"))
    existing.append(record)
    feedback_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

    # Obdelani par vprašanje-odgovor odstranimo iz spomina.
    qa_data.pop(index)

    # Če ni več neobdelanih vprašanj prikažemo thank-you.
    if not qa_data:
        return HTMLResponse('<script>window.location.href="/thank-you";</script>')

    # V nasprotnem primeru poiščemo naslednjega, ki se uspešno izriše.
    next_idx = index
    while next_idx < len(qa_data):
        partial = render_qa_partial(next_idx, edit_mode=False)
        if partial:
            return HTMLResponse(partial)
        logging.info(f"Skipping broken QA at index {next_idx}")
        qa_data.pop(next_idx)

    # Če se je seznam izpraznil prikažemo thank-you.
    return HTMLResponse('<script>window.location.href="/thank-you";</script>')
