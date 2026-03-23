# Web server (built with FastAPI framework) that displays question-answer pairs
# based on text segments from PDF documents. The user reviews them, can correct them, 
# and submits feedback on whether the given pair seems appropriately formulated. 
# Results are saved in app_data/feedback.json for later analysis.
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from pathlib import Path
import json
import requests
 # Enables opening, displaying, and drawing on PDFs.
import fitz
from PIL import Image, ImageDraw
import hashlib
from jinja2 import Template
from time import time
from datetime import datetime, timedelta
import logging
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

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

 # Define the path to the file containing question-answer pairs.
qa_data_path = Path("app_data/qa_data.json")
feedback_path = Path("app_data/feedback.json")
if not feedback_path.exists():
    feedback_path.write_text("[]", encoding="utf-8")

# Read raw data from qa_data_path and convert it to Python objects.
raw_data = json.loads(qa_data_path.read_text(encoding="utf-8"))
# Since the data is nested, store it in a list where all items are at the same level, i.e., 
# each element contains a question-answer pair and additional metadata.
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
# Final list of question-answer pairs, together with metadata.
qa_data = flattened_data
qa_lock = asyncio.Lock()
feedback_lock = FileLock(str(feedback_path) + ".lock")

index_template = Template(Path("templates/index.html").read_text(encoding="utf-8"))
no_qa_template = Template(Path("templates/no_qa.html").read_text(encoding="utf-8"))
thank_you_template = Template(Path("templates/thank_you.html").read_text(encoding="utf-8"))
login_template = Template(Path("templates/login.html").read_text(encoding="utf-8"))
qa_item_readonly_template = Template(Path("templates/qa_item_readonly.html").read_text(encoding="utf-8"))
qa_item_edit_template = Template(Path("templates/qa_item_edit.html").read_text(encoding="utf-8"))

 # Set up logging (minimum level of messages to be recorded is INFO).
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def download_pdf(url: str) -> Path:
    pdf_hash = hashlib.md5(url.encode()).hexdigest()
    pdf_path = Path(f"static/{pdf_hash}.pdf")
    if not pdf_path.exists():
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            pdf_path.write_bytes(resp.content)
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading PDF from {url}: {e}")
            return None
    return pdf_path

def render_pdf_page(pdf_path: Path, page_number: int, bounding_box: dict) -> str:
    if pdf_path is None:
        return ""
    
    try:
        doc = fitz.open(pdf_path)
        # page_number starts at 1, while fitz starts counting from 0, so page_number - 1.
        page = doc.load_page(page_number - 1)
    except Exception as e:
            logging.error(f"Error opening PDF file {pdf_path}: {e}")
            return ""
    
    render_dpi = 150
    scale_factor = render_dpi / 72
    render_matrix = fitz.Matrix(scale_factor, scale_factor)
    # PDF is a vector-based file, so it must be rendered to pixels for display or saving in image format.
    # The PDF page is converted (rendered) to a raster image or pixmap object, which contains pixel data 
    # (color, alpha channel, etc.).
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
    # Slightly increase the area around the bounding box.
    PADDING_FACTOR = 0.05
    l = bounding_box["l"]
    t = bounding_box["t"]
    r = bounding_box["r"]
    b = bounding_box["b"]
    pad_x = (r - l) * PADDING_FACTOR
    pad_y = (t - b) * PADDING_FACTOR
    # Expand the box according to PADDING_FACTOR, but be careful not to go beyond the page edge.
    l = max(0, l - pad_x)
    r = min(page_width, r + pad_x)
    t = min(page_height, t + pad_y)
    b = max(0, b - pad_y)

    # Vertically flip coordinates, because the y-coordinate in the PDF coordinate system 
    # (as defined by PyMuPDF/fitz)
    # is measured from bottom up (i.e., origin (0,0) is at the bottom left corner of the page), 
    # while the reference point of our bounding_box coordinates is the top left corner.
    t_corrected = page_height - t
    b_corrected = page_height - b

    l_scaled = l * x_scale
    r_scaled = r * x_scale
    t_scaled = t_corrected * y_scale
    b_scaled = b_corrected * y_scale

    # Draw a semi-transparent pink layer.
    draw.rectangle([l_scaled, t_scaled, r_scaled, b_scaled], fill=(255, 182, 193, 100))

    # Combine the original image and the pink layer.
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

 # Definition of the main, i.e., home HTTP route:
# - If qa_data does not contain any question-answer pairs, display the 'no_qa' page.
# - Otherwise, find the first question-answer pair for which the render_qa_partial 
#   function returns valid HTML. If there are issues, skip the selected pair and move to the next.
# - When valid HTML is available, insert it into the main index.html template.
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

@app.get("/thank-you", response_class=HTMLResponse)
async def thank_you(request: Request):
    if "email" not in request.session:
        return HTMLResponse('<script>window.location.href="/login";</script>')
    return HTMLResponse(thank_you_template.render())

def render_qa_partial(index: int, edit_mode: bool) -> str:
    item = qa_data[index]

    key = item["fileS3Path"]
    fresh_url = get_fresh_presigned_url(key)

    if not fresh_url:
        logging.warning("Cannot generate presigned URL for %s", key)
        return ""

    pdf_path = download_pdf(fresh_url)
    
    if pdf_path is None:
        logging.warning(f"Skipping QA item at index {index} due to PDF download failure.")
        return ""

    image_url = render_pdf_page(pdf_path, item["pageNumber"], item["boundingBox"])
    
    if not image_url:
        logging.warning(f"Skipping QA item at index {index} due to rendering failure.")
        return ""
    
    # Add parameter t(time) so the browser always requests a new copy of the image and does 
    # not use the old one from cache.
    image_url = f"/{image_url}?t={int(time())}"

    if edit_mode:
        # If edit_mode=True is provided, return template with option to enter corrections.
        return qa_item_edit_template.render(
            index=index,
            question=item["question"],
            answer=item["answer"],
            image_url=image_url
        )
    else:
        # Otherwise, return template in read-only mode.
        return qa_item_readonly_template.render(
            index=index,
            question=item["question"],
            answer=item["answer"],
            image_url=image_url
        )

# HTTP route that calls the template in edit mode for question-answer pairs. 
# Through the GET parameter 'index', obtain information about the displayed element. 
# Similar to home(), skip invalid elements until a displayable one is found. When elements run out, redirect the user to thank-you.
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

 # Works similarly to /edit_qa route, but calls the template in read-only mode.
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

# Process user's evaluation of the question-answer pair:
# - If the index is invalid, redirect directly to thank-you.
# - Prepare a 'record' object containing the previous question-answer pair, original metadata, and new values (evaluation, corrections).
# - If evaluation == "skip", mark record["skipped"] = True.
# - If evaluation == "adequate" or "inadequate", set record["evaluation"] accordingly.
# - If evaluation == "corrected", save the corrected question and answer.
# - Also record the entry time in 'record' (i.e., add timestamp).
# - Write 'record' to app_data/feedback.json (append to existing records, not overwrite).
# - When the element is processed, remove it from qa_data.
# - If there are no more elements, display thank-you; otherwise, find the next one.
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
