from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json
import requests
import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import hashlib
from jinja2 import Template
from time import time
from datetime import datetime
import logging

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# Flattened QA data
qa_data_path = Path("app_data/qa_data.json")
feedback_path = Path("app_data/feedback.json")
if not feedback_path.exists():
    feedback_path.write_text("[]", encoding="utf-8")

raw_data = json.loads(qa_data_path.read_text(encoding="utf-8"))
flattened_data = []
for item in raw_data:
    for qa in item["questions_answers"]:
        flattened_data.append({
            "question": qa["question"],
            "answer": qa["answer"],
            "chunkID": item["chunkID"],
            "text": item["text"],
            "fileUrl": item["fileUrl"],
            "pageNumber": item["pageNumber"],
            "boundingBox": item["boundingBox"]
        })
qa_data = flattened_data

# Load templates
index_template = Template(Path("templates/index.html").read_text(encoding="utf-8"))
no_qa_template = Template(Path("templates/no_qa.html").read_text(encoding="utf-8"))
thank_you_template = Template(Path("templates/thank_you.html").read_text(encoding="utf-8"))

# PARTIAL TEMPLATES: read-only QA and edit QA
qa_item_readonly_template = Template(Path("templates/qa_item_readonly.html").read_text(encoding="utf-8"))
qa_item_edit_template = Template(Path("templates/qa_item_edit.html").read_text(encoding="utf-8"))

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# TODO nadaljuj z implementacijo varivalk za manjkajoce datoteke

def download_pdf(url: str) -> Path:
    pdf_hash = hashlib.md5(url.encode()).hexdigest()
    pdf_path = Path(f"static/{pdf_hash}.pdf")
    if not pdf_path.exists():
        try:
            resp = requests.get(url)
            resp.raise_for_status()  # Raise an exception for bad responses
            pdf_path.write_bytes(resp.content)
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading PDF from {url}: {e}")
            return None  # Return None if download fails
    return pdf_path

def render_pdf_page(pdf_path: Path, page_number: int, bounding_box: list) -> str:
    if pdf_path is None:
        return "" # Skip rendering if the PDF isn't available
    
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number - 1)
    except Exception as e:
            logging.error(f"Error opening PDF file {pdf_path}: {e}")
            return ""  # Return empty string to indicate we should skip rendering
    render_dpi = 150
    scale_factor = render_dpi / 72
    render_matrix = fitz.Matrix(scale_factor, scale_factor)
    pix = page.get_pixmap(matrix=render_matrix)
    img_path = Path("static/rendered.png")
    pix.save(img_path)

    if bounding_box:
        img = Image.open(img_path).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        page_width = page.rect.width
        page_height = page.rect.height
        img_width, img_height = img.size
        x_scale = img_width / page_width
        y_scale = img_height / page_height
        PADDING_FACTOR = 0.05
        # TODO This for loop maybe not needed, since we have only one bounding box each time (it
        # is nested though). We could in preprocess.py change to fetch boundingBox[0] instead of just
        # boundingBox.
        for box in bounding_box:
            l = box["l"]
            t = box["t"]
            r = box["r"]
            b = box["b"]
            pad_x = (r - l) * PADDING_FACTOR
            pad_y = (t - b) * PADDING_FACTOR
            l = max(0, l - pad_x)
            r = min(page_width, r + pad_x)
            t = min(page_height, t + pad_y)
            b = max(0, b - pad_y)

            # flip vertically
            t_corrected = page_height - t
            b_corrected = page_height - b

            l_scaled = l * x_scale
            r_scaled = r * x_scale
            t_scaled = t_corrected * y_scale
            b_scaled = b_corrected * y_scale

            draw.rectangle([l_scaled, t_scaled, r_scaled, b_scaled], fill=(255, 182, 193, 100))

        img = Image.alpha_composite(img, overlay)
        img.save(img_path)

    return str(img_path)

@app.get("/", response_class=HTMLResponse)
def home():
    """
    Full-page load:
    - If we still have QA data, render index.html (with the FIRST QA partial injected).
    - If no data left, show the 'no-qa' template.
    """
    # if no items at all, go straight to "no QA" page
    if not qa_data:
        return HTMLResponse(no_qa_template.render())

    # find first item that actually renders
    idx = 0
    partial_html = ""
    while idx < len(qa_data):
        partial_html = render_qa_partial(idx, edit_mode=False)
        if partial_html:
            break
        # failed → log and remove
        logging.info(f"Skipping broken QA at index {idx}")
        qa_data.pop(idx)

    # after skipping, if list is empty → no QA
    if not qa_data:
        return HTMLResponse(no_qa_template.render())

    # render the good one
    final_html = index_template.render(qa_content=partial_html)
    return HTMLResponse(final_html)


@app.get("/no-qa", response_class=HTMLResponse)
def no_qa():
    """
    This is a full-page route for "no more QA".
    We do NOT force a reload again—this is the final template.
    """
    return HTMLResponse(no_qa_template.render())


@app.get("/thank-you", response_class=HTMLResponse)
def thank_you():
    return HTMLResponse(thank_you_template.render())


def render_qa_partial(index: int, edit_mode: bool) -> str:
    """
    Returns *only* the QA snippet (no <html>, no heading).
    If index is out of range, return an empty string or handle externally.
    """
    item = qa_data[index]
    pdf_path = download_pdf(item["fileUrl"].split("#")[0])
    

    if pdf_path is None:
        logging.warning(f"Skipping QA item at index {index} due to PDF download failure.")
        return ""  # Skip the item if PDF download fails

    image_url = render_pdf_page(pdf_path, item["pageNumber"], item.get("boundingBox", []))
    
    if not image_url:  # If the image couldn't be rendered
        logging.warning(f"Skipping QA item at index {index} due to rendering failure.")
        return ""  # Skip the item if rendering fails

    image_url = f"/{image_url}?t={int(time())}"

    if edit_mode:
        return qa_item_edit_template.render(
            index=index,
            question=item["question"],
            answer=item["answer"],
            image_url=image_url
        )
    else:
        return qa_item_readonly_template.render(
            index=index,
            question=item["question"],
            answer=item["answer"],
            image_url=image_url
        )


@app.get("/edit_qa", response_class=HTMLResponse)
def edit_qa(index: int):
    # skip broken entries
    while index < len(qa_data):
        partial = render_qa_partial(index, edit_mode=True)
        if partial:
            return HTMLResponse(partial)
        logging.info(f"Skipping broken QA at index {index}")
        qa_data.pop(index)

    # no more → no-qa page
    return HTMLResponse('<script>window.location.href="/no-qa";</script>')


@app.get("/display_qa", response_class=HTMLResponse)
def display_qa(index: int):
    # skip until we find a good one
    while index < len(qa_data):
        partial = render_qa_partial(index, edit_mode=False)
        if partial:
            return HTMLResponse(partial)
        logging.info(f"Skipping broken QA at index {index}")
        qa_data.pop(index)

    # none left → go to no-qa
    return HTMLResponse('<script>window.location.href="/no-qa";</script>')

@app.post("/evaluate", response_class=HTMLResponse)
def evaluate(
    index: int = Form(...),
    evaluation: str = Form(...),
    correctedQuestion: str = Form(None),
    correctedAnswer: str = Form(None)
):
    if index < 0 or index >= len(qa_data):
        # out of range => no QA left
        return HTMLResponse('<script>window.location.href="/no-qa";</script>')

    item = qa_data[index]

    # Add a 'timestamp' field:
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    record = {
        "question": item["question"],
        "answer": item["answer"],
        "chunkID": item["chunkID"],
        "text": item["text"],
        "evaluation": None,
        "correctedQuestion": None,
        "correctedAnswer": None,
        "skipped": False,
        "timestamp": current_timestamp
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

    # append to feedback
    existing = json.loads(feedback_path.read_text(encoding="utf-8"))
    existing.append(record)
    feedback_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

    # pop from in-memory
    qa_data.pop(index)

    # If no more QA => force full page load of no-qa
    if not qa_data:
        return HTMLResponse('<script>window.location.href="/no-qa";</script>')

    next_idx = index
    while next_idx < len(qa_data):
        partial = render_qa_partial(next_idx, edit_mode=False)
        if partial:
            return HTMLResponse(partial)
        logging.info(f"Skipping broken QA at index {next_idx}")  # only logged
        qa_data.pop(next_idx)

    # If we exhausted the list during skipping, go to no-qa
    return HTMLResponse('<script>window.location.href="/no-qa";</script>')
