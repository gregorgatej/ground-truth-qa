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

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Flattened QA data, as before
qa_data_path = Path("data/processed_qa_data.json")
feedback_path = Path("data/feedback.json")
if not feedback_path.exists():
    feedback_path.write_text("[]", encoding="utf-8")

raw_data = json.loads(qa_data_path.read_text(encoding="utf-8"))
flattened_data = []
for item in raw_data:
    for qa in item["questions_answers"]:
        flattened_data.append({
            "rationale": item["rationale"],
            "question": qa["question"],
            "answer": qa["answer"],
            "chunk": item["chunk"],
            "suggestedText": item["suggestedText"],
            "documentUrl": item["documentUrl"],
            "pageNumber": item["pageNumber"],
            "boundingBoxes": item["boundingBoxes"]
        })
qa_data = flattened_data

# Load templates
index_template = Template(Path("templates/index.html").read_text(encoding="utf-8"))
no_qa_template = Template(Path("templates/no_qa.html").read_text(encoding="utf-8"))
thank_you_template = Template(Path("templates/thank_you.html").read_text(encoding="utf-8"))

# PARTIAL TEMPLATES: read-only QA and edit QA
qa_item_readonly_template = Template(Path("templates/qa_item_readonly.html").read_text(encoding="utf-8"))
qa_item_edit_template = Template(Path("templates/qa_item_edit.html").read_text(encoding="utf-8"))


def download_pdf(url: str) -> Path:
    pdf_hash = hashlib.md5(url.encode()).hexdigest()
    pdf_path = Path(f"static/{pdf_hash}.pdf")
    if not pdf_path.exists():
        resp = requests.get(url)
        pdf_path.write_bytes(resp.content)
    return pdf_path


def render_pdf_page(pdf_path: Path, page_number: int, bounding_boxes: list) -> str:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number - 1)
    render_dpi = 150
    scale_factor = render_dpi / 72
    render_matrix = fitz.Matrix(scale_factor, scale_factor)
    pix = page.get_pixmap(matrix=render_matrix)
    img_path = Path("static/rendered.png")
    pix.save(img_path)

    if bounding_boxes:
        img = Image.open(img_path).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        page_width = page.rect.width
        page_height = page.rect.height
        img_width, img_height = img.size
        x_scale = img_width / page_width
        y_scale = img_height / page_height
        PADDING_FACTOR = 0.05
        for box in bounding_boxes:
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
    if not qa_data:
        return HTMLResponse(no_qa_template.render())

    # We have QA: render the entire layout once with the first item
    partial_html = render_qa_partial(0, edit_mode=False)
    # Now inject that partial into the main layout
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
    pdf_path = download_pdf(item["documentUrl"].split("#")[0])
    image_url = render_pdf_page(pdf_path, item["pageNumber"], item.get("boundingBoxes", []))
    image_url = f"/{image_url}?t={int(time())}"

    if edit_mode:
        return qa_item_edit_template.render(
            index=index,
            rationale=item["rationale"],
            question=item["question"],
            answer=item["answer"],
            image_url=image_url
        )
    else:
        return qa_item_readonly_template.render(
            index=index,
            rationale=item["rationale"],
            question=item["question"],
            answer=item["answer"],
            image_url=image_url
        )


@app.get("/edit_qa", response_class=HTMLResponse)
def edit_qa(index: int):
    if index < 0 or index >= len(qa_data):
        # no more QA => full page load of /no-qa
        return HTMLResponse('<script>window.location.href="/no-qa";</script>')
    partial_html = render_qa_partial(index, edit_mode=True)
    return HTMLResponse(partial_html)


@app.get("/display_qa", response_class=HTMLResponse)
def display_qa(index: int):
    if index < 0 or index >= len(qa_data):
        return HTMLResponse('<script>window.location.href="/no-qa";</script>')
    partial_html = render_qa_partial(index, edit_mode=False)
    return HTMLResponse(partial_html)


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
    record = {
        "rationale": item["rationale"],
        "question": item["question"],
        "answer": item["answer"],
        "chunk": item["chunk"],
        "suggestedText": item["suggestedText"],
        "evaluation": None,
        "correctedQuestion": None,
        "correctedAnswer": None,
        "skipped": False
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

    # Otherwise, return the next QA partial (same index) in read-only mode
    # because after popping, the "next" item is now at the same index
    return HTMLResponse(render_qa_partial(index, edit_mode=False))
