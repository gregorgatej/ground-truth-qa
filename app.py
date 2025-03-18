from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json
import requests
import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import hashlib
from jinja2 import Template

app = FastAPI()

# Mount the static/ directory for images/CSS/etc.
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the QA data from JSON
qa_data_path = Path("data/processed_qa_data.json")
qa_data = json.loads(qa_data_path.read_text(encoding="utf-8"))

# Read the partial template (QA item)
qa_template = Template(Path("templates/qa_item.html").read_text(encoding="utf-8"))
no_qa_template = Template(Path("templates/no_qa.html").read_text(encoding="utf-8"))

def download_pdf(url: str) -> Path:
    """Downloads and caches the PDF file in static/."""
    pdf_hash = hashlib.md5(url.encode()).hexdigest()
    pdf_path = Path(f"static/{pdf_hash}.pdf")

    if not pdf_path.exists():
        resp = requests.get(url)
        pdf_path.write_bytes(resp.content)

    return pdf_path


def render_pdf_page(pdf_path: Path, page_number: int, bounding_boxes: list) -> str:
    """
    Renders a PDF page to PNG and applies bounding boxes.
    Returns the path to the final PNG (in static/).
    """
    # Open PDF
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number - 1)  # zero-indexed

    # Use high DPI for rendering
    render_dpi = 150
    scale_factor = render_dpi / 72
    render_matrix = fitz.Matrix(scale_factor, scale_factor)
    pix = page.get_pixmap(matrix=render_matrix)

    # Save rendered image
    img_path = Path("static/rendered.png")
    pix.save(img_path)

    if bounding_boxes:
        # Re-open the saved PNG with Pillow
        img = Image.open(img_path).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        page_width = page.rect.width
        page_height = page.rect.height

        # Convert PDF coordinates to image coordinates
        img_width, img_height = img.size
        x_scale = img_width / page_width
        y_scale = img_height / page_height

        PADDING_FACTOR = 0.05  # 5% padding

        for box in bounding_boxes:
            l = box["l"]
            t = box["t"]
            r = box["r"]
            b = box["b"]

            # Compute padding (5% of width and height)
            pad_x = (r - l) * PADDING_FACTOR
            pad_y = (t - b) * PADDING_FACTOR

            # Expand the bounding box with padding while ensuring it stays within page limits
            l = max(0, l - pad_x)
            r = min(page_width, r + pad_x)
            t = min(page_height, t + pad_y)
            b = max(0, b - pad_y)


            # Flip vertically, since PDF origin is bottom-left
            t_corrected = page_height - t
            b_corrected = page_height - b

            l_scaled = l * x_scale
            r_scaled = r * x_scale
            t_scaled = t_corrected * y_scale
            b_scaled = b_corrected * y_scale


            # Light pink overlay
            draw.rectangle([l_scaled, t_scaled, r_scaled, b_scaled],
                           fill=(255, 182, 193, 100))

        # Combine overlay with original
        img = Image.alpha_composite(img, overlay)
        img.save(img_path)

    return str(img_path)


@app.get("/", response_class=HTMLResponse)
async def home():
    """
    Immediately render and display the first QA item, or a 'no more QA' message.
    Embeds the rendered QA into templates/index.html.
    """
    if not qa_data:
        return HTMLResponse(no_qa_template.render())

    # Pop the first QA
    qa_item = qa_data.pop(0)
    pdf_path = download_pdf(qa_item["documentUrl"].split("#")[0])
    image_url = render_pdf_page(pdf_path, qa_item["pageNumber"], qa_item.get("boundingBoxes", []))

    # Render QA item partial
    rendered_qa = qa_template.render(
        rationale=qa_item["rationale"],
        question=qa_item["questions_answers"][0]["question"],
        answer=qa_item["questions_answers"][0]["answer"],
        image_url=f"/{image_url}"
    )

    # Read index.html as a Jinja template, inject the QA content
    index_template_text = Path("templates/index.html").read_text(encoding="utf-8")
    index_template = Template(index_template_text)
    final_html = index_template.render(qa_content=rendered_qa)

    return HTMLResponse(content=final_html, status_code=200)


@app.get("/get_next_qa", response_class=HTMLResponse)
async def get_next_qa():
    """Serve next QA pair or trigger a full page reload to /no-qa when empty."""
    if not qa_data:
        return HTMLResponse(
            '<script>window.location.href="/no-qa";</script>', status_code=200
        )  #Forces a full browser reload

    qa_item = qa_data.pop(0)
    pdf_path = download_pdf(qa_item["documentUrl"].split("#")[0])
    image_url = render_pdf_page(pdf_path, qa_item["pageNumber"], qa_item.get("boundingBoxes", []))
    
    from time import time
    image_url = f"/{image_url}?t={int(time())}"  # Prevents caching issues


    return qa_template.render(
        rationale=qa_item["rationale"],
        question=qa_item["questions_answers"][0]["question"],
        answer=qa_item["questions_answers"][0]["answer"],
        image_url=image_url
    )


@app.get("/thank-you", response_class=HTMLResponse)
async def thank_you():
    """Renders the thank-you page when 'Zaključi' is clicked."""
    thank_you_template = Path("templates/thank_you.html").read_text(encoding="utf-8")
    return HTMLResponse(content=thank_you_template, status_code=200)

@app.get("/no-qa", response_class=HTMLResponse)
async def no_qa():
    """Serve the no QA pairs page."""
    no_qa_template = Path("templates/no_qa.html").read_text(encoding="utf-8")
    return HTMLResponse(content=no_qa_template, status_code=200)
