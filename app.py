from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import shutil
from pathlib import Path

app = FastAPI()

# Direktori untuk menyimpan gambar
input_dir = Path("static/input")
output_dir = Path("static/output")
input_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

# Mount folder static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Template Jinja2
templates = Jinja2Templates(directory="templates")


# Fungsi Dark Channel untuk Dehazing
def dark_channel(image, size=15):
    min_channel = cv2.min(cv2.min(image[:, :, 0], image[:, :, 1]), image[:, :, 2])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(min_channel, kernel)
    return dark

def atmospheric_light(image, dark_channel):
    flat_dark = dark_channel.ravel()
    flat_image = image.reshape((-1, 3))
    num_pixels = len(flat_dark)
    top_percent = int(max(num_pixels * 0.001, 1))
    indices = np.argpartition(flat_dark, -top_percent)[-top_percent:]
    atmo = np.mean(flat_image[indices], axis=0)
    return atmo

def dehaze(image, omega=0.95, size=15):
    # Step 1: Calculate Dark Channel
    dark = dark_channel(image, size)
    atmo = atmospheric_light(image, dark)

    # Step 2: Calculate Transmission Map
    transmission = 1 - omega * dark / atmo.max()
    transmission = np.clip(transmission, 0.1, 1.0)

    # Refine Transmission Map
    guided = cv2.ximgproc.guidedFilter(
        guide=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        src=transmission.astype(np.float32),
        radius=60,
        eps=1e-3,
    )

    # Step 3: Recover Scene Radiance
    restored = np.empty_like(image, dtype=np.float32)
    for i in range(3):
        restored[:, :, i] = (image[:, :, i] - atmo[i]) / guided + atmo[i]
    restored = np.clip(restored, 0, 255).astype(np.uint8)

    # Step 4: Final Adjustment (CLAHE)
    lab = cv2.cvtColor(restored, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge((l, a, b))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced_image


# Fungsi Enhancement untuk Gambar Underwater
def underwater_enhancement(image):
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Enhance luminance using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Adjust color channels
    a = cv2.add(a, 20)
    b = cv2.subtract(b, 20)

    # Merge and convert back to BGR
    enhanced_lab = cv2.merge((l, a, b))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced_image


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "input_image": None, "output_image": None})


@app.post("/", response_class=HTMLResponse)
async def process_image(request: Request, file: UploadFile = File(...), mode: str = Form(...)):
    input_path = input_dir / file.filename
    output_path = output_dir / file.filename

    # Simpan file yang diunggah
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Baca gambar
    image = cv2.imread(str(input_path))
    if image is None:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Invalid image format."})

    # Proses gambar sesuai mode
    if mode == "dehaze":
        result = dehaze(image)
    elif mode == "underwater":
        result = underwater_enhancement(image)
    else:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Invalid mode selected."})

    # Simpan hasilnya
    cv2.imwrite(str(output_path), result)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "input_image": f"/static/input/{file.filename}",
        "output_image": f"/static/output/{file.filename}",
        "error": None,
    })
