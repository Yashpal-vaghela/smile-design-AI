import os
import base64
import uuid
from io import BytesIO
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render
from PIL import Image
import numpy as np
import cv2

def index(request):
    return render(request, "index.html", {})

def save_image_to_media(img_pil, filename):
    media_root = Path(settings.MEDIA_ROOT)
    media_root.mkdir(parents=True, exist_ok=True)
    path = media_root / filename
    img_pil.save(path)
    return str(path)

def auto_brightness_and_color_correction_cv(img_bgr):
    # Convert to LAB and apply CLAHE on L-channel for brightness/contrast
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    img_clahe = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # Simple white balance (Gray World)
    result = img_clahe.astype(np.float32)
    avg_b = np.mean(result[:,:,0])
    avg_g = np.mean(result[:,:,1])
    avg_r = np.mean(result[:,:,2])
    avg = (avg_b + avg_g + avg_r) / 3.0
    result[:,:,0] *= (avg / avg_b)
    result[:,:,1] *= (avg / avg_g)
    result[:,:,2] *= (avg / avg_r)
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Optional denoise (fast)
    result = cv2.fastNlMeansDenoisingColored(result, None, 10, 10, 7, 21)
    return result

def pil_from_cv2(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def cv2_from_pil(img_pil):
    arr = np.array(img_pil.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def upload_image(request):
    """
    Accepts multipart-form 'image' (file) or base64 in 'image_base64'.
    Returns JSON with 'processed_base64' (data:image/png;base64,...)
    """
    if request.method != "POST":
        return HttpResponseBadRequest("POST only")

    # Accept file upload
    uploaded = request.FILES.get("image")
    image_base64 = request.POST.get("image_base64")

    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
    elif image_base64:
        header, data = image_base64.split(",", 1) if "," in image_base64 else (None, image_base64)
        decoded = base64.b64decode(data)
        img_pil = Image.open(BytesIO(decoded)).convert("RGB")
    else:
        return HttpResponseBadRequest("No image provided")

    # Save original
    uid = uuid.uuid4().hex[:12]
    orig_name = f"{uid}_orig.jpg"
    save_image_to_media(img_pil, orig_name)

    # Convert to cv2
    cv2_img = cv2_from_pil(img_pil)

    # Preprocess: brightness & color correction
    processed = auto_brightness_and_color_correction_cv(cv2_img)

    # Convert processed back to PIL and save
    processed_pil = pil_from_cv2(processed)
    processed_name = f"{uid}_processed.jpg"
    processed_path = save_image_to_media(processed_pil, processed_name)

    # Encode processed image to base64 to return for immediate preview
    buffer = BytesIO()
    processed_pil.save(buffer, format="JPEG", quality=90)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    data_url = "data:image/jpeg;base64," + b64

    return JsonResponse({
        "success": True,
        "processed_url": settings.MEDIA_URL + processed_name,
        "processed_base64": data_url,
    })
