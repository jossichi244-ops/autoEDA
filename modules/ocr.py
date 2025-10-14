import base64
import re
import easyocr
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from difflib import SequenceMatcher
from pdf2image import convert_from_bytes
from docx import Document
from io import BytesIO

from modules.eda import infer_date_format

def load_trocr_models():
    proc_print = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
    model_print = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")
    proc_hand = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model_hand = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    return (proc_print, model_print), (proc_hand, model_hand)
(troc_print, troc_hand) = load_trocr_models()

def infer_schema_from_ocr_fields(extracted: dict):

    fields = extracted.get("fields", [])
    if not fields:
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "SmartGeneratedSchema",
            "type": "object",
            "properties": {}
        }

    df = pd.DataFrame(fields)
    if "field" not in df or "value" not in df:
        return {}

    # Gom nhóm giá trị theo tên trường
    grouped = df.groupby("field")["value"].apply(list).to_dict()
    props = {}

    for field, values in grouped.items():
        clean_values = [v for v in values if str(v).strip()]
        if not clean_values:
            continue

        series = pd.Series(clean_values).astype(str)
        dtype = "string"
        fmt = None

        # 🔹 Nhận dạng kiểu dữ liệu
        # → 1️⃣ integer
        if series.str.match(r"^-?\d+$").all():
            dtype = "integer"

        # → 2️⃣ float / number
        elif series.str.match(r"^-?\d+(\.\d+)?$").all():
            dtype = "number"

        # → 3️⃣ boolean
        elif series.str.lower().isin(["yes", "no", "true", "false", "có", "không"]).all():
            dtype = "boolean"

        # → 4️⃣ datetime (phát hiện bằng infer_date_format)
        else:
            detected_fmt = infer_date_format(series)
            if detected_fmt:
                dtype = "string"
                fmt = "date-time"
            else:
                dtype = "string"

        # 🔹 Tạo schema property
        prop = {"type": dtype}
        if fmt:
            prop["format"] = fmt
            # pattern gợi ý: giúp tái tạo mẫu định dạng nếu cần
            prop["pattern"] = detected_fmt

        props[field] = prop

    # 🔹 Kết quả schema JSON
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "SmartGeneratedSchema",
        "type": "object",
        "properties": props,
        "required": list(props.keys())  # có thể tùy chỉnh: chỉ required với field có dữ liệu
    }

    return schema

def classify_text_region(crop_img, ocr_text):
    pil = Image.fromarray(crop_img).convert("RGB")

    def infer(proc, mod):
        inputs = proc(images=pil, return_tensors="pt").pixel_values
        ids = mod.generate(inputs)
        return proc.batch_decode(ids, skip_special_tokens=True)[0]

    try:
        txt_p = infer(*troc_print)
    except:
        txt_p = ""
    try:
        txt_h = infer(*troc_hand)
    except:
        txt_h = ""

    sim = lambda a, b: SequenceMatcher(None, a, b).ratio()
    score_p = sim(txt_p, ocr_text)
    score_h = sim(txt_h, ocr_text)

    return "printed" if score_p >= score_h else "handwritten"

def extract_text(image):
    reader = easyocr.Reader(['vi', 'en'], gpu=False)
    results = reader.readtext(image)

    ocr_texts = []
    raw_text_list = []
    annotated = image.copy()

    for (bbox, text, conf) in results:
        if conf > 0.3:
            x_min = int(min([pt[0] for pt in bbox]))
            y_min = int(min([pt[1] for pt in bbox]))
            x_max = int(max([pt[0] for pt in bbox]))
            y_max = int(max([pt[1] for pt in bbox]))
            crop = image[y_min:y_max, x_min:x_max]

            label = classify_text_region(crop, text)
            color = (0, 255, 0) if label == "printed" else (0, 0, 255)

            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(annotated, f"{label}: {text}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            ocr_texts.append({
                "text": text,
                "confidence": conf,
                "bbox": {"x": x_min, "y": y_min, "width": x_max - x_min, "height": y_max - y_min},
                "type": label
            })
            raw_text_list.append(f"[{label.upper()}] {text}")

    return " ".join(raw_text_list), ocr_texts, annotated

def process_image(content: bytes):
    """Process image bytes and return OCR result."""
    img = Image.open(BytesIO(content)).convert("RGB")
    img_array = np.array(img)
    text, details, annotated = extract_text(img_array)

    # Encode annotated image as base64 for JSON response
    _, buffer = cv2.imencode(".jpg", annotated)
    annotated_b64 = np.array(buffer).tobytes()

    return {
        "text": text,
        "details": details,
        "annotated_image": annotated_b64.hex()  # or base64 if frontend needs image
    }

def process_pdf(content: bytes):
    """Extract OCR from PDF bytes (safe version with explicit poppler path)."""
    import os

    poppler_path = r"D:\Program Files\Release-25.07.0-0\poppler-25.07.0\Library\bin"
    print("⚙️ Using Poppler path:", poppler_path)

    if not os.path.exists(poppler_path):
        raise RuntimeError(f"Poppler not found at {poppler_path}")

    try:
        pages = convert_from_bytes(content, poppler_path=poppler_path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"PDF conversion failed: {str(e)}")

    all_text = ""
    all_details = []
    annotated_previews = []

    for i, page in enumerate(pages):
        try:
            img_array = np.array(page)
            text, details, annotated = extract_text(img_array)
            all_text += f"\n--- Page {i+1} ---\n{text}"
            all_details.extend(details)

            if i == 0:
                _, buffer = cv2.imencode(".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                annotated_previews.append(base64.b64encode(buffer).decode('utf-8'))
        except Exception as e:
            print(f"⚠️ OCR error on page {i+1}: {e}")
            traceback.print_exc()

    return {
        "text": all_text.strip(),
        "details": all_details,
        "preview": annotated_previews[0] if annotated_previews else None
    }

def process_docx(content: bytes):
    """Extract text content from DOCX bytes."""
    file_stream = BytesIO(content)
    doc = Document(file_stream)
    full_text = [p.text for p in doc.paragraphs if p.text.strip()]
    return {"text": "\n".join(full_text), "details": []}

def extract_fields_from_text(text: str):
    """Phân tích văn bản OCR, tách các trường có dấu ':'"""
    lines = text.splitlines()
    results = []
    unmatched = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(r"^(.*?)\s*:\s*(.*)$", line)
        if match:
            field_name, value = match.groups()
            results.append({
                "field": field_name.strip(),
                "value": value.strip(),
                "type": None  # để người dùng chỉnh trên giao diện
            })
        else:
            unmatched.append(line)

    return {
        "fields": results,
        "unmatched": unmatched  
    }