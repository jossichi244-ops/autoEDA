import base64
import os
import re
import easyocr
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from difflib import SequenceMatcher
from pdf2image import convert_from_bytes
from docx import Document
from io import BytesIO
from paddleocr import TextDetection
import pytesseract
from PIL import Image, ImageDraw, ImageFont
from modules.eda import infer_date_format
from mmocr.apis import TextRecInferencer
import mmcv
import mmengine
import traceback
import concurrent.futures
import urllib.request
# from langdetect  import detect
import fasttext
from pdf2docx import Converter
import fitz 

# from transformers import TrOCRProcessor, VisionEncoderDecoderModel

url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
urllib.request.urlretrieve(url, "lid.176.ftz")
model_lang_detect = fasttext.load_model("lid.176.ftz")
print("Tải thành công lid.176.ftz")

print("mmcv version:", mmcv.__version__)
print("mmengine version:", mmengine.__version__)

print("MMOCR inference ready!")

trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

# def load_trocr_models():
#     proc_print = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
#     model_print = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")
#     proc_hand = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
#     model_hand = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
#     return (proc_print, model_print), (proc_hand, model_hand)

# (troc_print, troc_hand) = load_trocr_models()

def load_text_detection_model():
    """Load PaddleOCR (DBNet/DBNet++) for detection only."""
    ocr_detector = TextDetection(model_name="PP-OCRv5_server_det")
    return ocr_detector

text_detector = load_text_detection_model()

def fasttext_detect_lang(text):
    if not text or not text.strip():
        return "unknown"

    text_norm = text.strip()

    # Dự đoán bằng FastText
    try:
        labels, probs = model_lang_detect.predict(text_norm)
        lang = labels[0].replace("__label__", "")
        prob = probs[0]
    except Exception:
        lang, prob = "unknown", 0.0

    # Nếu confidence cao → tin tưởng kết quả
    if prob >= 0.5:
        return lang

    # --- Fallback khi FastText không chắc chắn ---
    clean_text = re.sub(r'[^\w\s]', '', text_norm, flags=re.UNICODE).strip()
    if not clean_text:
        return "unknown"

    # Kiểm tra tiếng Việt: có dấu thanh hoặc chữ "đ/Đ"
    if re.search(r'[àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]', clean_text, re.IGNORECASE):
        return "vi"

    # Nếu chỉ chứa ký tự Latin cơ bản (không dấu) → coi là tiếng Anh
    if re.fullmatch(r'[a-zA-Z0-9\s]+', clean_text):
        return "en"

    # Không xác định
    return "unknown"

def detect_text_regions(image):
    """Detect text boxes using PaddleOCR TextDetection (DBNet v5)."""
    import tempfile
    tmp_path = tempfile.mktemp(suffix=".jpg")
    cv2.imwrite(tmp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    results = text_detector.predict(tmp_path, batch_size=1)
    boxes = []

    if not results:
        print(" Không phát hiện vùng chữ nào.")
        return boxes

    for res in results:
        res.print()

        # Hỗ trợ nhiều kiểu trả về: object, dict, tuple
        if hasattr(res, "res"):
            res_data = res.res
        elif isinstance(res, dict):
            res_data = res
        elif hasattr(res, "__dict__"):
            res_data = res.__dict__
        else:
            print(" Không nhận dạng được kiểu kết quả DBNet:", type(res))
            continue

        # Debug thử khóa có trong res_data
        print("Keys trong res_data:", res_data.keys())

        dt_polys = res_data.get("dt_polys")
        dt_scores = res_data.get("dt_scores", [])

        if dt_polys is None:
            print(" Không có dt_polys trong res_data.")
            continue

        for i, poly in enumerate(dt_polys):
            try:
                points = np.array(poly, dtype=np.int32)
                x_min = int(np.min(points[:, 0]))
                y_min = int(np.min(points[:, 1]))
                x_max = int(np.max(points[:, 0]))
                y_max = int(np.max(points[:, 1]))
                score = float(dt_scores[i]) if i < len(dt_scores) else 1.0
                boxes.append({
                    "bbox": (x_min, y_min, x_max, y_max),
                    "score": score
                })
            except Exception as e:
                print(f" Lỗi khi xử lý polygon: {e}")
                continue

    print(f"Đã phát hiện {len(boxes)} vùng chữ.")
    return boxes


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

def draw_unicode_text(img, text, pos, color=(0, 255, 0), font_path="Roboto-Black.ttf", font_size=18):
    """
    Vẽ chữ Unicode (tiếng Việt có dấu) lên ảnh OpenCV bằng Pillow.
    """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def extract_text(image):
    annotated = image.copy()
    ocr_texts, raw_text_list = [], []

    # --- Load các mô hình OCR ---
    reader = easyocr.Reader(['vi', 'en'], gpu=False)
    rec_inferencer = TextRecInferencer(model='sar')

    # ========== HÀM OCR ==========
    def run_ocr_tesseract(crop):
        try:
            # gray = preprocess_for_ocr(crop)
            config = '--oem 3 --psm 6 -l vie+eng'
            text = pytesseract.image_to_string(crop, config=config).strip()
            text = re.sub(r'\s+', ' ', text)
            conf = 0.5 if text else 0.0
            lang = fasttext_detect_lang(text) if text else "unknown"
            return {"text": text, "conf": conf, "lang": lang}
        except Exception as e:
            print(f"❌ Lỗi Tesseract: {e}")
            return {"text": "", "conf": 0.0}

    def run_ocr_easyocr(crop):
        try:
            # gray = preprocess_for_ocr(crop)
            results = reader.readtext(crop)
            if not results:
                return {"text": "", "conf": 0.0}
            _, text, conf = max(results, key=lambda x: x[2])
            lang = fasttext_detect_lang(text) if text else "unknown"
            return {"text": text, "conf": float(conf),  "lang": lang}
        except Exception as e:
            print(f"❌ Lỗi EasyOCR: {e}")
            return {"text": "", "conf": 0.0}

    def run_ocr_mmocr(crop):
        try:
            # crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            result = rec_inferencer(crop)
            preds = result['predictions'][0]
            text, conf = preds['text'], preds['scores']
            lang = fasttext_detect_lang(text) if text else "unknown"
            return {"text": text, "conf": float(conf), "lang": lang}
        except Exception as e:
            print(f"❌ Lỗi MMOCR: {type(e).__name__} - {e}")
            traceback.print_exc()
            return {"text": "", "conf": 0.0}

    def run_ocr_trocr(crop):
        try:
            # image_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            pixel_values = trocr_processor(images=crop, return_tensors="pt").pixel_values
            with torch.no_grad():
                outputs = trocr_model.generate(
                    pixel_values,
                    output_scores=True,
                    return_dict_in_generate=True
                )
            text = trocr_processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()
            conf = 0.0
            if hasattr(outputs, "scores") and len(outputs.scores) > 0:
                probs = []
                sequence = outputs.sequences[0][1:]
                for score_tensor, token_id in zip(outputs.scores, sequence):
                    token_id = int(token_id)
                    probs.append(F.softmax(score_tensor.squeeze(0), dim=-1)[token_id].item())
                if probs:
                    conf = float(np.mean(probs))
            lang = fasttext_detect_lang(text) if text else "unknown"
            return {"text": text, "conf": conf,"lang": lang}
        except Exception as e:
            print(f"❌ Lỗi TrOCR: {type(e).__name__} - {e}")
            traceback.print_exc()
            return {"text": "", "conf": 0.0}

    # ========== CÔNG CỤ HỖ TRỢ ==========
    def clean_text(t):
        t = re.sub(r'[^0-9A-Za-zÀ-ỹ\s.,:/%()+=-]', '', t)
        t = re.sub(r'\s+', ' ', t).strip()
        return t
    
    def has_good_spacing(text):
        return bool(re.search(r'[A-Za-zÀ-ỹ]+\s+[A-Za-zÀ-ỹ]+', text))

    def normalize_case(text):
        if not text.strip():
            return text

        letters = re.findall(r'[A-Za-zÀ-ỹ]', text)
        if not letters:
            return text  

        upper_count = sum(1 for c in letters if c.isupper())
        lower_count = len(letters) - upper_count

        if upper_count >= 0.7 * len(letters):
            return text.upper()

        elif lower_count >= 0.7 * len(letters):
            return text.lower()
        else:
            return text
        
    def text_similarity(a, b):
        a_clean = re.sub(r'[^A-Za-zÀ-ỹ0-9]', '', a.lower())
        b_clean = re.sub(r'[^A-Za-zÀ-ỹ0-9]', '', b.lower())
        ratio = SequenceMatcher(None, a_clean, b_clean).ratio()
        return ratio * (1 - abs(len(a_clean) - len(b_clean)) / max(len(a_clean), len(b_clean), 1))
 
    def choose_best_text(results):
        texts = [r for r in results.values() if r["text"]]
        for r in texts:
            r["text"] = normalize_case(clean_text(r["text"]))  
        if not texts:
            return "(rỗng)", 0.0, "unknown"

        # --- Lọc ra những kết quả có lang hợp lệ ---
        valid_langs = [t for t in texts if t.get("lang") and t["lang"] != "unknown"]
        pool = valid_langs if valid_langs else texts

        # 🔹 Ưu tiên text có spacing
        spaced = [t for t in pool if has_good_spacing(t["text"])]
        if spaced:
            # Trong nhóm có spacing, chọn conf cao nhất
            best = max(spaced, key=lambda t: t["conf"])
            return best["text"], float(best["conf"]), best["lang"]
        
        # --- Nếu có kết quả hợp lệ ---
        if valid_langs:
            valid_texts = [t["text"] for t in valid_langs]

            all_same = all(text_similarity(valid_texts[0], t) > 0.92 for t in valid_texts[1:])

            if all_same:
                best_valid = max(valid_langs, key=lambda t: t["conf"])
                final_text = normalize_case(best_valid["text"])  # đảm bảo đồng bộ chữ
                return final_text, float(best_valid["conf"]), best_valid["lang"]
            else:
                # --- Nhóm theo độ giống nhau ---
                groups = []
                for t in valid_langs:
                    for t in pool:
                        found = False
                        for g in groups:
                            if text_similarity(g[0]["text"], t["text"]) > 0.88:
                                g.append(t)
                                found = True
                                break
                        if not found:
                            groups.append([t])
                best_group = max(groups, key=lambda g: len(g))
                best = max(best_group, key=lambda t: t["conf"])
                final_text = normalize_case(best["text"])
                final_lang = best["lang"]
                return final_text, float(best["conf"]), final_lang

        # --- Nếu không có lang hợp lệ ---
        groups = []
        for t in texts:
            found = False
            for g in groups:
                if text_similarity(g[0]["text"], t["text"]) > 0.88:
                    g.append(t)
                    found = True
                    break
            if not found:
                groups.append([t])

        best_group = max(groups, key=lambda g: len(g))
        best = max(best_group, key=lambda t: t["conf"])
        final_text = normalize_case(best["text"])

        try:
            lang = fasttext_detect_lang(final_text)
        except:
            lang = "unknown"

        return final_text, float(best["conf"]), lang

    # ========== PHÁT HIỆN VÙNG CHỮ ==========
    boxes = detect_text_regions(image)
    if not boxes:
        print(" Không phát hiện vùng chữ, fallback toàn ảnh.")
        boxes = [(0, 0, image.shape[1], image.shape[0])]

    # ========== CHẠY OCR TRÊN TỪNG VÙNG ==========
    for idx, box in enumerate(boxes, start=1):
        try:
            (x_min, y_min, x_max, y_max) = box if not isinstance(box, dict) else box["bbox"]
            pad = 10
            x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
            x_max, y_max = min(image.shape[1], x_max + pad), min(image.shape[0], y_max + pad)
            crop = image[y_min:y_max, x_min:x_max]

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    "tesseract": executor.submit(run_ocr_tesseract, crop),
                    "easyocr": executor.submit(run_ocr_easyocr, crop),
                    "mmocr": executor.submit(run_ocr_mmocr, crop),
                    "trocr": executor.submit(run_ocr_trocr, crop)
                }
                results = {k: f.result() for k, f in futures.items()}

            # ---- Chọn text tốt nhất ----
            best_text, best_conf, best_lang = choose_best_text(results)

            color = (0, 255, 0) if best_text != "(rỗng)" else (0, 200, 255)
            annotated = draw_unicode_text(annotated, best_text, (x_min, max(0, y_min - 18)), color)
            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, 2)

            ocr_texts.append({
                "bbox": {"x": x_min, "y": y_min, "width": x_max - x_min, "height": y_max - y_min},
                "models": results,
                "final_text": best_text,
                "final_conf": best_conf,
                "lang": best_lang
            })
            raw_text_list.append(best_text)

            print(f"\nVùng {idx}:")
            for name, r in results.items():
                print(f"  {name:<12} → {r['text']} ({r['conf']:.2f}) [lang={r.get('lang','unknown')}]")
            print(f" Chọn: {best_text} ({best_conf:.2f}) [lang={best_lang}]")

        except Exception as e:
            print(f" Lỗi xử lý vùng {idx}: {e}")
            traceback.print_exc()

    print(f"\nHoàn tất OCR: {len(ocr_texts)} vùng.")
    return " ".join(raw_text_list), ocr_texts, annotated

def process_image(content: bytes):
    """Process image bytes and return OCR result."""
    img = Image.open(BytesIO(content)).convert("RGB")
    img_array = np.array(img)
    text, details, annotated = extract_text(img_array)

    # Encode annotated image as base64 for JSON response
    _, buffer = cv2.imencode(".jpg", annotated)
    annotated_b64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "text": text,
        "details": details,
        "annotated_image": annotated_b64  
    }

def process_pdf(content: bytes):
    """Hybrid PDF processor:
    - Mỗi trang kiểm tra: có text thật không?
      → Nếu có: lấy text đó.
      → Nếu không: chạy OCR.
    - Có thể trộn text thật và OCR cùng lúc.
    """
    all_text = ""
    all_details = []
    annotated_previews = []

    try:
        with fitz.open(stream=content, filetype="pdf") as doc:
            for i, page in enumerate(doc):
                page_text = page.get_text("text").strip()
                text_result = ""
                details_result = []

                if page_text:
                    # ✅ Có text thật — lấy trực tiếp
                    print(f"📄 Trang {i+1}: có text thật.")
                    text_result = page_text
                else:
                    # 🔍 Không có text — chạy OCR
                    print(f"🖼️ Trang {i+1}: không có text → OCR...")
                    pix = page.get_pixmap(dpi=200)
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

                    text_result, details_result, annotated = extract_text(img)

                    if i == 0:  # chỉ preview trang đầu
                        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                        annotated_previews.append(base64.b64encode(buffer).decode('utf-8'))

                all_text += f"\n--- Page {i+1} ---\n{text_result}"
                all_details.extend(details_result)

    except Exception as e:
        print("❌ Lỗi khi xử lý PDF:", e)
        traceback.print_exc()

    return {
        "text": all_text.strip(),
        "details": all_details,
        "preview": annotated_previews[0] if annotated_previews else None
    }

def pdf_to_images_no_poppler(content: bytes, dpi=200):
    """Chuyển PDF sang ảnh mà không cần Poppler."""
    images = []
    with fitz.open(stream=content, filetype="pdf") as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            images.append(img)
    return images


def process_pdf_ocr(content: bytes):
    """Fallback OCR pipeline cho PDF scan."""
   
    pages = pdf_to_images_no_poppler(content)

    all_text, all_details, annotated_previews = "", [], []

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
            print(f" OCR error on page {i+1}: {e}")
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
