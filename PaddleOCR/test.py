import cv2
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import re
import numpy as np

# === Convert Arabic-Indic digits to Western ===
def convert_arabic_indic_to_western(text):
    return text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))

# === Fix common misreads of Arabic zero ===
def fix_common_zero_errors(text):
    return text.replace('.', '0').replace('·', '0')


# === Load Image ===
image_path = r"C:\Users\2002r\Desktop\testing.jpg"
img = cv2.imread(image_path)
h, w, _ = img.shape

# === Crop Bottom 35% and Right 60% ===
cropped = img[int(h * 0.65):, int(w * 0.4):]
cropped_path = "cropped_id_zone.jpg"
cv2.imwrite(cropped_path, cropped)

# === Run PaddleOCR (Arabic) ===
ocr = PaddleOCR(lang='arabic', use_angle_cls=True)
results = ocr.ocr(cropped_path, cls=True)

# === Process OCR Results ===
print("\n🔍 OCR Results (Bottom 35% / Right 60%):")
id_number = ""
for line_group in results:
    if not line_group:
        continue
    for box in line_group:
        text, score = box[1]
        text = convert_arabic_indic_to_western(text.strip())
        text = fix_common_zero_errors(text)
        if re.fullmatch(r"\d{5,}", text):
            id_number = text
            print(f"Raw: '{text}' | Score: {round(score, 3)}")

# === Post-process and Append Zeros ===
if id_number:
    id_number += "0000"
    print(f"\n🆔 Detected ID Number:\n✅ Final ID: {id_number}")
else:
    print("\n❌ No valid ID number detected.")

# === Save Visual OCR Output ===
image = Image.open(cropped_path).convert("RGB")
boxes, txts, scores = [], [], []
for line_group in results:
    for box in line_group:
        boxes.append(box[0])
        txts.append(box[1][0])
        scores.append(box[1][1])

im_show = draw_ocr(image, boxes, txts, scores, font_path=r"C:\Users\2002r\Desktop\paddleocr\PaddleOCR\simfang.ttf")
Image.fromarray(im_show).save("PaddleOCR_result.jpg")
print("✅ Visual result saved as 'PaddleOCR_result.jpg'")
