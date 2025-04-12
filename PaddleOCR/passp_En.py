import cv2
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import re

# === Load Image ===
image_path = r"C:\Users\2002r\Desktop\WhatsApp Image 2024-10-02 at 20.03.36_626fa63f.jpg"
img = cv2.imread(image_path)
h, w, _ = img.shape

# === General Crop (adjust if needed, or use full image) ===
# For now, try full image to ensure capture
cropped = img
cropped_path = "cropped_passport.jpg"
cv2.imwrite(cropped_path, cropped)

# === Initialize PaddleOCR for English ===
ocr = PaddleOCR(lang='en', use_angle_cls=True)
results = ocr.ocr(cropped_path, cls=True)

# === Process OCR Results ===
print("\n OCR Results (English - Full Passport):")
all_text = ""
passport_number = ""

for line_group in results:
    if not line_group:
        continue
    for box in line_group:
        text, score = box[1]
        text = text.strip().upper()
        all_text += text + " "
        print(f"Text: '{text}' | Score: {round(score, 3)}")

# === Extract Passport Number Using Regex ===
match = re.search(r"\b[A-Z0-9]{8,9}\b", all_text)
if match:
    passport_number = match.group()
    print(f"\n Detected Passport Number:\n Final Passport: {passport_number}")
else:
    print("\n No valid passport number detected.")

# === Save Visual OCR Output ===
image = Image.open(cropped_path).convert("RGB")
boxes, txts, scores = [], [], []
for line_group in results:
    for box in line_group:
        boxes.append(box[0])
        txts.append(box[1][0])
        scores.append(box[1][1])

im_show = draw_ocr(image, boxes, txts, scores, font_path=r"C:\Users\2002r\Desktop\paddleocr\PaddleOCR\simfang.ttf")
Image.fromarray(im_show).save("PassportOCR_result.jpg")
print(" Visual result saved as 'PassportOCR_result.jpg'")
