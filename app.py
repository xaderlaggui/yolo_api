from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import base64
import numpy as np
from PIL import Image
import os
from pyngrok import ngrok
import threading

app = Flask(__name__)

# Load YOLO model once
model = YOLO("best.pt")  # make sure 'best.pt' is in the same folder

# ---------- Routes ----------

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Braille YOLO API is running."})

@app.route("/predict", methods=["POST"])
def predict():
    """Handles single image uploads (camera or gallery)."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image = Image.open(file.stream).convert("RGB")
    results = model.predict(image, save=True, conf=0.1)

    boxes = results[0].boxes
    letters = []

    # Extract letters with bounding boxes
    for box, cls_idx in zip(boxes.xyxy, boxes.cls):
        raw_class = results[0].names[int(cls_idx)]
        letter = ''.join([c for c in raw_class if c.isalpha()])
        x1, y1, x2, y2 = box.cpu().numpy()
        letters.append({'letter': letter, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

    if not letters:
        predicted_text = "No letters detected"
    else:
        # Remove duplicates via overlap filtering
        filtered = []
        for l in letters:
            if not any(
                max(0, min(l['x2'], f['x2']) - max(l['x1'], f['x1'])) *
                max(0, min(l['y2'], f['y2']) - max(l['y1'], f['y1'])) / 
                min((l['x2']-l['x1'])*(l['y2']-l['y1']), (f['x2']-f['x1'])*(f['y2']-f['y1'])) > 0.5
                for f in filtered
            ):
                filtered.append(l)
        letters = filtered

        # Group by line (y distance)
        lines = []
        for letter in letters:
            added = False
            for line in lines:
                if abs(letter['y1'] - line[0]['y1']) < 25:
                    line.append(letter)
                    added = True
                    break
            if not added:
                lines.append([letter])

        # Sort letters & lines
        for line in lines:
            line.sort(key=lambda l: l['x1'])
        lines.sort(key=lambda line: sum(l['y1'] for l in line)/len(line))

        # Build full text
        paragraph = ""
        for line in lines:
            prev_x = None
            for l in line:
                if prev_x and (l['x1'] - prev_x) > 5:
                    paragraph += " "
                paragraph += l['letter']
                prev_x = l['x1']
            paragraph += "\n"
        predicted_text = paragraph.strip()

    # Convert result image to base64
    result_dir = results[0].save_dir
    saved_files = [f for f in os.listdir(result_dir) if f.endswith(('.jpg', '.png'))]
    if saved_files:
        img_path = os.path.join(result_dir, saved_files[0])
        img_cv = cv2.imread(img_path)
        _, buffer = cv2.imencode('.jpg', img_cv)
        img_b64 = base64.b64encode(buffer).decode("utf-8")
    else:
        img_b64 = ""

    return jsonify({
        "predicted_text": predicted_text,
        "result_image": img_b64
    })

@app.route("/predict_realtime", methods=["POST"])
def predict_realtime():
    """Handles frame-by-frame real-time prediction."""
    if "image" not in request.files:
        return jsonify({"error": "No frame provided"}), 400

    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    results = model.predict(frame, conf=0.25, imgsz=640)
    annotated = results[0].plot()

    # Convert annotated frame + text to response
    _, buffer = cv2.imencode('.jpg', annotated)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    boxes = results[0].boxes
    text = ""
    if boxes:
        detected = [results[0].names[int(cls)] for cls in boxes.cls]
        text = "".join(detected)
    else:
        text = "No letters detected"

    return jsonify({
        "result_image": encoded_image,
        "predicted_text": text
    })

# ---------- ngrok ----------
def start_ngrok():
    url = ngrok.connect(5000)
    print(" * ngrok tunnel URL:", url)

# ---------- Main ----------
if __name__ == "__main__":
    threading.Thread(target=start_ngrok, daemon=True).start()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
