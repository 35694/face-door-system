# app.py
import os
import base64
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for

# โฟลเดอร์ dataset เดียวกับที่ใช้ train โมเดล
DATASET_DIR = "dataset_faces"

app = Flask(__name__)


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


@app.route("/")
def index():
    """
    หน้าแรก: แสดงรายชื่อคนในบ้าน + จำนวนรูปใบหน้าที่เก็บแล้ว
    """
    ensure_dir(DATASET_DIR)
    people = []

    for name in sorted(os.listdir(DATASET_DIR)):
        person_dir = os.path.join(DATASET_DIR, name)
        if not os.path.isdir(person_dir):
            continue
        image_files = [
            f
            for f in os.listdir(person_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        people.append({"name": name, "count": len(image_files)})

    return render_template("index.html", people=people)


@app.route("/register")
def register():
    """
    หน้าแบบฟอร์มลงทะเบียน + กล้องเว็บแคม
    รับ query param ?name=Anna ไว้เติมชื่ออัตโนมัติได้
    """
    name = request.args.get("name", "").strip()
    return render_template("register.html", initial_name=name)


@app.route("/save_face", methods=["POST"])
def save_face():
    """
    รับรูปจากหน้าเว็บ (Base64) แล้วเซฟเป็นไฟล์
    body: { "name": "...", "imageData": "data:image/png;base64,...." }
    """
    data = request.get_json()
    name = (data.get("name") or "").strip()
    image_data = data.get("imageData")

    if not name:
        return {"ok": False, "error": "No name provided"}, 400
    if not image_data:
        return {"ok": False, "error": "No image data"}, 400

    # สร้างโฟลเดอร์ของคนนี้ เช่น dataset_faces/Puridat
    person_dir = os.path.join(DATASET_DIR, name)
    ensure_dir(person_dir)

    # ตัด prefix 'data:image/png;base64,...'
    try:
        header, b64data = image_data.split(",", 1)
    except ValueError:
        return {"ok": False, "error": "Invalid image data format"}, 400

    img_bytes = base64.b64decode(b64data)

    # ตั้งชื่อไฟล์ด้วย timestamp + running index แบบง่าย ๆ
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{name}_{ts}.png"
    filepath = os.path.join(person_dir, filename)

    with open(filepath, "wb") as f:
        f.write(img_bytes)

    # ส่งคืนจำนวนรูปปัจจุบันของคนนี้ (เพื่ออัปเดต UI)
    count = len(
        [
            f
            for f in os.listdir(person_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )
    return {"ok": True, "filename": filename, "count": count}


if __name__ == "__main__":
    # รัน local: http://127.0.0.1:5000
    app.run(debug=True)
