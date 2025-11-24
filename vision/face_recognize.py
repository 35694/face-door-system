# vision/face_recognize.py

import os
import json
import cv2
import numpy as np

from config import (
    LBPH_MODEL_PATH,
    LABEL_MAP_PATH,
    IMAGE_SIZE,
)

# ถ้าใน config ยังไม่มี CONF_THRESHOLD ให้ใช้ค่า default = 70
try:
    from config import CONF_THRESHOLD
except ImportError:
    CONF_THRESHOLD = 70


def load_label_map(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label map not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # key จาก JSON จะเป็น string → แปลงเป็น int
    return {int(k): v for k, v in raw.items()}


def load_model():
    """โหลดโมเดล LBPH + label_map"""
    if not os.path.exists(LBPH_MODEL_PATH):
        raise FileNotFoundError(
            f"LBPH model not found: {LBPH_MODEL_PATH}. "
            f"ลองรัน train_lbph.py ก่อน"
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(LBPH_MODEL_PATH)

    label_map = load_label_map(LABEL_MAP_PATH)

    return recognizer, label_map


def preprocess_face(face_gray):
    """resize ใบหน้าให้เป็นขนาดเดียวกับตอน train"""
    return cv2.resize(face_gray, IMAGE_SIZE)


def recognize_face(model, label_map, face_gray):
    """
    รับภาพใบหน้า (gray) แล้วคืนค่า (name, conf)

    - name: ชื่อคน หรือ 'Stranger'
    - conf: ค่าความมั่นใจของ LBPH (ยิ่งต่ำยิ่งมั่นใจ)
    """
    face_resized = preprocess_face(face_gray)

    # model.predict ต้องการภาพ 2D grayscale
    label_pred, conf = model.predict(face_resized)

    name = label_map.get(label_pred, "Unknown")

    # LBPH: ยิ่ง conf ต่ำ → matching ดี
    # ถ้า conf สูงเกิน threshold → ถือว่า Stranger
    if conf > CONF_THRESHOLD or name == "Unknown":
        return "Stranger", conf

    return name, conf
