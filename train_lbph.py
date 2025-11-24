#!/usr/bin/env python3
"""
train_lbph.py

อ่านรูปใบหน้าจากโฟลเดอร์:
    dataset_faces/<person_name>/*.jpg, *.png

- ใช้ Haar cascade (ผ่าน vision.face_detect.detect_faces) ตัดเฉพาะใบหน้า
- resize ใบหน้าให้เป็น IMAGE_SIZE
- เทรนโมเดล LBPH ของ OpenCV
- เซฟ:
    models/lbph_model.xml
    models/label_map.json
"""

import os
import json
import argparse
from collections import defaultdict

import cv2
import numpy as np

# ใช้ detector เดียวกับตอน run จริง
from vision.face_detect import detect_faces

# ดึงค่าจาก config.py (ถ้าไม่มี จะใช้ค่า default ด้านล่าง)
try:
    import config

    DEFAULT_DATASET_DIR = getattr(config, "DATASET_DIR", "dataset_faces")
    DEFAULT_MODEL_PATH = getattr(config, "LBPH_MODEL_PATH", "models/lbph_model.xml")
    DEFAULT_LABEL_MAP_PATH = getattr(config, "LABEL_MAP_PATH", "models/label_map.json")
    DEFAULT_IMAGE_SIZE = getattr(config, "IMAGE_SIZE", (200, 200))
except ImportError:
    DEFAULT_DATASET_DIR = "dataset_faces"
    DEFAULT_MODEL_PATH = "models/lbph_model.xml"
    DEFAULT_LABEL_MAP_PATH = "models/label_map.json"
    DEFAULT_IMAGE_SIZE = (200, 200)


def gather_images_and_labels(dataset_dir, image_size=(200, 200), min_samples_per_person=5):
    """
    อ่านรูปจาก dataset_dir แล้ว:
      - ใช้ detect_faces หาหน้าในแต่ละรูป
      - เลือกใบหน้าที่ใหญ่ที่สุด
      - crop + resize เป็น image_size
    แล้วคืนค่า:
      faces, labels, label_map, per_person_counts
    """
    faces = []
    labels = []
    label_map = {}
    name_to_label = {}
    per_person_counts = defaultdict(int)
    current_label = 0

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    persons = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    if not persons:
        raise RuntimeError(f"No person folders found inside {dataset_dir}")

    print("🔎 Found persons:")
    for person in persons:
        print(f"  - {person}")

    for person in persons:
        person_dir = os.path.join(dataset_dir, person)
        img_files = sorted([
            f for f in os.listdir(person_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if len(img_files) < min_samples_per_person:
            print(
                f"⚠️  Skip '{person}' : only {len(img_files)} images "
                f"(min required = {min_samples_per_person})"
            )
            continue

        name_to_label[person] = current_label
        label_map[current_label] = person

        for img_name in img_files:
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️  Cannot read image: {img_path} (skipped)")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # หาหน้าในรูป (ใช้ detector เดียวกับตอน runtime)
            detected = detect_faces(gray)
            if len(detected) == 0:
                print(f"⚠️  No face found in {img_path} (skipped)")
                continue

            # เอาใบหน้าที่ใหญ่ที่สุด
            x, y, w, h = sorted(detected, key=lambda f: f[2] * f[3], reverse=True)[0]
            face_roi = gray[y:y + h, x:x + w]

            try:
                resized = cv2.resize(face_roi, image_size)
            except Exception as e:
                print(f"⚠️  Resize failed for {img_path}: {e} (skipped)")
                continue

            faces.append(resized)
            labels.append(current_label)
            per_person_counts[person] += 1

        current_label += 1

    if not faces:
        raise RuntimeError("No valid face images found for training.")

    return faces, labels, label_map, per_person_counts


def compute_training_accuracy(model, faces, labels, label_map):
    """วัด accuracy บน training set (ช่วยเช็คว่าโมเดลพอใช้ได้ไหม)"""
    preds = []
    for face in faces:
        label_pred, conf = model.predict(face)
        preds.append(label_pred)

    labels = np.array(labels, dtype=np.int32)
    preds = np.array(preds, dtype=np.int32)

    acc = float((preds == labels).sum()) / len(labels)

    print(f"\n📊 Training accuracy (บน training set): {acc * 100:.2f}%")
    print("   (ค่านี้แค่ช่วยเช็คคร่าว ๆ ยังไม่ใช่การทดสอบจริง)")

    print("\nLabel mapping:")
    for lbl, name in label_map.items():
        print(f"  {lbl} -> {name}")

    return acc


def train_lbph(dataset_dir, model_path, labels_path,
               image_size=(200, 200), min_samples_per_person=5):
    print("== TRAIN LBPH FACE RECOGNIZER ==")
    print(f"📂 Dataset dir : {dataset_dir}")
    print(f"💾 Model path  : {model_path}")
    print(f"💾 Label map   : {labels_path}")
    print(f"🖼  Image size  : {image_size[0]}x{image_size[1]}")
    print(f"👥 Min samples per person: {min_samples_per_person}")

    faces, labels, label_map, per_person_counts = gather_images_and_labels(
        dataset_dir, image_size=image_size, min_samples_per_person=min_samples_per_person
    )

    print("\n✅ Using persons for training:")
    for name, count in per_person_counts.items():
        print(f"  - {name}: {count} images")

    # สร้างโมเดล LBPH
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,
            neighbors=8,
            grid_x=8,
            grid_y=8
        )
    except Exception as e:
        raise RuntimeError(
            "Cannot create LBPHFaceRecognizer. "
            "ตรวจสอบว่าได้ติดตั้ง opencv-contrib-python แล้วหรือยัง"
        ) from e

    # เทรน
    print("\n🚀 Training...")
    recognizer.train(faces, np.array(labels, dtype=np.int32))
    print("✅ Training finished.")

    # สร้างโฟลเดอร์ models ถ้ายังไม่มี
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    # เซฟโมเดล
    recognizer.save(model_path)
    print(f"💾 Saved model to: {model_path}")

    # เซฟ label map เป็น JSON
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({int(k): v for k, v in label_map.items()}, f,
                  ensure_ascii=False, indent=2)
    print(f"💾 Saved label map to: {labels_path}")

    # ดู accuracy บน training set เล็กน้อย
    compute_training_accuracy(recognizer, faces, labels, label_map)

    print("\n🎉 Done. ตอนนี้คุณสามารถใช้โมเดลนี้ร่วมกับ main.py / face_recognize.py ได้แล้ว")


def parse_args():
    parser = argparse.ArgumentParser(description="Train LBPH model from dataset_faces.")
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET_DIR,
        help=f"Dataset directory (default: {DEFAULT_DATASET_DIR})"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_PATH,
        help=f"Output model path (default: {DEFAULT_MODEL_PATH})"
    )
    parser.add_argument(
        "--labels", default=DEFAULT_LABEL_MAP_PATH,
        help=f"Output label map path (default: {DEFAULT_LABEL_MAP_PATH})"
    )
    parser.add_argument(
        "--size", type=int, default=DEFAULT_IMAGE_SIZE[0],
        help=f"Resize face to SIZE x SIZE (default: {DEFAULT_IMAGE_SIZE[0]})"
    )
    parser.add_argument(
        "--min-samples", type=int, default=5,
        help="Minimum images per person (default: 5)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    img_size = (args.size, args.size)
    train_lbph(
        dataset_dir=args.dataset,
        model_path=args.model,
        labels_path=args.labels,
        image_size=img_size,
        min_samples_per_person=args.min_samples,
    )
