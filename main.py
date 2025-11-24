# main.py
"""
Face Door System - เวอร์ชันปรับให้ทำงานลื่นขึ้นบน Mac

ปรับอะไรบ้าง:
- ลด resolution กล้องเป็น 640x480
- ย่อภาพ grayscale ลง (เช่น 0.5x) ก่อนส่งเข้า detect_faces แล้วค่อย scale กลับ
- detect หน้าเฉพาะทุก N เฟรม (เช่น ทุก 3 เฟรม)
- ตัด while loop ฝั่ง Stranger ออก (ไม่ block loop หลัก)
"""

import cv2
import os
from datetime import datetime

from config import (
    CAMERA_INDEX,
    STRANGER_TIMEOUT,   # ยังเผื่ออนาคต ถ้าจะทำ logic รอแบบไม่ block
    DEBUG,
    MSG_STRANGER_DETECTED,
    MSG_DOOR_OPENED,
)

from vision.face_detect import detect_faces
from vision.face_recognize import load_model, recognize_face
from hardware.door_control_mock import (
    door_unlock,
    stranger_alert,
    is_scan_button_pressed,   # ตอนนี้ยังไม่ใช้ แต่เก็บไว้
)
from line.line_notify import send_line_message


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_stranger_image(frame) -> str:
    folder = "strangers"
    ensure_dir(folder)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"stranger_{ts}.jpg"
    path = os.path.join(folder, filename)
    cv2.imwrite(path, frame)
    if DEBUG:
        print(f"[DEBUG] Stranger image saved at {path}")
    return path


def main():
    print("🔁 Loading LBPH face model...")
    model, label_map = load_model()
    print("✅ Model loaded.")
    if DEBUG:
        print(f"[DEBUG] Label map: {label_map}")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"❌ Cannot open camera at index {CAMERA_INDEX}")
        return

    # ลด resolution กล้องไม่ให้หนักเกิน
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("✅ Camera opened. Press 'q' to exit.")

    # ตัวแปรช่วยลดงาน
    frame_count = 0
    DETECT_EVERY_N_FRAMES = 3      # detect ทุก 3 เฟรม
    DOWNSCALE = 0.5                # ย่อภาพก่อน detect

    last_face_box = None           # (x, y, w, h)
    last_identity = None           # ชื่อคนล่าสุด
    last_conf = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame from camera")
                break

            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # -----------------------------------------
            # 1) ตรวจจับใบหน้าเป็นครั้งคราว (ทุก N เฟรม)
            # -----------------------------------------
            do_detect = (frame_count % DETECT_EVERY_N_FRAMES == 0)

            if do_detect:
                # ย่อภาพลงเพื่อให้ detect เร็วขึ้น
                small_gray = cv2.resize(
                    gray, (0, 0),
                    fx=DOWNSCALE,
                    fy=DOWNSCALE
                )

                # detect บนภาพย่อ
                faces_small = detect_faces(small_gray)

                if len(faces_small) > 0:
                    # scale กลับไปยังพิกัดของภาพจริง
                    faces_full = []
                    for (x, y, w, h) in faces_small:
                        x_f = int(x / DOWNSCALE)
                        y_f = int(y / DOWNSCALE)
                        w_f = int(w / DOWNSCALE)
                        h_f = int(h / DOWNSCALE)
                        faces_full.append((x_f, y_f, w_f, h_f))

                    # เลือกหน้าใหญ่สุด
                    faces_full.sort(key=lambda f: f[2] * f[3], reverse=True)
                    last_face_box = faces_full[0]
                else:
                    last_face_box = None
                    last_identity = None
                    last_conf = None

            # ------------------------------------------------
            # 2) ถ้ามี face box ล่าสุด -> ทำ recognize + วาดกรอบ
            # ------------------------------------------------
            if last_face_box is not None:
                x, y, w, h = last_face_box

                # ป้องกัน index error ถ้า box เลยขอบภาพ
                h_max, w_max = gray.shape[:2]
                x = max(0, min(x, w_max - 1))
                y = max(0, min(y, h_max - 1))
                w = max(1, min(w, w_max - x))
                h = max(1, min(h, h_max - y))

                face_roi = gray[y:y + h, x:x + w]

                # เรียก recognize ทุกครั้งที่มี face
                name, conf = recognize_face(model, label_map, face_roi)
                last_identity = name
                last_conf = conf

                if DEBUG and frame_count % DETECT_EVERY_N_FRAMES == 0:
                    print(f"[DEBUG] Recognized: {name}, conf={conf:.2f}")

                is_stranger = (name == "Stranger")
                color = (0, 255, 0) if not is_stranger else (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label_text = f"{name} ({conf:.1f})"
                cv2.putText(
                    frame,
                    label_text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

                # ----- logic เปิดประตู / แจ้งเตือน -----
                if not is_stranger:
                    cv2.putText(
                        frame,
                        "Door Unlocked",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                    door_unlock()

                    try:
                        send_line_message(MSG_DOOR_OPENED)
                    except Exception as e:
                        if DEBUG:
                            print(f"[DEBUG] LINE notify error (door opened): {e}")

                else:
                    # Stranger: แจ้งเตือนเลย (แบบไม่เข้า while loop ซ้อน)
                    cv2.putText(
                        frame,
                        "Stranger detected!",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                    stranger_img_path = save_stranger_image(frame)
                    print("🚨 Stranger detected -> trigger alarm + LINE.")

                    stranger_alert()
                    try:
                        send_line_message(
                            MSG_STRANGER_DETECTED,
                            image_path=stranger_img_path,
                        )
                    except Exception as e:
                        if DEBUG:
                            print(f"[DEBUG] LINE notify error (stranger): {e}")

            # ---------------------------------------
            # 3) แสดงภาพ
            # ---------------------------------------
            cv2.imshow("Face Door System", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Exit Face Door System.")


if __name__ == "__main__":
    main()
