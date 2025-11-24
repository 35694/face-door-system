# config.py
"""
Global configuration for Face Door System
"""

# --------- DATASET & MODEL PATHS ---------
DATASET_DIR = "dataset_faces"

LBPH_MODEL_PATH = "models/lbph_model.xml"
LABEL_MAP_PATH = "models/label_map.json"

# ขนาดภาพใบหน้าที่ใช้ตอน train และ recognize
IMAGE_SIZE = (200, 200)


# --------- CAMERA SETTINGS ---------
# กล้องตัวไหน (0 = กล้องหลักใน Mac / Webcam ตัวแรก)
CAMERA_INDEX = 0


# --------- RECOGNITION / LOGIC SETTINGS ---------
# ระยะเวลารอให้ Stranger ยืนยัน (วินาที)
STRANGER_TIMEOUT = 7

# เกณฑ์ความมั่นใจของ LBPH (ยิ่งต่ำยิ่งเข้มงวด)
CONF_THRESHOLD = 70



# เปิด debug log เพิ่มใน terminal หรือไม่
DEBUG = False


# --------- LINE NOTIFY SETTINGS ---------
# ถ้าคุณยังไม่มี token ให้ใส่ค่าเดิมนี้ไปก่อน
LINE_NOTIFY_TOKEN = "YOUR_LINE_NOTIFY_TOKEN_HERE"

# ข้อความ template ที่จะส่งไป LINE
MSG_STRANGER_DETECTED = "🚨 Stranger detected at your door!"
MSG_DOOR_OPENED = "🔓 Door opened for authorized person."
