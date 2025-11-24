# line/line_notify.py
"""
ฟังก์ชันสำหรับส่งแจ้งเตือนไป LINE
ถ้ายังไม่ได้ใส่ LINE_NOTIFY_TOKEN จริง -> จะทำงานแบบ MOCK (พิมพ์ใน terminal แทน)
"""

import requests
from config import LINE_NOTIFY_TOKEN


def send_line_message(message: str, image_path: str | None = None) -> int:
    """
    ส่งข้อความ (และรูปถ้ามี) ไป LINE Notify
    ถ้า LINE_NOTIFY_TOKEN ยังเป็น "YOUR_LINE_NOTIFY_TOKEN_HERE" -> จะไม่ยิง API จริง แค่ print เฉย ๆ

    คืนค่า: HTTP status code (200 = สำเร็จ ถ้ายิงจริง)
    """
    if not LINE_NOTIFY_TOKEN or LINE_NOTIFY_TOKEN == "YOUR_LINE_NOTIFY_TOKEN_HERE":
        # โหมด MOCK (ยังไม่ตั้งค่า token จริง)
        print(f"📨 [MOCK LINE] {message}")
        if image_path:
            print(f"📨 [MOCK LINE] with image: {image_path}")
        return 200

    headers = {
        "Authorization": f"Bearer {LINE_NOTIFY_TOKEN}"
    }
    data = {
        "message": message
    }
    files = None

    if image_path:
        try:
            files = {"imageFile": open(image_path, "rb")}
        except FileNotFoundError:
            print(f"⚠️ [LINE] image not found: {image_path}")
            files = None

    resp = requests.post(
        "https://notify-api.line.me/api/notify",
        headers=headers,
        data=data,
        files=files
    )

    print(f"📨 [LINE] status={resp.status_code}, response={resp.text[:100]}")
    return resp.status_code
