# hardware/door_control_mock.py
import time

# ทำ cooldown กันมันยิงซ้ำทุกเฟรม
_LAST_UNLOCK_TIME = 0
_LAST_ALERT_TIME = 0
UNLOCK_COOLDOWN = 5   # วินาที
ALERT_COOLDOWN = 5    # วินาที


def door_unlock():
    global _LAST_UNLOCK_TIME
    now = time.time()
    if now - _LAST_UNLOCK_TIME < UNLOCK_COOLDOWN:
        return  # เพิ่งปลดล็อกไป ไม่ต้องทำอะไรซ้ำ
    _LAST_UNLOCK_TIME = now
    print("🔓 [MOCK] Door unlocked (no sleep)")

def stranger_alert():
    global _LAST_ALERT_TIME
    now = time.time()
    if now - _LAST_ALERT_TIME < ALERT_COOLDOWN:
        return
    _LAST_ALERT_TIME = now
    print("🚨 [MOCK] Stranger alert! (siren would sound here)")

def is_scan_button_pressed():
    return False
