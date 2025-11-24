# vision/face_detect.py
import cv2

# ใช้ Haar Cascade ที่มากับ OpenCV
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def detect_faces(gray_frame):
    """
    รับภาพ gray แล้วคืน list ของ faces: (x, y, w, h)
    ปรับ scaleFactor / minNeighbors / minSize ให้บาลานซ์ระยะใกล้-ไกล
    """
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.2,   # ปรับตรงนี้ถ้าจับหน้าไกล/ใกล้ไม่ดี
        minNeighbors=5,
        minSize=(80, 80)   # กันหน้าเล็กเกิน (อยู่ไกลเกิน)
    )
    return faces

def test_face_detect(cam_index=0):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print("✅ Face detect running. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)

        # วาดกรอบรอบหน้า
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Face Detect Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_face_detect(0)
