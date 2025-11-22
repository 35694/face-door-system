# vision/camera_test.py
import cv2

def test_camera(cam_index=0):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print("✅ Camera opened. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break

        cv2.imshow("Camera Test", frame)

        # กด q เพื่อออก
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera(0)
