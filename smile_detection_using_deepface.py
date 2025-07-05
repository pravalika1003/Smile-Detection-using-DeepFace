import cv2
from deepface import DeepFace

def analyze_emotions(frame):
    try:
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if isinstance(results, dict):  # Single face
            results = [results]
        return results
    except Exception as e:
        print("⚠️ Error in DeepFace:", e)
        return []

def draw_faces_with_emotions(frame, analysis_results):
    for face in analysis_results:
        region = face.get('region', {})
        x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
        emotion = face.get("dominant_emotion", "unknown")
        label = "😊 Smile Detected" if emotion == "happy" else f"😐 {emotion.title()}"
        color = (0, 255, 0) if emotion == "happy" else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame


def detect_smile_real_time():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not access webcam.")
        return

    print("🎥 Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = analyze_emotions(frame)
        frame = draw_faces_with_emotions(frame, results)

        cv2.imshow("Real-Time Smile Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_smile_in_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Could not load image.")
        return

    results = analyze_emotions(img)
    img = draw_faces_with_emotions(img, results)

    cv2.imshow("Image Smile Detector", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_smile_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video.")
        return

    print("🎥 Press 'q' to stop playback.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = analyze_emotions(frame)
        frame = draw_faces_with_emotions(frame, results)

        cv2.imshow("Video Smile Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# --- Menu ---
print("😊 Smile Detection App (DeepFace powered) 😊")
print("1. Detect smile in real-time (Webcam)")
print("2. Detect smile in uploaded image")
print("3. Detect smile in video file")

choice = input("Choose an option (1/2/3): ")

if choice == '1':
    detect_smile_real_time()
elif choice == '2':
    path = input("Enter the image file path: ").strip('"')
    detect_smile_in_image(path)
elif choice == '3':
    path = input("Enter the video file path: ").strip('"')
    detect_smile_in_video(path)
else:
    print("Invalid choice. Please enter 1, 2, or 3.")
