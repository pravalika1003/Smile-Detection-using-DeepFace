import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def detect_smile_real_time():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Couldn't access the webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 6)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            smiles = smile_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.7,
                minNeighbors=20,
                minSize=(25, 25)
            )

            color = (0, 255, 0) if len(smiles) > 0 else (0, 0, 255)
            label = "😊 Smile Detected!" if len(smiles) > 0 else "😐 No Smile"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 255, 0), 2)

        cv2.imshow("Smile Detector - Real-Time", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_smile_in_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Couldn't load image.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)

        color = (0, 255, 0) if len(smiles) > 0 else (0, 0, 255)
        label = "Smile Detected!" if len(smiles) > 0 else "No Smile Detected"
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (255, 255, 0), 2)

    cv2.imshow("Smile Detector - Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_smile_in_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Couldn't open video file.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ Video processing complete.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 6)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            smiles = smile_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.7,
                minNeighbors=20,
                minSize=(25, 25)
            )

            color = (0, 255, 0) if len(smiles) > 0 else (0, 0, 255)
            label = "😊 Smile Detected!" if len(smiles) > 0 else "😐 No Smile"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 255, 0), 2)

        cv2.imshow("Smile Detector - Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


print("😊 Smile Detection App 😊")
print("1. Detect smile in real-time (Webcam)")
print("2. Detect smile in uploaded image")
print("3. Detect smile in video file")

choice = input("Choose an option (1/2/3): ")

if choice == '1':
    detect_smile_real_time()
elif choice == '2':
    image_path = input("Enter the image file path: ").strip('"')
    detect_smile_in_image(image_path)
elif choice == '3':
    video_path = input("Enter the video file path: ").strip('"')
    detect_smile_in_video(video_path)
else:
    print("Invalid choice. Please enter 1, 2, or 3.")
