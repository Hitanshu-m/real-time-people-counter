import cv2

# Haarcascade face detection model load karo
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Webcam start karo
cap = cv2.VideoCapture(0)
print("Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize aur grayscale
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Faces detect karo
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Rectangle draw karo aur count show karo
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    people_count = len(faces)
    cv2.putText(frame, f"People Count: {people_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show frame
    cv2.imshow("People Counter", frame)

    # Quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

