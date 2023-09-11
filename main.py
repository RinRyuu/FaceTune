import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import Counter

# Load the trained emotion recognition model
emotion_model = load_model('emotion_model.h5')  # Replace with your actual model path

# List of emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('/home/rin/PycharmProjects/FaceTune/IOT Project/haarcascade-frontalface-default.xml')  # Replace with your actual XML file path

# Initialize variables for emotion tracking
window_emotions = []  # Emotions detected in the 10-second window
window_duration = 10  # 10 seconds
frame_rate = 30  # Assuming 30 frames per second
show_video = [True]  # Flag to control video display using a mutable list

# Create an OpenCV window with a graphical button
cv2.namedWindow('Emotion Detection')
button_text = "Toggle Video On" if show_video[0] else "Toggle Video Off"

# Define button colors and fonts
button_bg_color = (0, 0, 255)  # Red background when video is off
button_text_color = (255, 255, 255)  # White text color
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_thickness = 2
button_height, button_width = 40, 200
button_x, button_y = 20, 20

def toggle_video_display(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        show_video[0] = not show_video[0]
        update_button_text()

def update_button_text():
    global button_text
    button_text = "Toggle Video On" if show_video[0] else "Toggle Video Off"

# Set the mouse callback function to handle button clicks
cv2.setMouseCallback('Emotion Detection', toggle_video_display)

# Open the camera
cap = cv2.VideoCapture(0)  # Initialize the video capture object

while True:
    ret, frame = cap.read()  # Capture a frame from the camera

    if show_video[0]:
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]  # Extract the face region
            resized = cv2.resize(face_roi, (48, 48))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 48, 48, 3))

            # Predict the emotion using the loaded model
            emotion_scores = emotion_model.predict(reshaped)
            emotion_index = np.argmax(emotion_scores)
            detected_emotion = emotion_labels[emotion_index]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Add detected emotion to the window_emotions list
            window_emotions.append(detected_emotion)

    # Draw the toggle button
    if show_video[0]:
        button_bg_color = (0, 255, 0)  # Green background when video is on
    else:
        button_bg_color = (0, 0, 255)  # Red background when video is off

    cv2.rectangle(frame, (button_x, button_y), (button_x + button_width, button_y + button_height), button_bg_color, -1)
    cv2.putText(frame, button_text, (button_x + 10, button_y + 30), font, font_scale, button_text_color, font_thickness)

    cv2.imshow('Emotion Detection', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # Calculate the time elapsed
    elapsed_time = len(window_emotions) / frame_rate

    # Check if the 10-second window has elapsed
    if elapsed_time >= window_duration:
        # Count the most frequent emotion in the window
        emotion_counter = Counter(window_emotions)
        most_common_emotion = emotion_counter.most_common(1)[0][0]

        # Print the most common emotion
        print(f"Most Common Emotion in the Last 10 Seconds: {most_common_emotion}")

        # Clear the window_emotions list for the next window
        window_emotions.clear()

cap.release()
cv2.destroyAllWindows()