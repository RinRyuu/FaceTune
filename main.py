import cv2
import numpy as np
import pygame
from tensorflow.keras.models import load_model
from collections import deque

model = load_model('emotion_model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

pygame.mixer.init()

emotion_queue = deque(maxlen=5)
emotion_songs = {
    'Angry': 'angry.mp3',
    'Happy': 'happy.mp3',
    'Sad': 'sad.mp3',
}

current_song = None

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    detected_emotions = []

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        resized = cv2.resize(face_roi, (48, 48))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 48, 48, 3))

        result = model.predict(reshaped)
        emotion_index = np.argmax(result)
        emotion = emotion_labels[emotion_index]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        detected_emotions.append(emotion)

    emotion_queue.append(detected_emotions)

    if len(emotion_queue) == 5:
        flat_emotions = [emotion for sublist in emotion_queue for emotion in sublist]

        if flat_emotions:
            most_common_emotion = max(set(flat_emotions), key=flat_emotions.count)

            if most_common_emotion in emotion_songs:
                song_filename = emotion_songs[most_common_emotion]

                if current_song != song_filename:
                    if current_song is not None:
                        pygame.mixer.music.stop()
                    current_song = song_filename
                    pygame.mixer.music.load(song_filename)
                    pygame.mixer.music.play()

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
