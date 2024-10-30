import cv2
import numpy as np
import os
import pickle
from fastapi import APIRouter, HTTPException, Form
from sklearn.metrics.pairwise import cosine_similarity
import face_recognition
from utils.face_utils import save_face_data, load_face_data, DATA_PATH

router = APIRouter()

@router.post("/register/")
async def register_user(name: str = Form(...)):
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier(f"{DATA_PATH}/haarcascade_frontalface_default.xml")
    embeddings = []

    while True:
        ret, frame = video.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb_frame)

        for (top, right, bottom, left) in faces:
            face_encodings = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])
            if face_encodings:
                embeddings.append(face_encodings[0])
                cv2.putText(frame, str(len(embeddings)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
                cv2.rectangle(frame, (left, top), (right, bottom), (50, 50, 255), 2)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == ord('q') or len(embeddings) == 100:
            break

    video.release()
    cv2.destroyAllWindows()

    embeddings = np.asarray(embeddings)
    save_face_data(name, embeddings)

    return {"message": "User registered successfully!"}


@router.post("/login/")
async def login_user():
    SIMILARITY_THRESHOLD = 0.97
    timeout = 15

    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier(f"{DATA_PATH}/haarcascade_frontalface_default.xml")

    # Load stored face embeddings and labels
    LABELS, EMBEDDINGS = load_face_data(DATA_PATH)

    start_time = cv2.getTickCount()
    while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < timeout:
        ret, frame = video.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb_frame)

        for (top, right, bottom, left) in faces:
            face_encodings = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])
            if face_encodings:
                current_embedding = face_encodings[0].reshape(1, -1)
                similarities = cosine_similarity(current_embedding, EMBEDDINGS)
                max_similarity_index = np.argmax(similarities)
                max_similarity = similarities[0][max_similarity_index]

                # Show detected face and matching status in real-time
                cv2.rectangle(frame, (left, top), (right, bottom), (50, 255, 50), 2)
                if max_similarity > SIMILARITY_THRESHOLD:
                    name = LABELS[max_similarity_index]
                    cv2.putText(frame, f"Welcome, {name}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Login Frame", frame)
                    cv2.waitKey(2000)  # Show successful message briefly
                    video.release()
                    cv2.destroyAllWindows()
                    return {"message": f"Login successful! Welcome, {name}."}
                else:
                    cv2.putText(frame, "Scanning...", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Login Frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    raise HTTPException(status_code=401, detail="Login unsuccessful. Please try again or register.")
