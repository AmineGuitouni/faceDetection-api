import cv2
from utils.face_detection import FaceDetection
import numpy as np

detection = FaceDetection()

video = cv2.VideoCapture(0)

emb = []
while True:
    ret, frame = video.read()
    face_tensor = detection.getFace(frame)
    if face_tensor == None:
        continue

    face = face_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    face = (face + 1) / 2
    face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

    cv2.imshow('frame', face)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord(" "):
        face_emb = detection.getEmbedding(face_tensor)
        
        for i, v in enumerate(emb):
            print(f"dictance {i}: {np.linalg.norm(face_emb - v)}")

        print(f"av distance:", {np.linalg.norm(face_emb - np.mean(emb, axis=0))})

        emb.append(face_emb)