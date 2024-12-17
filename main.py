from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
import base64
import io
from PIL import Image
import numpy as np

from utils.face_detection import FaceDetection
from utils.db import SQLiteUtils
from utils.esp_cam import EspCam

# Initialization
app = FastAPI()

db_path = "faceDetection.db"
face_detection = FaceDetection()
sqlite_utils = SQLiteUtils(db_path)
espcam = EspCam("http://192.168.137.48/capture")
# Request models
class AddUserRequest(BaseModel):
    first_name: str
    last_name: str
    images: list[str]

@app.get("/")
def check_health():
    return {"status": "ok"}

@app.post("/add_user")
async def add_user(request: AddUserRequest):
    embeddings = []

    # Process each image
    for image_base64 in request.images:
        # Decode the base64 image
        try:
            header, base64_data = image_base64.split(',', 1)
            image_data = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_data))
            image = np.array(image)
            print(image.shape)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

        # Detect face and get embedding
        cropped_face = face_detection.getFace(image)
        if cropped_face is None:
            raise HTTPException(status_code=400, detail="No face detected in one of the images.")

        print("face detected")
        embedding = face_detection.getEmbedding(cropped_face)
        print("vectore", embedding)
        embeddings.append(embedding)

    if len(embeddings) == 0:
        return {"status": 0}

    # Average the embeddings if multiple images are provided
    if len(embeddings) > 1:
        final_embedding = np.mean(embeddings, axis=0)
    else:
        final_embedding = embeddings[0]

    # Insert the user into the database
    sqlite_utils.insert_user(
        image_path="aaa",
        embedding_vector=final_embedding,
        first_name=request.first_name,
        last_name=request.last_name
    )

    return {"message": "User added successfully."}

@app.delete("/delete_user/{user_id}")
async def delete_user(user_id: int):
    try:
        sqlite_utils.delete_by_id(user_id)
        return {"message": "User deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting user: {str(e)}")

@app.get("/check_user")
async def check_user():
    print("asdasd")
    # Decode the base64 image
    try:
        image = espcam.get_image()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

    print("image", image.shape)
    # Detect face and get embedding
    cropped_face = face_detection.getFace(image)
    print(cropped_face)
    if cropped_face is None:
        raise HTTPException(status_code=400, detail="No face detected in the image.")

    embedding = face_detection.getEmbedding(cropped_face)
    print(embedding)
    # Fetch similar users from the database
    results = sqlite_utils.distance_similarity_fetch(
        vector=embedding,
        limit=1,
        max_distance=1
    )

    print(results)

    if not results:
        raise HTTPException(status_code=404, detail="No matching user found.")

    user_id, first_name, last_name, distance = results[0]
    print(user_id, first_name, last_name, distance)
    return {
        "user_id": user_id,
        "first_name": first_name,
        "last_name": last_name,
    }

@app.get("/get_users")
async def delete_user():
    dbUser = sqlite_utils.fetch_all()
    print(len(dbUser))
    users = []
    for user in dbUser:
        user_id, image_path, embedding_blob, first_name, last_name = user
        print(user_id, first_name, last_name)
        # vector = np.frombuffer(embedding_blob, dtype=np.float32)
        users.append({
            "user_id":user_id,
            "first_name":first_name,
            "last_name":last_name
        })

    return users

@app.on_event("shutdown")
async def shutdown_event():
    sqlite_utils.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)