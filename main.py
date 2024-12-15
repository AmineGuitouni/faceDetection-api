from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
import base64
import io
from PIL import Image
import numpy as np

from utils.face_detection import FaceDetection
from utils.db import SQLiteUtils

# Initialization
app = FastAPI()

db_path = "faceDetection.db"
face_detection = FaceDetection()
sqlite_utils = SQLiteUtils(db_path)

# Request models
class AddUserRequest(BaseModel):
    first_name: str
    last_name: str
    images: list[str]

class CheckUserRequest(BaseModel):
    image: str

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
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            image = np.array(image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

        # Detect face and get embedding
        cropped_face = face_detection.getFace(image)
        if cropped_face is None:
            raise HTTPException(status_code=400, detail="No face detected in one of the images.")

        embedding = face_detection.getEmbedding(cropped_face)
        embeddings.append(embedding)

    # Average the embeddings if multiple images are provided
    if len(embeddings) > 1:
        final_embedding = np.mean(embeddings, axis=0)
    else:
        final_embedding = embeddings[0]

    # Insert the user into the database
    sqlite_utils.insert_user(
        image_path="",
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

@app.post("/check_user")
async def check_user(request: CheckUserRequest):
    # Decode the base64 image
    try:
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data))
        image = np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

    # Detect face and get embedding
    cropped_face = face_detection.getFace(image)
    if cropped_face is None:
        raise HTTPException(status_code=400, detail="No face detected in the image.")

    embedding = face_detection.getEmbedding(cropped_face)

    # Fetch similar users from the database
    results = sqlite_utils.distance_similarity_fetch(
        vector=embedding,
        limit=1,
        max_distance=0.6
    )

    if not results:
        raise HTTPException(status_code=404, detail="No matching user found.")

    user_id, first_name, last_name, distance = results[0]
    return {
        "user_id": user_id,
        "first_name": first_name,
        "last_name": last_name,
        "distance": distance
    }

@app.on_event("shutdown")
async def shutdown_event():
    sqlite_utils.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)