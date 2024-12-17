import requests
import os
import numpy as np
from PIL import Image
from io import BytesIO
from time import time

class EspCam:
    def __init__(self, URL):
        self.URL = URL
        self.save_dir = "captured_images"
        os.makedirs(self.save_dir, exist_ok=True)

    def get_image(self):
        
        try:
            print(f"Fetching image from {self.URL}...")
            response = requests.get(self.URL, stream=True)

            if response.status_code == 200:
                # Convert the image from the response to a NumPy array
                image = Image.open(BytesIO(response.content))
                image_array = np.array(image)

                # Save the NumPy array to a file
                npy_file_path = os.path.join(self.save_dir, f"{time()}.jpg")
                image.save(npy_file_path, format="JPEG")

                print(f"Image saved as array at: {npy_file_path}")
                print(f"Image shape: {image_array.shape}")
                return image_array
            else:
                print(f"Failed to fetch image. HTTP Status: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching image: {e}")
            return None
