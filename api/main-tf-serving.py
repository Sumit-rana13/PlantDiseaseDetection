# from fastapi import FastAPI, File , UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import tensorflow as tf
#
# app = FastAPI()
#
# origins = [
#     "http://localhost",
#     "http://localhost:3000",
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
#
#
# MODEL = tf.keras.models.load_model("saved_models/1")
#
# CLASS_NAMES = ["Tomato_Early_blight", "Tomato_healthy"]
#
# @app.get("/ping")
# async def ping():
#     return "Hello, I am alive"
#
# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image
#
# @app.post("/predict")
# async def predict(
#     file: UploadFile = File(...)
# ):
#     image = read_file_as_image(await file.read())
#     img_batch = np.expand_dims(image, 0)
#
#     predictions = MODEL.predict(img_batch)
#
#     predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#     confidence = np.max(predictions[0])
#     return {
#         'class': predicted_class,
#         'confidence': float(confidence)
#     }
#
# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)
import numpy as np
from fastapi import FastAPI , File , UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests


app = FastAPI()

endpoint = "http://localhost:8501/v1/models/tomatoes_model:predict"

MODEL = tf.keras.models.load_model("../saved_models/1")
CLASS_NAMES = ["Tomato_Early_blight", "Tomato_healthy", "YellowLeaf__Curl_Virus"]

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    json_data = {
        "instances": img_batch.tolist()
    }

    response = requests.post(endpoint, json=json_data)
    prediction = np.array(response.json()["predictions"][0])

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    np.max(prediction)
    confidence = np.max(prediction)

    return {
        'class': predicted_class,
        'container': confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
