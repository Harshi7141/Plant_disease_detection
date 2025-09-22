import tensorflow as tf
#from flask_cors import CORS
#CORS(app)
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import numpy as np
from PIL import Image
from io import BytesIO
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Load the trained model
model = tf.keras.models.load_model("potatoes.h5")

# Define class names with precautions and pesticides
class_names = [
    {
        "name": "Early Blight",
        "precautions": "Remove infected leaves, avoid overhead irrigation, rotate crops annually.",
        "pesticides": "Use fungicides like Mancozeb or Chlorothalonil."
    },
    {
        "name": "Late BLight",
        "precautions": "Use resistant varieties, destroy infected plants, avoid wet foliage.",
        "pesticides": "Apply fungicides such as Metalaxyl or Copper-based fungicides."
    },
    {
        "name": "Healthy",
        "precautions": "Continue good agricultural practices, monitor regularly.",
        "pesticides": "No pesticides needed."
    }
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image):
    """Preprocess image for model prediction."""
    image = image.resize((256, 256))  # Resize to match model input size
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_tensor = tf.expand_dims(img_array, axis=0)

    image_processing = preprocess_image(image)
    prediction = model.predict(image_processing)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_info = class_names[predicted_class]

    return JSONResponse({
        "filename": file.filename,
        "shape": img_tensor.shape.as_list(),  # [1, 224, 224, 3]
        "dtype": str(img_tensor.dtype),
        "prediction": predicted_info["name"],
        "precautions": predicted_info["precautions"],
        "pesticides": predicted_info["pesticides"]
    })

@app.get('/')
def index():
    return """Welcome to Plant Disease Detection Backend API"""

@app.get("/upload-file/", response_class=HTMLResponse)
def form():
    return """
    <form action="/process-image/" enctype="multipart/form-data" method="post">
        <input name="file" type="file">
        <input type="submit">
    </form>
    """

if __name__ == "__main__":
    app.run(debug=True)
