import tensorflow as tf
from PIL import Image
import numpy as np
from io import BytesIO
import base64
from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

# Load your trained model
model = tf.keras.models.load_model('diabetic_retinopathy_model.h5')

def preprocess_image(image):
    img = image.resize((224, 224)) 
    img_array = np.array(img) / 255.0 
    return np.expand_dims(img_array, axis=0)

def predict(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    # Adjust this based on your model's output
    return "Diabetic Retinopathy Detected" if prediction[0][0] > 0.5 else "No Diabetic Retinopathy Detected"

# FastAPI app
app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],  # Frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prediction endpoint
@app.post("/predict")
async def predict_endpoint(image: bytes = File(...)):
    try:
        img = Image.open(BytesIO(image))
        label = predict(img)
        return JSONResponse(content={"prediction": label})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run FastAPI with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)