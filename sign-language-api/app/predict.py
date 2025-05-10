from app.model_loader import letter_model, word_interpreter, number_interpreter
from app.label_dicts import letter_labels, word_labels
from fastapi import UploadFile
from PIL import Image
import numpy as np
import io
from mediapipe.tasks.python.vision.core import vision_task_running_mode
from mediapipe.tasks.python.vision import Image as MPImage
from mediapipe.tasks.python.vision import ImageFormat

def preprocess(file: UploadFile):
    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    image = image.resize((224, 224))
    return np.expand_dims(np.array(image) / 255.0, axis=0).astype(np.float32)

def run_tflite(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    return int(np.argmax(output))

async def predict_letters(file: UploadFile):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    mp_image = MPImage(image_format=ImageFormat.SRGB, data=np.array(image))
    result = letter_model.classify(mp_image)
    class_name = result.classifications[0].categories[0].category_name
    return letter_labels.get(class_name, class_name)

async def predict_words(file: UploadFile):
    image = preprocess(file)
    prediction = run_tflite(word_interpreter, image)
    class_names = list(word_labels.keys())
    predicted_class = class_names[prediction]
    return word_labels[predicted_class]

async def predict_numbers(file: UploadFile):
    image = preprocess(file)
    return str(run_tflite(number_interpreter, image))
