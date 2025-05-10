from fastapi import FastAPI, UploadFile, File
from app.predict import predict_letters, predict_words, predict_numbers

app = FastAPI()

@app.get("/")
def home():
    return {"msg": "Arabic Sign Language Recognition API"}

@app.post("/predict/letters")
async def letter(file: UploadFile = File(...)):
    result = await predict_letters(file)
    return {"prediction": result}

@app.post("/predict/words")
async def word(file: UploadFile = File(...)):
    result = await predict_words(file)
    return {"prediction": result}

@app.post("/predict/numbers")
async def number(file: UploadFile = File(...)):
    result = await predict_numbers(file)
    return {"prediction": result}
