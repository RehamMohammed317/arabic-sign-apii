import tensorflow as tf
from mediapipe.tasks.python import vision
from app.label_dicts import letter_labels, word_labels

# Load letters model (.task)
base_options = vision.BaseOptions(model_asset_path="/content/sign-language-api/app/models/gesture_recognizer.task")
options = vision.ImageClassifierOptions(base_options=base_options)
letter_model = vision.ImageClassifier.create_from_options(options)

# Load TFLite models
word_interpreter = tf.lite.Interpreter(model_path="/content/sign-language-api/app/models/arsl_word_model.tflite")
word_interpreter.allocate_tensors()

number_interpreter = tf.lite.Interpreter(model_path="/content/sign-language-api/app/models/sign_language_model.tflite")
number_interpreter.allocate_tensors()
