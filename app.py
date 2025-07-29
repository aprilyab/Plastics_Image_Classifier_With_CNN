import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load model from 'models' folder
model_path = os.path.join("models", "Plastics_Image_Classifier_CNN_Model.h5")
model = tf.keras.models.load_model(model_path)

# Class labels
classes = ["PET", "HDPE", "PP", "PS"]

# Define prediction function
def classify_image(img):
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    img = img.reshape(1, 256, 256, 3)
    prediction = model.predict(img).argmax()
    return classes[prediction]

# Gradio interface
interface = gr.Interface(fn=classify_image, inputs="image", outputs="label")
interface.launch(server_name="0.0.0.0", server_port=7860)
