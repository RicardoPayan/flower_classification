import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from tensorflow_hub import KerasLayer
from collections import deque
from tensorflow.keras.models import load_model


model = tf.keras.models.load_model(
       ("flower_classification.h5"),
       custom_objects={'KerasLayer': KerasLayer}
)

labels = ["Aster", "Daisy", "Iris", "Lavander", "Lily", "Marigold", "Orchid", "Poppy", "Rose", "Sunflower"]

def process_frame(frame):
    # Redimensionar el frame al tamaño esperado por el modelo
    resized_frame = cv2.resize(frame, (256, 256))

    # Normalizar los valores de píxel en el rango [0, 1]
    normalized_frame = resized_frame / 255.0

    # Agregar una dimensión adicional al tensor del frame para que coincida con la forma esperada por el modelo
    input_frame = np.expand_dims(normalized_frame, axis=0)

    # Realizar la predicción utilizando el modelo
    predictions = model.predict(input_frame)

    # Obtener la clase con la mayor probabilidad
    predicted_class = np.argmax(predictions)

    # Obtener la etiqueta correspondiente a la clase predicha
    predicted_label = labels[predicted_class]

    # Imprimir la etiqueta en la consola
    return predicted_label

st.title("Flower Classification")

frame_placeholder = st.empty()

#Put the start and stop buttons in the same row
col1_start, col2_stop = st.columns([4,1])

with col1_start:
    start_button = st.button("Start")
with col2_stop:
    stop_button = st.button("Stop")

prediction = st.empty()

if start_button:
        cap = cv2.VideoCapture(0) 
        
        while cap.isOpened() and not stop_button:
            # Get the frame
            ok, frame = cap.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not ok:
                print("Error while reading camera frame")
                break

            frame_placeholder.image(frame_rgb, channels="RGB")
            flower = process_frame(frame_rgb)
            prediction.header(flower)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
            