import streamlit as st
from tensorflow import keras
from PIL import Image
import numpy as np

model = keras.models.load_model("mnist.keras")

st.title("MNIST Digit Recognizer")
st.write("Upload a handwritten digit image (0-9)")

uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    img = Image.open(uploaded).convert("L").resize((28, 28))
    st.image(img, caption="Uploaded image", width=150)

    arr = np.array(img).reshape(1, 784).astype("float32") / 255
    pred = model.predict(arr)
    digit = np.argmax(pred)
    confidence = pred[0][digit] * 100

    st.success(f"Prediction: **{digit}** ({confidence:.1f}% confidence)")
