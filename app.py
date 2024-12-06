import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np
from PIL import Image
import streamlit as st

# ปิด OneDNN optimization
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# โหลด MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# ตรวจสอบขนาดข้อมูล
print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")

# Normalize ข้อมูล
x_train, x_test = x_train / 255.0, x_test / 255.0

# สร้างโมเดล Sequential
model = models.Sequential([
    layers.Input(shape=(28, 28)), 
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# สรุปโครงสร้างโมเดล
model.summary()

# คอมไพล์โมเดล
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ฝึกโมเดล
model.fit(x_train, y_train, epochs=15)

# ประเมินผลโมเดล
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc * 100:.2f}%")

# ฟังก์ชันทำนายผล
def predict_image(img):
    img = img.convert("L")
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28)
    prediction = model.predict(img_array)
    return np.argmax(prediction)

# Streamlit UI
st.title("MNIST Digit Recognition")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image.", use_container_width=True)
    st.write("")
    
    # แสดงภาพที่ถูกแปลงเป็น 28x28
    img_resized = img.resize((28, 28))
    st.image(img_resized, caption="Resized Image to 28x28", use_container_width=True)
    
    # ทำนายผล
    prediction = predict_image(img)
    st.write(f"Predicted Digit: {prediction}")
    st.write(f"\nTest accuracy: {test_acc * 100:.2f}%")