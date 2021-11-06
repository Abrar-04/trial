import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import cv2
import os
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image

activities = ["Flower Classifier", "Gender Classifier",
              "Garbage Classifier", "OKRA Leaf Classifier", "CAT-DOG"]
choice = st.sidebar.selectbox("Select", activities)


def import_and_predict(image_data, model):

    size = (150, 150)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.

    img_reshape = img[np.newaxis, ...]
    prediction = model.predict_classes(img_reshape)

    return prediction


if choice == "Flower Classifier":

    flowerClasses = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    flowerModel = load_model('FlowerClassifier.h5')

    st.title("Welcome to FLower CLassifier")
    st.header("Identify what's the flower!")

    if st.checkbox("These are the classes of flowers it can identify"):
        st.write(flowerClasses)

    file = st.file_uploader("Please upload a Flower Image", type=[
                            "jpg", "png", "jpeg"])

    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        predictions = import_and_predict(image, flowerModel)
        result = predictions
        st.write(predictions)
        if result == 0:
            st.write("Prediction : Daisy")
        elif result == 1:
            st.write("Prediction : Dandelion")
        elif result == 2:
            st.write("Prediction : Rose")
        elif result == 3:
            st.write("Prediction : Sunflower")
        elif result == 4:
            st.write("Prediction : Tulip")


if choice == "Gender Classifier":

    genderClasses = ['female', 'male']
    genderModel = load_model('GenderClassifier.h5')

    st.title("Welcome to Gender CLassifier")
    st.header("Identify what's the Gender!")

    if st.checkbox("These are the classes of Gender it can identify"):
        st.write(genderClasses)

    file = st.file_uploader("Please upload a Gender Image", type=[
                            "jpg", "png", "jpeg"])

    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        predictions = import_and_predict(image, genderModel)
        result = predictions
        st.write(predictions)
        if result == 0:
            st.write("Prediction : female")
        elif result == 1:
            st.write("Prediction : male")


if choice == "Garbage Classifier":

    garbageClasses = ['biological', 'cardboard',
                      'glass', 'metal', 'paper', 'plastic', 'trash']
    garbageModel = load_model('GarbageClassifier.h5')

    st.title("Welcome to Garbage CLassifier")
    st.header("Identify what's the Garbage!")

    if st.checkbox("These are the classes of Garbage it can identify"):
        st.write(garbageClasses)

    file = st.file_uploader("Please upload a Garbage Image", type=[
                            "jpg", "png", "jpeg"])

    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        predictions = import_and_predict(image, garbageModel)
        result = predictions
        st.write(predictions)

        if result == 0:
            st.write("Prediction : biological")
        elif result == 1:
            st.write("Prediction : cardboard")
        elif result == 2:
            st.write("Prediction : glass")
        elif result == 3:
            st.write("Prediction : metal")
        elif result == 4:
            st.write("Prediction : paper")
        elif result == 5:
            st.write("Prediction : plastic")
        elif result == 6:
            st.write("Prediction : trash")

if choice == "OKRA Leaf Classifier":

    okraClasses = ['diseased okra leaf', 'fresh okra leaf']
    okraModel = load_model('OkraClassifier.h5')

    st.title("Welcome to OKRA Leaf CLassifier")
    st.header("Identify what's the OKRA Leaf!")

    if st.checkbox("These are the classes of OKRA Leaf it can identify"):
        st.write(okraClasses)

    file = st.file_uploader("Please upload a OKRA Leaf Image", type=[
                            "jpg", "png", "jpeg"])

    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        predictions = import_and_predict(image, okraModel)
        result = predictions
        st.write(predictions)

        if result == 0:
            st.write("Prediction : diseased okra leaf")
        if result == 1:
            st.write("Prediction : fresh okra leaf")

# 0:cat,1:dog
if choice == "CAT-DOG":

    animalClasses = ['cat', 'dog']
    animalModel = load_model('animalClassifier.h5')

    st.title("Welcome to CAT-DOG CLassifier")
    st.header("Identify what's the CAT-DOG!")

    if st.checkbox("These are the classes of animals it can identify"):
        st.write(animalClasses)

    file = st.file_uploader("Please upload a CAT-DOG Image", type=[
                            "jpg", "png", "jpeg"])

    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        demo = np.array(image)
        demo = demo[:, :, ::-1].copy()
        #demo = cv2.imread(image)
        demo = tf.image.convert_image_dtype(demo, tf.float32)
        demo = tf.image.resize(demo, size=[224, 224])
        demo = np.expand_dims(demo, axis=0)
        pred = animalModel.predict(demo)
        result = np.argmax(pred)
        st.write(result)

        if result == 0:
            st.write("Prediction : CAT")
        if result == 1:
            st.write("Prediction : DOG")
