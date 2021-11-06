import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import cv2
from PIL import Image, ImageOps
import numpy as np



classes=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']


model=load_model('FlowerClassifier.h5')

st.title("Welcome to FLower CLassifier")
st.header("Identify what's the flower!")

if st.checkbox("These are the classes of flowers it can identify"):
    st.write(classes)


file = st.file_uploader("Please upload a Flower Image", type=["jpg", "png","jpeg"])

def import_and_predict(image_data, model):
    
        size = (150,150)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img[np.newaxis,...]
        prediction = model.predict_classes(img_reshape)
        
        return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    result=predictions
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

