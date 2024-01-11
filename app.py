import streamlit as st
import tensorflow as tf





@st.cache_data
def load_model():
    model=tf.keras.models.load_model('fruits.hdf5')
    return model
model=load_model()

st.write("""
       #ROTTEN AND FRESH FRUITS CLASSIFICATION
""")

file=st.file_uploader("Please upload an fruit image",type=['jpg','png'])

import cv2
from PIL import Image,ImageOps
import numpy as np

def predict_function(img,model):
    size=(32,32)
    image=ImageOps.fit(img,size,Image.ANTIALIAS)
    img_arr=np.asarray(image)
    img_scaled=img_arr/255
    img_reshape=np.reshape(img_scaled,[1,32,32,3])
    prediction=model.predict(img_reshape)
    result=np.argmax(prediction)
    if(result==0):
        return 'Yeah it was a FRESH APPLE'
    elif(result==1):
        return 'Yeah it was a FRESH BANANA'
    elif(result==2):
        return 'Yeah it was a FRESH ORANGE'
    elif(result==3):
        return 'Yup it was ROTTEN APPLE'
    elif(result==4):
        return 'Yup it was ROTTEN BANANA'
    elif(result==5):
        return 'Yup it was a ROTTEN ORANGE'
    

if file is None:
    st.text('Please upload an image file')
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    output=predict_function(image,model)
    st.success(output)