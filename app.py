import streamlit as st
from PIL import Image
import tensorflow as tf
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.models import load_model

image = Image.open('app_heading.webp')
st.image(image)

st.write("""
# Table Detection App
This app detects the table data in an image.""")

model_load_state = st.text('Loading model...')

model = load_model('model_148')

model_load_state.text('Model loading completed!')

file = st.file_uploader("Please upload an Image file")

def predict_lite(filepath):
    img = np.array(filepath.convert('RGB'))
    img = cv2.resize(img,(1024,1024), cv2.INTER_AREA)
    img_256 = cv2.resize(img,(256,256), cv2.INTER_NEAREST)
    if img_256.shape[2] == 4:
        img_256 = cv2.cvtColor(img_256, cv2.COLOR_RGBA2RGB)
    elif img_256.shape[2] == 1:
        img_256 = cv2.cvtColor(img_256, cv2.COLOR_GRAY2RGB)
    else:
        img_256 = cv2.cvtColor(img_256, cv2.COLOR_BGR2RGB)
    img_256 = np.expand_dims(img_256, axis = 0)
    img_256 = img_256.astype(np.float32)
    st.write('Predicting....')
    pred_mask1, pred_mask2 = model.predict(img_256)
    pred_tabmask = tf.argmax(pred_mask1, axis=-1)
    pred_tabmask = pred_tabmask[..., tf.newaxis]
    pred_colmask = tf.argmax(pred_mask2, axis=-1)
    pred_colmask = pred_colmask[..., tf.newaxis]
    pred_table_mask = pred_tabmask[0]
    pred_column_mask = pred_colmask[0]

    pred_table_mask = tf.keras.preprocessing.image.array_to_img(pred_table_mask)
    pred_column_mask = tf.keras.preprocessing.image.array_to_img(pred_column_mask)
    pred_table_mask = cv2.resize(np.array(pred_table_mask), (1024,1024), cv2.INTER_NEAREST)
    pred_column_mask = cv2.resize(np.array(pred_column_mask), (1024,1024), cv2.INTER_NEAREST)
    pred_table_mask = tf.keras.preprocessing.image.array_to_img(pred_table_mask[:,:,np.newaxis])
    pred_column_mask = tf.keras.preprocessing.image.array_to_img(pred_column_mask[:,:,np.newaxis])

    img_final = tf.keras.preprocessing.image.array_to_img(img)
    img_final.putalpha(pred_table_mask)
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    st.write('Extracting text...')
    pytesseract.pytesseract.tesseract_cmd = (r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    text = pytesseract.image_to_string(img_final)
    plt.figure(figsize = (10,10))
    plt.imshow(img_final)
    plt.title('Detected Table')
    plt.show()
    st.pyplot()
    if len(text) ==1:
        st.write('No table is present in the image')
    else:
        st.write('The detected text is:')
        st.write(text)
        
if file is None:
    st.write('Please upload an image')
else:
    image = Image.open(file)
    predict_lite(image)
