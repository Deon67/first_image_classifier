import streamlit as st
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
from PIL import Image
import keras
import tensorflow

st.write("# Animal Classifier")
st.set_option('deprecation.showfileUploaderEncoding', False)

st.write("""This is a simple image classification web app to predict three - Animals Elephant,Giraffe and Hippopotamus """)

file = st.file_uploader("Please upload an image file", type=["jpg", 'jpeg',"png"])


if file is None:
	st.text("Please upload an image file")
else:
	image4 = Image.open(file)
	st.image(image4, use_column_width=True)
	image2=image4.resize((64,64))
	test_image = image.img_to_array(image2)
	test_image = np.expand_dims(test_image,axis=0)
	classifier=load_model('animal_classifier.h5')
	result = classifier.predict(test_image)
	
	if result[0][0] == 1:
   	 	st.write('# Giraffe')
	elif result[0][1]==1:
    		st.write('# Hippopotamus')
	elif result[0][2]==1:
    		st.write('# Elephant')
  
