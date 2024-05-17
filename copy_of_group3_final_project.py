import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model

@st.cache(allow_output_mutation=True)
def load_fashion_model():
    model = tf.keras.models.load_model('saved_fashion.h5')
    return model

def import_and_predict(image_data, model):
    size = (28, 28)

    # Ensure image_data is in the correct data type and range
    image_data = (image_data * 255).astype(np.uint8)
    # Convert the NumPy array to an Image instance
    image = Image.fromarray(image_data)
    # Use ImageOps.fit with the Image instance
    
    image = ImageOps.fit(image, size)
    img = np.asarray(image)
    
    img = img[:, :, 0]

    img_reshape = img[np.newaxis, ..., np.newaxis]
    prediction = model.predict(img_reshape)
    return prediction

def load_image():

    
    # Check if the image is grayscale, if so, add a channel dimension
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    
    img = img / 255.0
    img = np.reshape(img, (1, 64, 64, img.shape[-1]))
    return img

model = load_fashion_model()

st.write("""# Fashion Dataset by Espiritu_Santos""")
file = st.file_uploader("Choose photo from computer", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)

    # Convert the image to a NumPy array
    image_array = img_to_array(image)

    # Normalize the image
    image_array = image_array / 255.0

    # Load the image into the model for prediction
    prediction = import_and_predict(image_array, model)

    class_names = ['T-shirt', 'Top', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

    result_class = np.argmax(prediction)
    result_label = class_names[result_class]
    string = f"Prediction: {result_label} ({prediction[0][result_class]:.2%} confidence)"
    st.success(string)
