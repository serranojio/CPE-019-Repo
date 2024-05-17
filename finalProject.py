import streamlit as st
import tensorflow as tf
print(tf.__version__)
import random
from PIL import Image, ImageOps
import numpy as np

# Setting the page configuration
st.set_page_config(
    page_title="Chess Piece",
    page_icon="♟️",
    initial_sidebar_state='auto'
)

# for model loading
st.set_option('deprecation.showfileUploaderEncoding', False)


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('chess_model.h5')
    return model


with st.spinner('Model is being loaded..'):
    model = load_model()

# Adding Background Image
page_bg_img = '''
<style>
[data-testid="stAppViewContainer"]{

   background-image: url("https://cdn.pixabay.com/photo/2017/09/08/20/29/chess-2730034_1280.jpg");
   background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

st.write("""
         # Chess Piece Identification 
         """
         )

# Side bar Setup
with st.sidebar:
    # st.image('mg.png')
    st.title("*What Piece Am I?*")
    st.subheader(
        "Accurate detection of diseases present in the mango leaves. This helps an user to easily detect the disease and identify it's cause.")

file = st.file_uploader("", type=["jpg", "png"])
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    return prediction


if file is None:
    st.text("Please upload an image file")


else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    index = np.argmax(predictions)
    x = predictions[0][index]
    st.sidebar.error(f"Accuracy: {x:.2f} %")

    class_names = ["Bishop", "King", "Knight", "Pawn", "Queen", "Rook"]

    string = "Detected Piece: " + class_names[np.argmax(predictions)]
    st.sidebar.info(string)

    if class_names[np.argmax(predictions)] == 'Bishop':
        st.markdown("## Information about Bishop")
        st.info("The bishop moves diagonally any number of squares. It is particularly powerful on open diagonals.")

    elif class_names[np.argmax(predictions)] == 'King':
        st.markdown("## Information about King")
        st.info(
            "The king moves one square in any direction. It is the most important piece, as the game is won by checkmating the king.")

    elif class_names[np.argmax(predictions)] == 'Knight':
        st.markdown("## Information about Knight")
        st.info(
            "The knight moves in an L-shape: two squares in one direction and then one square perpendicular to that. It can jump over other pieces.")

    elif class_names[np.argmax(predictions)] == 'Pawn':
        st.markdown("## Information about Pawn")
        st.info(
            "The pawn moves forward one square, but captures diagonally. On its first move, it can advance two squares.")

    elif class_names[np.argmax(predictions)] == 'Queen':
        st.markdown("## Information about Queen")
        st.info(
            "The queen moves any number of squares in any direction: horizontally, vertically, or diagonally. It is the most powerful piece.")

    elif class_names[np.argmax(predictions)] == 'Rook':
        st.markdown("## Information about Rook")
        st.info(
            "The rook moves any number of squares horizontally or vertically. It is especially powerful on open files and ranks.")