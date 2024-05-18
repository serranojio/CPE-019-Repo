import streamlit as st  
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np


#Setting the page configuration
st.set_page_config(
    page_title="Chess Piece",
    page_icon="♟️",
    initial_sidebar_state = 'auto'
)

# for model loading
st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('chess_model.h5')
    return model

with st.spinner('Model is being loaded..'):
    model = load_model()

# Main body 

#Preparing the backdrops 
page_bg_img = '''
    <style>
        [data-testid="stAppViewContainer"] {
            background-image: url("https://cdn.pixabay.com/photo/2017/09/08/20/29/chess-2730034_1280.jpg");
            background-size: cover;
            }
        
        [data-testid="stHeader"] {
           background-color: rgba(0,0,0,0);
        
        [data-testid="stSidebarContent"] {
            background-color: rgba(38,39,48,.3);
            transition: all 0.5s ease;
        }

    </style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)


#header name (center)
st.write("# Chess Piece Identification")

#for predicting the model 
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


#Side bar Setup
with st.sidebar:
           
    st.write("# Chess")     
    st.markdown("""<div style="text-align: justify;">
              A two-player strategy board game played on a checkered board,
              where each player starts with sixteen pieces. The pieces are 
              moved and used to capture the opponent's pieces following
              specific rules.
              </div>""", unsafe_allow_html=True)
    st.write("")
    st.write("")
    
    if file is None: 
       url='https://www.youtube.com/watch?v=PSzQw1AnvCE&t=0s'
       st.video(url, loop=True, autoplay=True, muted=False)
    

      
if file is None:
    st.text("Please upload an image file")
    
    
else:
    #predicting
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    
    
    #Labels
    class_names = ["Bishop", "King", "Knight", "Pawn", "Queen", "Rook"]
    
    info = {
        'Bishop': "The bishop moves diagonally any number of squares. It is particularly powerful on open diagonals.",
        'King': "The king moves one square in any direction. It is the most important piece, as the game is won by checkmating the king.",
        'Knight': "The knight moves in an L-shape: two squares in one direction and then one square perpendicular to that. It can jump over other pieces.",
        'Pawn': "The pawn moves forward one square, but captures diagonally. On its first move, it can advance two squares.",
        'Queen': "The queen moves any number of squares in any direction: horizontally, vertically, or diagonally. It is the most powerful piece.",
        'Rook': "The rook moves any number of squares horizontally or vertically. It is especially powerful on open files and ranks."
    }
    
    video_urls = {
    'Bishop': 'https://www.youtube.com/watch?v=_y3eA21rD1w&list=PL-qLOQ-OEls6ywMwN8sTJ7k7gRd2f5tO4&index=4',   
    'Knight': 'https://www.youtube.com/watch?v=VGoT8FR0O_8&list=PL-qLOQ-OEls6ywMwN8sTJ7k7gRd2f5tO4&index=5',
    'Pawn': 'https://www.youtube.com/watch?v=00uUlbcPz5E&list=PL-qLOQ-OEls6ywMwN8sTJ7k7gRd2f5tO4&index=6',
    'Queen': 'https://www.youtube.com/watch?v=vwgwI0wnULU&list=PL-qLOQ-OEls6ywMwN8sTJ7k7gRd2f5tO4&index=2',
    'Rook': 'https://www.youtube.com/watch?v=PlgnoYqsK-8&list=PL-qLOQ-OEls6ywMwN8sTJ7k7gRd2f5tO4&index=3'
}
    
      
    #Displaying the results 
    st.sidebar.title("Results")
    string = "Detected Piece: " + class_names[np.argmax(predictions)]
    st.sidebar.info(string)
    
    index = np.argmax(predictions)
    x = predictions[0][index]
    st.sidebar.error("Confidence : " + str(round(x*100,2)) + " %")
    
    detected_piece = class_names[np.argmax(predictions)]
    
    st.sidebar.markdown(f"# Information about {detected_piece}")
    st.sidebar.info(info[detected_piece])
    st.sidebar.write("")
    
    #chess pieces videos 
    if class_names[np.argmax(predictions)] == 'King':
        king_url = 'https://youtu.be/ZWjDKiHBvZo?list=PL-qLOQ-OEls6ywMwN8sTJ7k7gRd2f5tO4&t=27'
        st.sidebar.video(king_url,start_time=27, loop=True, autoplay=True, muted=False)    
    
    else:
        st.sidebar.video(video_urls[detected_piece], loop=True, autoplay=True, muted=False)


