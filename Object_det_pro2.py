import streamlit as st
import streamlit as st
import cv2
import numpy as np

file=open('objects_list.txt')
li=file.read().split('\n')
classes=list(map(str.strip,li))
file.close()

model=cv2.dnn_DetectionModel('yolov4.cfg','yolov4.weights')
model.setInputSize(416,416)
model.setInputScale(1/255)

def detect(path):
    file_bytes = np.asarray(bytearray(path.read()),dtype=np.uint8)
    img=cv2.imdecode (file_bytes, cv2.IMREAD_COLOR)
    classIds,classProbs,bboxes=model.detect(img,confThreshold=.75,nmsThreshold=.5)

    for box,cls,prob in zip(bboxes,classIds,classProbs):
         x, y, w, h = box
         cv2.rectangle(img,(x,y),(x+w,y+h), (255,0,255),2)
         cv2.putText(img,f'{classes[cls]}({prob:.2f})',(x,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),2)

    return img

# Custom Banner Style
st.markdown("""
    <style>
    .banner {
        background-color: #4CAF50; /* Banner background */
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        color: white;
        font-size: 32px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
import streamlit as st

# Make Streamlit content full width
st.markdown("""
    <style>
    .block-container {
        padding: 0rem;
        margin: 0 auto;
        max-width: 90%;
    }
    .banner {
        width: 100%;
        background:Orange;
        padding: 30px;
        text-align: center;
        color: White;
        font-size: 36px;
        font-weight: bold;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.image("flag.jpg")
st.sidebar.header("📞contact us")
st.sidebar.text("9754509189")

st.sidebar.header("👬About us")
st.sidebar.text("We are a group of AI Engineers working on CNNs")
# Full page banner
st.markdown('<div class="banner">🚀Object Detection Model🚀</div>', unsafe_allow_html=True)
uploaded_file= st.file_uploader("Upload a file", type=["png","jpg","jfif"], accept_multiple_files=False)

col1,col2=st.columns(2)
with col1:
    if uploaded_file:
        st.image(uploaded_file)
        btn= st.button("Prediction")
        with col2:
              if btn:
                img=detect(uploaded_file)
                st.image(img,channels="BGR")
                







