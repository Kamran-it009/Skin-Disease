import io
import numpy as np
from PIL import Image
import streamlit as st
from inference import predictor
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title='Skin Disease Detection',
    page_icon=":stethoscope:"
)

# 1. as sidebar menu
with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'Downloads', 'About',  'Contact Us'],
                        icons=['house', 'cloud-arrow-down', 'info-square', 'envelope', ], menu_icon="cast", default_index=0,
                        styles={"nav-link-selected": {"background-color": "green"}})
if selected == 'Home':
    # 2. Heading
    st.header('Skin Disease :blue :stethoscope:', divider='rainbow')
    # 3. Uploading file
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    click = None
    if uploaded_file is not None:
        st.image(uploaded_file, width=250)
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image.convert('RGB'))
        image_array = image_array * 255.0
        image_array = np.resize(image_array, (512, 512, 3))
        label = predictor(image_array, 'keras_model.h5')
        st.write('Image uploaded successfully..')

        # Create a centered button
        click = st.button(":white[Predict]", type='primary', disabled=False)
    else:
        st.button(":white[Predict]", type= 'primary', disabled=True)
    if click:
        st.subheader(f':green[Disease :] {label}')

elif selected == 'Downloads':
    st.header('Downloads', divider='rainbow')
    st.write(':green[**Dataset:**    **https://www.kaggle.com/code/yousefzidan101/skindiseas/input**]')
    st.write(':green[**Code:**]')
elif selected == 'About':
    st.header('About', divider='rainbow')
    st.write(':blue[**Draikin is a prediagnostic progressive web app that helps to scan and analyse skin pathology.**]')
else:
    st.header('Contact Us', divider='rainbow')
    st.write(':blue[If you have any questions about this Progressive Web App. You can contact us:]')
    st.write(':green[**By email: motubas@gmail.com**]')
st.sidebar.success('Select the above page')




