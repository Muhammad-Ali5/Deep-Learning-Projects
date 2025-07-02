import streamlit as st
import os
import matplotlib.pyplot as plt
from cnn_model import create_cnn_model, preprocess_image, get_feature_maps, plot_feature_maps

st.set_page_config(page_title="CNN Feature Maps", layout="centered")
st.title("🧠 CNN Hidden Layer Visualization")

uploaded_file = st.file_uploader(" Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_path = "temp_image.jpg"
    
    # Save uploaded file temporarily
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, caption="📷 Uploaded Image", width=224)
    
    # Run CNN visualization
    model = create_cnn_model()
    img_array = preprocess_image(temp_path)
    feature_maps, conv_layers = get_feature_maps(model, img_array)
    plots = plot_feature_maps(feature_maps, conv_layers)
    
    # Display each layer’s feature map
    for plot in plots:
        st.pyplot(plot)

    # Clean up
    plt.close('all')
    os.remove(temp_path)
