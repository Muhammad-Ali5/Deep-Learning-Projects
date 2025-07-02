# cnn_model.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

# ✅ Define a simple CNN model using Functional API
def create_cnn_model(input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', name='conv1')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', name='conv2')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', name='conv3')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# ✅ Preprocess image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

# ✅ Extract feature maps
def get_feature_maps(model, img_array):
    conv_layers = [layer for layer in model.layers if 'conv' in layer.name]
    feature_map_models = [Model(inputs=model.input, outputs=layer.output) for layer in conv_layers]
    feature_maps = [fm_model.predict(img_array, verbose=0) for fm_model in feature_map_models]
    return feature_maps, conv_layers

# ✅ Plot feature maps
def plot_feature_maps(feature_maps, conv_layers):
    plots = []
    for i, (fm, layer) in enumerate(zip(feature_maps, conv_layers)):
        fm = fm[0]
        num_filters = fm.shape[-1]
        fig = plt.figure(figsize=(15, 10))
        plt.suptitle(f'Feature Maps for {layer.name}', fontsize=16)
        n_cols = 4
        n_rows = min(4, (num_filters + n_cols - 1) // n_cols)
        for j in range(min(16, num_filters)):
            plt.subplot(n_rows, n_cols, j+1)
            plt.imshow(fm[:, :, j], cmap='viridis')
            plt.axis('off')
            plt.title(f'Filter {j+1}')
        plots.append(fig)
    return plots
