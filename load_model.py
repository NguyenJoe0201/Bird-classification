import streamlit as st
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.applications.resnet50 import preprocess_input

base_model = ResNet50(input_shape=(224,224,3),
                        include_top=False,
                        weights='imagenet')
base_model.trainable = True
global_average_layer = GlobalAveragePooling2D()
prediction_layer = Dense(260, activation='softmax')
inputs = Input(shape=(224, 224, 3))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
outputs = prediction_layer(x)
model = Model(inputs, outputs)
base_model.trainable = True
for layer in base_model.layers[:100]:
  layer.trainable = False
model.load_weights('D:\\AI\\Project\\weights_resnet_acc.hdf5')
