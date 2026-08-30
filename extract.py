import warnings
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
import cv2
import h5py
import os
import json
import datetime
import time
model=config["model"]
include_top=config["include_top"]
weights=config["weights"]
train_path=config["train_path"]
features_path=config["features_path"]
label_path=config["label_path"]
results=config["results"]
test_size 		= config["test_size"]
model_path 		= config["model_path"]
if model=="vgg16":
  base=VGG16(weights=weights)
  basee=Model(input=base.input,output=base.get_layer("fc1").output)
if model=="vgg19":
  base=VGG16(weights=weights)
  basee=Model(input=base.input,output=base.get_layer("fc1").output)
if model=="resnet50":
  base=ResNet50(weights=weights)
  basee=Model(input=base.input,output=base.get_layer("flatten").output)
if model=="inceptionv3":
  base=InceptionV3(include_top=include_top,weights=weights,input_tensor=Input(shape=(229,229,3)))
  basee=Model(input=base.input,output=base.get_layer('custom').output)
  image_size=(299,299)
if model=="inceptionresnetv2":
  base=InceptionResNetV2(include_top=include_top,weights=weights,input_tensor=Input(shape=(299,299,3)))
  basee=Model(input=base.input,output=base.get_layer('custom').output)
  image_size=(299,299)
if model=="mobilenet":
  base=MobileNet(include_top=include_top,weights=weights,input_tensor=Input(shape=(299,299,3)))
  basee=Model(input=base.input,output=base.get_layer('custom').output)
  image_size=(299,299)
if model=="xception":
  base=Xception(weights=weights)
  basee=Model(input=base.input,output=base.get_layer("custom").output)
  image_size=(299,299)
else:
  base=None

  
  
