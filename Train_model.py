from os import listdir
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import Build_Model

raw_folder = "data/"
Build_Model.save_data(raw_folder)
X,y = Build_Model.load_data()
X_train,X_test,y_train,y_test=Build_Model.Chia_Data(X,y,0.2)
Build_Model.Train_Model(4,X_train,X_test,y_train,y_test,50)           # 4 là số đầu ra cần nhận dạng 


