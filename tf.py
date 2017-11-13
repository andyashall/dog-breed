import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing import image
import os

# Import the data
path = './data/ten/'
size = (250, 250)
train = pd.DataFrame(columns=['id', 'img'])

for f in os.listdir(path):
  img = image.load_img(path + f, target_size=size) 
  img = image.img_to_array(img)
  train = train.append({'id': f, 'img': img}, ignore_index=True)

print(train.head())