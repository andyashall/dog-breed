import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing import image
import os

# Import the data
path = './data/train/'
size = (250, 250)
train = pd.DataFrame(columns=['id', 'img'])
labels = pd.read_csv('./data/labels.csv', index_col=0)

print(labels.head())

for f in os.listdir(path):
  img = image.load_img(path + f, target_size=size) 
  img = np.array(image.img_to_array(img))
  l = labels.loc[f.replace('.jpg', ''), 'breed']
  train = train.append({'id': f.replace('.jpg', ''), 'img': img, 'target': l}, ignore_index=True)

print(train.head())

train['target'] = train['target'].astype('category')
x = train.drop(['id', 'target'], 1)
y = train['target'].cat.codes.astype(np.int_)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=1001)

# train['img'] = train['img'].apply(list).apply(pd.Series).astype(np.float32)

print(np.array(X_train['img'])[0].shape)

# train.to_csv('./data/train.csv', index=False)
