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
labels = pd.read_csv('./data/labels.csv', index_col=0)
labels['breed'] = labels['breed'].astype('category')

print(labels.head())

for f in os.listdir(path):
  img = image.load_img(path + f, target_size=size) 
  img = image.img_to_array(img)
  l = labels.loc[f.replace('.jpg', ''), 'breed']
  train = train.append({'id': f, 'img': img, 'target': l}, ignore_index=True)

print(train.head())

x = train.drop(['id', 'target'], 1)
y = train['target']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=1001)

print(len(np.array(X_train['img'])[0]))

# Define feature columns
feature_columns = [
  tf.feature_column.numeric_column('img', shape=[250]),
]