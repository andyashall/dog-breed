import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense
import os

# Import the data
train_path = './data/train/'
test_path = './data/ten/'
size = (200, 200)
x = []
y = []
test = []
test_ids = []
labels = pd.read_csv('./data/labels.csv', index_col=0)
labels['breed'] = labels['breed'].astype('category')
labels['breed_code'] = labels['breed'].cat.codes.astype(np.int_)

print(labels.head())

print(labels['breed'].unique())

# Train data processing
for f in os.listdir(test_path):
  img = image.load_img(train_path + f)
  longer_side = max(img.size)
  horizontal_padding = (longer_side - img.size[0]) / 2
  vertical_padding = (longer_side - img.size[1]) / 2
  img = img.crop(
    (
      -horizontal_padding,
      -vertical_padding,
      img.size[0] + horizontal_padding,
      img.size[1] + vertical_padding
    )
  )
  img = img.resize(size)
  img = np.array(image.img_to_array(img))
  l = labels.loc[f.replace('.jpg', ''), 'breed_code']
  x.append(img)
  y.append(l)

x = np.array(x, np.float32)
y = np.array(y, np.int_)

print(x.shape)
print(y.shape)

sub = pd.DataFrame(columns=labels['breed'].unique())

print(sub)