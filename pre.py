import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense
import os

# Import the data
path = './data/ten/'
size = (300, 300)
train = pd.DataFrame(columns=['id', 'img'])
labels = pd.read_csv('./data/labels.csv', index_col=0)

print(labels.head())

for f in os.listdir(path):
  img = image.load_img(path + f)
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
  img.save(f'./data/proc/{f}')
  img = np.array(image.img_to_array(img))
  l = labels.loc[f.replace('.jpg', ''), 'breed']
  train = train.append({'id': f.replace('.jpg', ''), 'img': img, 'target': l}, ignore_index=True)

print(train.head())