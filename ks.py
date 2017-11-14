import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense
import os

# Import the data
path = './data/train/'
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

model = Sequential()

model.add(Dense(units=64, activation='relu', input_shape=(300, 300, 1)))
model.add(Dense(units=10, activation='softmax'))

model.compile(
  loss='categorical_crossentropy',
  optimizer='sgd',
  metrics=['accuracy']
)

model.fit(
  X_train['img'],
  y_train,
  batch_size=128
)


# loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)

loss_and_metrics = model.evaluate(
  X_test['img'],
  y_test,
  batch_size=128
)

print(loss_and_metrics)