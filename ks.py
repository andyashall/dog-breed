import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping
import os

# Import the data
path = './data/train/'
size = (300, 300)
x_train = pd.DataFrame(columns=['id', 'img'])
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

print(np.array(X_train['img']).shape)

# train.to_csv('./data/train.csv', index=False)

base_model = VGG19(
  weights = None,
  include_top=False,
  input_shape=(300, 300, 3)
)

# Add a new top layer
x = base_model.output
x = Flatten()(x)
predictions = Dense(120, activation='softmax')(x)

# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# First: train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False

model.compile(
  loss='categorical_crossentropy', 
  optimizer='adam', 
  metrics=['accuracy']
)

callbacks_list = [EarlyStopping(monitor='val_acc', patience=3, verbose=1)]

model.summary()

model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test), verbose=1)

