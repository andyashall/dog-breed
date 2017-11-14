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

# Train data processing
for f in os.listdir(train_path):
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

# Test data processing
for f in os.listdir(test_path):
  img = image.load_img(test_path + f)
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
  test.append(img)
  test_ids.append(f.replace('.jpg', ''))

test = np.array(test, np.float32)

print(test.shape)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=1001)

print(y_test.shape)

base_model = VGG19(
  weights = None,
  include_top=False,
  input_shape=(200, 200, 3)
)

num_class = y.shape[0]

# Add a new top layer
out = base_model.output
out = Flatten()(out)
predictions = Dense(num_class, activation='softmax')(out)

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

preds = model.predict(test, verbose=1)

sub = pd.DataFrame(preds)

sub.columns = labels['breed'].unique()

sub['id'] = test_ids

sub.head(5)
