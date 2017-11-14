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
size = (300, 300)
x = []
y = []
labels = pd.read_csv('./data/labels.csv', index_col=0)
labels['breed'] = labels['breed'].astype('category').cat.codes.astype(np.int_)

print(labels.head())

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
  l = labels.loc[f.replace('.jpg', ''), 'breed']
  x.append(img)
  y.append(l)

x = np.array(x, np.float32)
y = np.array(y, np.int_)

print(x.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=1001)

base_model = VGG19(
  weights = None,
  include_top=False,
  input_shape=(300, 300, 3)
)

# Add a new top layer
out = base_model.output
out = Flatten()(out)
predictions = Dense(120, activation='softmax')(out)

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

