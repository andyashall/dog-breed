import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing import image
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

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=1001)

# Define feature columns
feature_columns = [
  tf.feature_column.numeric_column('img', shape=(200, 200, 3)),
]

# Create model
model = tf.estimator.DNNClassifier(
  hidden_units  = [120, 240, 120],
  feature_columns=feature_columns,
  n_classes=120,
)

# Input for training
train_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={
    'img': X_train,
  },
  y=y_train,
  num_epochs=None,
  shuffle=True
)

# Traing the model
model.train(train_input_fn, steps=100)

# Input for testing
test_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={
    'img': X_test,
  },
  y=y_test,
  num_epochs=1,
  shuffle=False
)

# Get accuracy and print
accuracy_score = model.evaluate(input_fn=test_input_fn)['accuracy']
print(f'Acc: {accuracy_score}')

# Input for prediction
pred_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={
    'img': test,
  },
  num_epochs=1,
  shuffle=False
)

pred = model.predict(pred_input_fn)

preds = list(pred)

# Need to convert cat codes to names and match ids

sub = pd.DataFrame(columns=labels['breed'].unique())

n = 0
for p in preds:
  sub.loc[n] = p['probabilities']
  n += 1

sub['id'] = test_ids

print(sub.head())

