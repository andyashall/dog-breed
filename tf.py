import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing import image
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

train['target'] = train['target'].astype('category')
x = train.drop(['id', 'target'], 1)
y = train['target'].cat.codes.astype(np.int_)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=1001)

# train['img'] = train['img'].apply(list).apply(pd.Series).astype(np.float32)

print(np.array(X_train['img'])[0].shape)

# train.to_csv('./data/train.csv', index=False)

# Define feature columns
feature_columns = [
  tf.feature_column.numeric_column('img', shape=[1, 120]),
]

# Create model
model = tf.estimator.DNNClassifier(
  hidden_units  = [30, 60, 30],
  feature_columns=feature_columns,
  n_classes=120,
)

# Input for training
train_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={
    'img': np.array(X_train.img.as_matrix()),
  },
  y=np.array(y_train.as_matrix()),
  num_epochs=None,
  shuffle=True
)

# Traing the model
model.train(train_input_fn, steps=100)

# Input for testing
test_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={
    'img': np.array(X_test.img.as_matrix()),
  },
  y=np.array(y_test.as_matrix()),
  num_epochs=1,
  shuffle=False
)

# Get accuracy and print
accuracy_score = model.evaluate(input_fn=test_input_fn)['accuracy']
print(f'Acc: {accuracy_score}')