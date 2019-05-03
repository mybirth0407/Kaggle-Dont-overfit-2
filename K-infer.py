from keras import metrics # Keras metrics method
from keras.layers import Dense # Keras perceptron neuron layer implementation.
from keras.layers import Dropout # Keras Dropout layer implementation.
from keras.layers import Activation # Keras Activation Function layer implementation.
from keras.models import Sequential # Keras Model object.
from keras import optimizers # Keras Optimizer for custom user.
from keras import losses # Keras Loss for custom user.
from keras.models import load_model

import numpy as np # linear algebra
from sklearn.preprocessing import MinMaxScaler # Feature scaling

import os
import sys


def main(argv):
  X_test = load_dataset('test.csv')

  model = load_model(argv[1])

  Y_pred = model.predict(X_test)
  Y_pred = list(map(int, Y_pred > 0.5))
  f = open(argv[1] + '.csv', 'w', encoding='utf-8')
  f.write('id,target\n')
  for i in range(len(Y_pred)):
    f.write(str(i + 250) + ',' + str(Y_pred[i]) + '\n')


def load_dataset(test_file):
  scaler = MinMaxScaler()

  with open(test_file, 'r') as f:
    lines = f.readlines()

    X_test = []
    for line in lines[1:]:
      l = line.split(',')
      x = list(map(float, l[1:]))
      X_test.append(x)

  scaler.fit(X_test)
  X_test = scaler.transform(X_test)
  X_test = np.array(X_test)

  return X_test


if __name__ == '__main__':
  main(sys.argv)
