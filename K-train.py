from keras import metrics # Keras metrics method
from keras.layers import Dense # Keras perceptron neuron layer implementation.
from keras.layers import Dropout # Keras Dropout layer implementation.
from keras.layers import Activation # Keras Activation Function layer implementation.
from keras.models import Sequential # Keras Model object.
from keras import optimizers # Keras Optimizer for custom user.
from keras import losses # Keras Loss for custom user.
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.layers.normalization import BatchNormalization

import numpy as np # linear algebra
from sklearn.preprocessing import MinMaxScaler # Feature scaling

import os


checkpoint = ModelCheckpoint(
    filepath='./best-{epoch:003d}-{val_acc:.4f}.h5',
    monitor='val_acc',
    verbose=1,
    save_best_only=True)


def main():
  X_train, Y_train = load_dataset('train.csv')
  model = create_model(X_train.shape[1])

  history = model.fit(X_train, Y_train,
    batch_size=4,
    epochs=200,
    verbose=1,
    callbacks=[checkpoint],
    validation_split=0.2,
    shuffle=True)


def load_dataset(train_file):
  scaler = MinMaxScaler()

  with open(train_file, 'r') as f:
    lines = f.readlines()

    X_train = []
    Y_train = []

    for line in lines[1:]:
      l = line.split(',')
      y = l[1]
      x = list(map(float, l[2:]))
      X_train.append(x)
      Y_train.append(y)

  scaler.fit(X_train)
  X_train = scaler.transform(X_train)

  X_train = np.array(X_train)
  Y_train = np.array(Y_train)

  return X_train, Y_train

def plot(model):
  plot_model(
      model,
      to_file=os.path.basename(__file__) + '.png',
      show_shapes=True)

def create_model(fl, lr=0.003):
    """ Create Neural Networks Model
      @fl: feature length(num of input features)
      @lr: learning rate(step size), default value=0.001
    """
    model = Sequential()
    model.add(Dense(fl, input_dim=fl, kernel_initializer='uniform', activation='sigmoid'))
    model.add(Dense(50, kernel_initializer='uniform', activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1)
    # plot(model)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
  main()
