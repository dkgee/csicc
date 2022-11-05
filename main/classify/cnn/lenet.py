from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from config import Config
config = Config()


def Model(resize, classes):
    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=(5, 5),
                     strides=(1, 1),
                     input_shape=(resize, resize, 3),
                     padding='valid', activation='tanh',
                     kernel_initializer='uniform',
                     name='C1'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='S2', strides=(1, 1)))
    model.add(Conv2D(filters=16, kernel_size=(5, 5),
                     strides=(1, 1),
                     padding='valid',
                     activation='tanh',
                     kernel_initializer='uniform',
                     name='C3'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='S4'))
    model.add(Flatten())
    model.add(Dense(120, activation='tanh', name='F5'))
    model.add(Dropout(0.5))
    model.add(Dense(84, activation='tanh', name='F6'))
    model.add(Dense(classes, activation='softmax', name='Pre'))
    model.summary()
    return model

