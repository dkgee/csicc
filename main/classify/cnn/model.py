import keras.regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from config import Config
config = Config()


def Model(resize, classes):
    model = Sequential()
    model.add(Conv2D(filters=5, kernel_size=(4, 4), strides=(1, 1), input_shape=(resize, resize, 3),
                     padding='same', data_format='channels_last', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(filters=8, kernel_size=(4, 4), padding="same", activation="relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(classes, activation="softmax", kernel_regularizer=keras.regularizers.l1(0.01)))
    model.summary()
    return model
