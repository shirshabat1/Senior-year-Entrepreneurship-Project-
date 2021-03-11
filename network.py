#
import penny_files as p
import carmen_files as c
import menta_files as m
import execute as exe
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

model = models.Sequential()
model.add(layers.Conv2D(12, (6, 20), stride=1, activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
