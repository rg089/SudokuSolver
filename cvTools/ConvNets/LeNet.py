from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D


class LeNet:
    @staticmethod
    def build(width, height, depth, num_classes):
        input_shape = (width, height, depth)
        model = Sequential([
            Conv2D(20, (5, 5), padding="same", activation="relu", input_shape=input_shape),
            MaxPooling2D((2, 2), (2, 2)),
            Conv2D(50, (5, 5), padding="same", activation="relu"),
            MaxPooling2D((2, 2), (2, 2)),
            Flatten(),
            Dense(256, activation="relu"),
            Dense(num_classes, activation="softmax")
        ])

        return model
