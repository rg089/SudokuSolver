from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout


class DigitNet:
    @staticmethod
    def build(width, height, depth, num_classes):
        input_shape = (width, height, depth)
        model = Sequential([
            Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=input_shape),
            BatchNormalization(axis=-1),
            Conv2D(32, (3, 3), padding="same", activation="relu"),
            BatchNormalization(axis=-1),
            Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'),
            Dropout(0.4),

            Conv2D(64, (3, 3), padding="same", activation="relu"),
            BatchNormalization(axis=-1),
            Conv2D(64, (3, 3), padding="same", activation="relu"),
            BatchNormalization(axis=-1),
            Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'),
            Dropout(0.4),

            Flatten(),
            Dense(128, activation="relu"),
            BatchNormalization(axis=-1),
            Dropout(0.4),
            Dense(num_classes, activation="softmax")
        ])

        return model

