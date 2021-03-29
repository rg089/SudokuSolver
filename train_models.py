from cvTools.ConvNets.DigitNet import DigitNet
from cvTools.ConvNets.LeNet import LeNet
import imutils
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import numpy as np
from Generator import DigitGenerator

datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.20, width_shift_range=0.1, height_shift_range=0.1)
train_digitnet = False

if train_digitnet:
    for i in range(2, 6):
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data("mnist.npz")
        digit_gen = DigitGenerator(samples_per_digit=1000, width=28, height=28, font_size=17, test_split=0.13)
        X_train_gen, X_test_gen, y_train_gen, y_test_gen = digit_gen.generate_digits()

        X_train = np.concatenate((X_train, X_train_gen));
        X_test = np.concatenate((X_test, X_test_gen));
        y_train = np.concatenate((y_train, y_train_gen));
        y_test = np.concatenate((y_test, y_test_gen))

        X_train, X_test = imutils.normalize(X_train, X_test)
        X_train, X_test = imutils.addDimension(X_train, X_test)
        encoder, y_train, y_test = imutils.encodeY(y_train, y_test)

        # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

        checkpoint = ModelCheckpoint(f"models\\digitnet_mnist_augment_decay_{i}.h5", monitor="val_accuracy",
                                     save_best_only=True, verbose=1)
        annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

        epochs = 45

        opt = Adam()
        model = DigitNet.build(28, 28, 1, 10)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        print("Model Built.\nTraining model....")

        history = model.fit(datagen.flow(X_train, y_train, batch_size=64), validation_data=(X_test, y_test),
                            epochs=epochs,
                            batch_size=64, steps_per_epoch=X_train.shape[0] // 64, verbose=1, callbacks=[checkpoint])

        print("Evaluating Model...")
        predictions = model.predict(X_test, batch_size=64)
        print(classification_report(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1),
                                    target_names=list(map(str, encoder.classes_))))

        plt = imutils.plot_model(history, epochs)
        plt.savefig(f"plots\\digitnet_training_mnist_augment_decay_{i}.png")

else:
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data("mnist.npz")
    train_mask = y_train != 0; test_mask = y_test != 0
    X_train = X_train[train_mask]; y_train = y_train[train_mask]; X_test = X_test[test_mask]; y_test = y_test[test_mask]
    digit_gen = DigitGenerator(samples_per_digit=3500, width=28, height=28, font_size=17, test_split=0.13)
    X_train_gen, X_test_gen, y_train_gen, y_test_gen = digit_gen.generate_digits(includeZero=False)
    print("[INFO] Artificial Samples Generated.")
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    X_train = np.concatenate((X_train, X_train_gen));
    X_test = np.concatenate((X_test, X_test_gen));
    y_train = np.concatenate((y_train, y_train_gen));
    y_test = np.concatenate((y_test, y_test_gen))

    X_train, X_test = imutils.normalize(X_train, X_test)
    X_train, X_test = imutils.addDimension(X_train, X_test)
    encoder, y_train, y_test = imutils.encodeY(y_train, y_test)

    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

    epochs = 20

    checkpoint = ModelCheckpoint("models\\lenet_mnist_augment_decay.h5", monitor="val_accuracy",
                                 save_best_only=True, verbose=1)

    opt = Adam()
    model = LeNet.build(28, 28, 1, 9)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    print("Model Built.\nTraining model....")

    history = model.fit(datagen.flow(X_train, y_train, batch_size=64), validation_data=(X_test, y_test), epochs=epochs,
                        batch_size=64, steps_per_epoch=X_train.shape[0] // 64, verbose=1, callbacks=[checkpoint])

    print("Evaluating Model...")
    predictions = model.predict(X_test, batch_size=64)
    print(classification_report(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1),
                                target_names=list(map(str, encoder.classes_))))

    plt = imutils.plot_model(history, epochs)
    plt.savefig(f"plots\\lenet_training_mnist_augment_decay.png")
