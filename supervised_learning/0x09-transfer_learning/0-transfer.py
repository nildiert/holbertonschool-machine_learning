#!/usr/bin/env python3
"""Transfer learning"""

import tensorflow.keras as K
import numpy as np
import matplotlib.pyplot as plt


def preprocess_data(X, Y):
    """
    This method pre-processes the data
    for the model
    """
    X_p = K.applications.vgg19.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, num_classes=10)
    return (X_p, Y_p)


if __name__ == '__main__':

    """
    Transfer learning of the model VGG19
    and save it in a file cifar10.h5
    """
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    learn_rate = .001
    batch_size = 256
    epochs = 1

    x_train = K.applications.vgg19.preprocess_input(x_train)
    x_test = K.applications.vgg19.preprocess_input(x_test)

    y_train = K.utils.to_categorical(y_train, 10)
    y_test = K.utils.to_categorical(y_test, 10)

    train_generator = K.preprocessing.image.ImageDataGenerator(
        rotation_range=2,
        horizontal_flip=True,
        zoom_range=.1)

    val_generator = K.preprocessing.image.ImageDataGenerator(
        rotation_range=2,
        horizontal_flip=True,
        zoom_range=.1)

    test_generator = K.preprocessing.image.ImageDataGenerator(
        rotation_range=2,
        horizontal_flip=True,
        zoom_range=.1)

    train_generator.fit(x_train)
    test_generator.fit(x_test)

    lrr = K.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=.01,
        patience=3,
        min_lr=1e-5)

    base_model_1 = K.applications.VGG19(
        include_top=False,
        weights='imagenet',
        input_shape=(32, 32, 3),
        classes=y_train.shape[1])

    for layer in base_model_1.layers[0:3]:
        layer.trainable = False

    sgd = K.optimizers.SGD(
        lr=learn_rate,
        momentum=.9,
        nesterov=False)
    save = K.callbacks.ModelCheckpoint('cifar10.h5',
                                       monitor='val_accuracy',
                                       save_best_only=True)

    model_1 = K.Sequential()
    model_1.add(base_model_1)
    model_1.add(K.layers.Flatten())

    model_1.add(K.layers.Dense(1024, activation=('relu'), input_dim=512))
    model_1.add(K.layers.Dense(512, activation=('relu')))
    model_1.add(K.layers.Dense(256, activation=('relu')))
    model_1.add(K.layers.Dense(128, activation=('relu')))
    model_1.add(K.layers.Dense(10, activation=('softmax')))

    model_1.compile(
        optimizer=sgd,
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    model_1.summary()

    history = model_1.fit_generator(
        train_generator.flow(x_train,
                             y_train,
                             batch_size=batch_size),
        epochs=epochs,
        steps_per_epoch=x_train.shape[0]//batch_size,
        validation_data=val_generator.flow(x_test,
                                           y_test,
                                           batch_size=batch_size),
        validation_steps=250,
        callbacks=[lrr, save], verbose=1)

    plt.plot(history.history['accuracy'], linestyle='dashed')
    plt.plot(history.history['val_accuracy'])
    plt.show()
    model_1.save('cifar10.h5')
