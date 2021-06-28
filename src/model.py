from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import Precision
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow import random


def malaria_cnn(lr=0.01):

    # Set tensorflow seed to 0
    random.set_seed(0)

    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(130, 130, 3)))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=lr),
                  metrics=['accuracy', Precision()])

    return model


def train(model: Sequential,
          train_gen: ImageDataGenerator,
          validation_gen: ImageDataGenerator,
          epochs: int,
          steps_per_epoch: int,
          validation_steps: int):

    lrs = LearningRateScheduler(lambda epoch: 1e-3/(10**(epoch/20)))
    print('Training starts:')
    history = model.fit(train_gen,
                        validation_data=validation_gen,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        validation_steps=validation_steps,
                        callbacks=[lrs],
                        verbose=1)

    return history
