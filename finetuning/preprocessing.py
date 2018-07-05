import os
import numpy as np
from keras.models import Sequential
from keras.preprocessing.image\
import ImageDataGenerator, array_to_img, img_to_array, load_img


if __name__ == '__main__':

    train_data_argumentation = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    
    test_data_argumentation = ImageDataGenerator(
        rescale=1.0 / 255
    )

    train_argumentation = train_data_argumentation.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    validation_argument = test_data_argumentation.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

tensor = model.fit_generator(
        train_argumentation,
        samples_per_epoch=2000,
        nb_epoch=nb_epoch,
        validation_data=validation_argument,
        nb_val_samples=800
    )