import os

from models import build_model

from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from keras.callbacks import TensorBoard
import pandas as pd

def build_data_generator(data_dir, image_shape, batch_size):
    """

    :param data_dir:
    :param image_shape:
    :param batch_size:
    :return:
    """
    # from https://keras.io/preprocessing/image/
    data_generator = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-6,
        rotation_range=0.,
        width_shift_range=0.,
        height_shift_range=0.,
        shear_range=0.,
        zoom_range=0.,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format=K.image_data_format())

    generator = data_generator.flow_from_directory(
        data_dir,
        target_size=image_shape,
        batch_size=batch_size,
        class_mode='categorical')

    return generator


def run(model_name, image_shape, train_data, test_data, lr,
          batch_size, epochs, model_save_name=None):
    width, height, n_channels = image_shape

    model = build_model(model_name, image_shape, optimizer_params={'lr': lr})

    train_generator = build_data_generator(train_data, (width, height), batch_size)
    valid_generator = build_data_generator(test_data, (width, height), batch_size)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=650,
        epochs=epochs,
        validation_steps=800,
        workers=4,
        max_queue_size=256,
        callbacks=[TensorBoard('logs/')]
    )


    if model_save_name is None:
        model_save_name = model_name + '.h5'

        # save model
        while os.path.exists(model_save_name):
            if not '_' in model_save_name:
                model_save_name = model_save_name.split('.')[0] + '_1.h5'
            else:
                name, number = model_save_name.split('_')
                number = int(number.split('.')[0])
                number += 1

                model_save_name = name + '_' + str(number) + '.h5'


    history = pd.DataFrame(history.history)


    model.save(model_save_name)

    test_loss, accuracy = model.evaluate_generator(valid_generator, workers=4, max_queue_size=128)
    metrics = pd.DataFrame({'Accuracy': [accuracy * 100]})

    model_save_name = model_save_name.split('.')[0]
    metrics.to_csv(model_save_name + '_metrics.csv')
    history.to_csv(model_save_name+ '.csv')

