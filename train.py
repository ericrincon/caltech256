import os

from models import build_model

from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.applications.inception_resnet_v2 import preprocess_input as irv2_preprocess
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess

from keras.callbacks import (
    TensorBoard,
    ModelCheckpoint
)

import util as file_util
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score
from keras.preprocessing.image import Iterator

class PreproccessImageGenerator(ImageDataGenerator):
    """
    Custom generator class for preprocessing images when using ImageDataGenerator

    The super class is ImageDataGenerator because the keras fit_generator
    function requires that the generator being used is of type Iterator
    a keras class
    """

    def __init__(self, preprocessing_function, image_generator):
        """

        :param preprocessing_function:
        :param image_generator:
        """
        super(PreproccessImageGenerator, self).__init__()
        self.preprocesing_function = preprocessing_function
        self.image_generator = image_generator


    def __next__(self):
        batch_x, batch_y = self.image_generator.next()

        return self.preprocesing_function(batch_x), batch_y


def build_custom_generator(preprocessing_func, train_data, width, height, batch_size):
    generator = build_data_generator(train_data, (width, height), batch_size)

    return PreproccessImageGenerator(preprocessing_func, generator)


def build_data_generator(data_dir, image_shape, batch_size, preprocessing_func):
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
        preprocessing_function=preprocessing_func,
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

    if model_name in ['cirsv2', 'irsv2']:
        preprocessing_func = irv2_preprocess

    elif model_name in ['pvgg16', 'vgg16']:
        preprocessing_func = vgg16_preprocess
    else:
        preprocessing_func = None

    # train_generator = build_custom_generator(preprocessing_func, train_data, width, height, batch_size)
    train_generator = build_data_generator(train_data, (width, height), batch_size, preprocessing_func)


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



    history = model.fit_generator(
        train_generator,
        steps_per_epoch=650,
        epochs=epochs,
        validation_steps=800,
        workers=4,
        max_queue_size=1000,
        callbacks=[TensorBoard('logs/'), ModelCheckpoint(model_save_name + '.k', monitor='acc',
                                                         verbose=1, save_best_only=True,
                                                         mode='max')
]
    )



    history = pd.DataFrame(history.history)

    model.save(model_save_name)
    # valid_generator = build_custom_generator(preprocessing_func, test_data,
    #                                          width, height, batch_size)
    #
    valid_generator = build_data_generator(test_data, (width, height), batch_size, preprocessing_func)


    test_loss, accuracy = model.evaluate_generator(valid_generator, workers=4, max_queue_size=1000)
    # print(predictions)
    # accuracy = accuracy_score(predictions, )

    metrics = pd.DataFrame({'Accuracy': [accuracy * 100]})

    print('Accuracy: {}'.format(accuracy * 100))

    model_save_name = model_save_name.split('.')[0]
    metrics.to_csv(model_save_name + '_metrics.csv')
    history.to_csv(model_save_name+ '.csv')

