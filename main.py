import argparse

from models import build_model
from train import train_model

from keras.preprocessing.image import ImageDataGenerator
from keras.backend import K


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
        horizontal_flip=False,
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


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--model')
    argument_parser.add_argument('--train-data')
    argument_parser.add_argument('--test=data')
    argument_parser.add_argument('--height', default=256, type=int)
    argument_parser.add_argument('--width', default=256, type=int)
    argument_parser.add_argument('--batch_size', default=32, type=int)
    argument_parser.add_argument('--image_shape', default='256,256', type=str)
    argument_parser.add_argument('--epochs')

    args = argument_parser.parse_args()

    image_shape = [int(x) for x in args.image_shape.split(',')]

    model = build_model(args.model_name, (args.width, args.height))

    train_generator = build_data_generator(args.train_data, image_shape, args.batch_size)
    valid_generator = build_data_generator(args.valid_data, image_shape, args.batch_size)

    model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=valid_generator,
        validation_steps=800)










if __name__ == '__main__':
    main()
