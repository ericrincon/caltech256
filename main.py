import argparse

from models import build_model
from train import train_model

from keras.preprocessing.image import ImageDataGenerator
from keras.backend import K

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--model')
    argument_parser.add_argument('--data')
    args = argument_parser.parse_args()

    # from https://keras.io/preprocessing/image/
    image_generator = ImageDataGenerator(
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

    image_generator.flow_from_directory(args.data)



    # model = build_model(args.model_name, )






if __name__ == '__main__':
    main()
