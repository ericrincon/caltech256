from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
from keras.applications.inception_resnet_v2 import InceptionResNetV2

_opts = {
    'adam': Adam
}


def get_opt(opt_name, *args, **kwargs):
    """
    Simple factory function for getting different keras optmizers
    :return:
    """

    if opt_name not in _opts:
        raise ValueError("Optimizer \"{}\" not valid!".format(opt_name))
    else:
        return _opts[opt_name](*args, **kwargs)



def custom_pretrained_inception_resnetV2(model_input):
    flat_irv2_output = pretrained_inception_resnetV2(model_input)

    dense = Dense(256, activation='relu')(flat_irv2_output)

    return dense


def pretrained_inception_resnetV2(model_input, model_shape, *args, **kwargs):
    model_irsv2 = InceptionResNetV2(weights='imagenet',input_shape=model_shape,
                                    include_top=False)

    irv2_output = model_irsv2(model_input)
    flat_irv2_output = Flatten()(irv2_output)

    return flat_irv2_output

def pretrained_vgg16(model_input, *args, **kwargs):
    """
    Use the vgg16 trained on imagenet to train on new dataset
    note use a smaller learning rate to fine tune weights
    :param model_input:
    :return:
    """
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

    vgg_output = model_vgg16_conv(model_input)
    flat_vgg_output = Flatten()(vgg_output)
    dense = Dense(256, activation='relu')(flat_vgg_output)

    return dense


def vggnet(model_input, *args, **kwargs):
    """
    VGG Style network note the parameters have been reduced significantly
    :param model_input:
    :return:
    """

    def build_conv_layer(layer_input, nb_convs, nb_filters, kernel_size, strides=1):
        """

        :param layer_input:
        :param nb_filters:
        :param kernel_size:
        :param strides:
        :return:
        """
        convolution = layer_input

        for i in range(nb_convs):
            convolution = Conv2D(filters=nb_filters, kernel_size=kernel_size,
                                 strides=strides, activation='relu')(convolution)
        pooling = MaxPooling2D(pool_size=kernel_size, strides=2)(convolution)

        return pooling

    layer_one = build_conv_layer(model_input, 2, 64, (3,3))
    layer_two = build_conv_layer(layer_one, 2, 64, (3, 3))
    layer_three = build_conv_layer(layer_two, 2, 64, (3,3))
    layer_four = build_conv_layer(layer_three, 3, 64, (3, 3))
    # layer_five = build_conv_layer(layer_four, 3, 512, (3, 3))

    # Fully connected layers
    flatten = Flatten()(layer_four)
    dense_one = Dense(256)(flatten)
    last_layer = Dense(256)(dense_one)

    return last_layer


def lenet(model_input, *args, **kwargs):
    """

    :return:
    """

    def build_conv_layer(layer_input, nb_filters, kernel_size, strides=(2,2)):
        """
        Builds one layer of a LeNet style CNN
        INPUT => CONV => RELU => POOL => CONV => RELU => POOL => FC => RELU => FC
        :param layer_input:
        :return:
        """

        convolution = Conv2D(filters=nb_filters, kernel_size=kernel_size,
                             strides=strides, activation='relu')(layer_input)
        pooling = MaxPooling2D(pool_size=(2, 2))(convolution)

        return pooling

    layer_one = build_conv_layer(model_input, 50, 3)
    layer_two = build_conv_layer(layer_one, 50, 3)
    last_layer = Flatten()(layer_two)

    return last_layer


_MODELS = {
    'lenet': lenet,
    'vggnet': vggnet,
    'pvgg16': pretrained_vgg16,
    'irsv2': pretrained_inception_resnetV2

}


def get_available_models():
    return _MODELS.keys()


def _build_model(model, input_shape, loss='categorical_crossentropy',
                 optimizer='adam', optimizer_params=None):

    model_input = Input(shape=input_shape)
    last_model_layer = model(model_input, input_shape)
    model_output = Dense(260, activation='softmax')(last_model_layer)

    optimizer = get_opt(optimizer, **optimizer_params)

    model = Model(inputs=model_input, outputs=model_output)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model



def build_model(model_name, model_shape, optimizer_params):
    if model_name not in  _MODELS:
        raise ValueError('Model {} is not defined!'.format(model_name))
    else:
        model = _MODELS[model_name]

        return _build_model(model, model_shape, optimizer_params=optimizer_params)


