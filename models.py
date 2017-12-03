from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense


def lenet(model_input):
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
        pooling = MaxPooling2D()(convolution)

        return pooling


    layer_one = build_conv_layer(model_input, 50, 3)
    layer_two = build_conv_layer(layer_one, 50, 3)
    flattened_layer = Flatten()(layer_two)
    output = Dense(activation='softmax')(flattened_layer)

    return output


_MODELS = {
    'lenet': lenet
}


def _build_model(model, model_shape, loss='categorical_crossentropy',
                 optimizer='adam'):
    model_input = Input(shape=model_shape)
    model_output = model(model_input)

    model = Model(inputs=model_input, outputs=model_output)
    model.compile(optimizer=optimizer, loss=loss)

    return model




def build_model(model_name, model_shape):
    if model_name not in  _MODELS:
        raise ValueError('Model {} is not defines!'.format(model_name))
    else:
        model = _MODELS[model_name]

        return _build_model(model, model_shape)


