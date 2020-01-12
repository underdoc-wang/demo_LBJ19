from keras.models import Model
from keras.layers import Input, Activation, add
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D

n_res_units = 2

def _residual_unit():
    def f(input):
        residual = BatchNormalization()(input)
        residual = Activation('relu')(residual)
        residual = Conv2D(32, (3, 3), padding='same')(residual)
        residual = BatchNormalization()(residual)
        residual = Activation('relu')(residual)
        residual = Conv2D(32, (3, 3), padding='same')(residual)
        return add([input, residual])
    return f

def res_units(repetations):
    def f(input):
        for i in range(repetations):
            input = _residual_unit()(input)
        return input
    return f

def st_resnet(close_input):
    input = Input(shape=close_input)
    x = Conv2D(32, (3, 3), padding='same')(input)
    x = res_units(n_res_units)(x)
    x = Activation('relu')(x)
    output = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
    model = Model(inputs=input, outputs=output)
    return model

