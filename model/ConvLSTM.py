from keras.models import Sequential
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D


def convlstm(seq_input):
    model = Sequential()

    # 1-layer convlstm
    model.add(ConvLSTM2D(filters = 1, kernel_size = (3, 3), input_shape = seq_input,
                       activation = 'relu', padding = 'same', return_sequences = False))

    '''
    # 1-layer convlstm + BN + feature extraction
    model.add(ConvLSTM2D(filters = 32, kernel_size = (3, 3), input_shape = seq_input,
                       padding = 'same', return_sequences = False))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 1, kernel_size=(1, 1), padding='same', activation='relu'))

    # 2-layer convlstm
    model.add(ConvLSTM2D(filters = 32, kernel_size = (3, 3), input_shape = seq_input,
                       padding = 'same', return_sequences = True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters = 1, kernel_size = (3, 3), activation = 'relu',
                       padding = 'same', return_sequences = False))
    
    # 4-layer convlstm + feature extraction -> disperse result
    model.add(ConvLSTM2D(filters = 32, kernel_size = (3, 3), input_shape = seq_input,
                       padding = 'same', return_sequences = True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters = 32, kernel_size = (3, 3),
                       padding = 'same', return_sequences = True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters = 32, kernel_size = (3, 3),
                       padding = 'same', return_sequences = True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters = 32, kernel_size = (3, 3),
                       padding = 'same', return_sequences = False))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 1, kernel_size = (1, 1), padding = 'same', activation = 'relu'))   # feature extraction
    '''

    return model

