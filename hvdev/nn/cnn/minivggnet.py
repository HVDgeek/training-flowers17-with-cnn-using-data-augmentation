from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras import backend as K 

class MiniVggNet:
    @staticmethod
    def build(width , height, depth , classes):
        inputShape = (width , height , depth)
        model = Sequential()
        chaDIm = -1

        if K.image_data_format() == 'first_channels':
            inputShape = (depth , width , height)
            chaDIm = 1

        
        model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = inputShape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chaDIm))
        model.add(Conv2D(32, (3, 3), padding = "same"))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chaDIm))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chaDIm))
        model.add(Conv2D(64, (3, 3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chaDIm))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model