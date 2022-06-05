import tensorflow.keras as keras

def build_model(input_shape, learning_rate, num_keywords, error = 'sparse_categorical_crossentropy', print_summary = 1):

    # build network
    model = keras.Sequential()

    # conv layer 1
    model.add(keras.layers.Conv2D(  64, 
                                    kernel_size = (3,3),
                                    activation = 'relu',
                                    input_shape = input_shape,
                                    kernel_regularizer = keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3,3) , strides = (2,2), padding = 'same'))

    # conv layer 2
    model.add(keras.layers.Conv2D(  32, 
                                    kernel_size = (3,3),
                                    activation = 'relu',
                                    kernel_regularizer = keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3,3) , strides = (2,2), padding = 'same'))

    # conv layer 3
    model.add(keras.layers.Conv2D(  32, 
                                    kernel_size = (2,2),
                                    activation = 'relu',
                                    kernel_regularizer = keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2,2) , strides = (2,2), padding = 'same'))

    # flatten the output feed it into a dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units = 64, activation = 'relu',))
    model.add(keras.layers.Dropout(rate = 0.3))

    # softmax classifier
    model.add(keras.layers.Dense(units = num_keywords, activation = 'softmax')) 

    # compile model
    optimizer = keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(optimizer = optimizer, loss = error , metrics = ['accuracy'])

    # print model overview
    if print_summary == 1:
        model.summary()

    return model
    