import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers



def get_model(input_size, output_size, fs=64, layers_deep=3, kernel_size=3):
    inputs = keras.Input(shape=input_size)

    x = inputs
    for _ in range(layers_deep):
        x = layers.Conv2D(fs, kernel_size, strides=1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    
    # policy_head
    x_pol = layers.Conv2D(2, 1, strides=1)(x)
    x_pol = layers.BatchNormalization()(x_pol)
    x_pol = layers.ReLU()(x_pol)
    x_pol = layers.Flatten()(x_pol)

    pol = layers.Dense(output_size, activation=None)(x_pol)

    # value head
    x_val = layers.Conv2D(1, (1, 1), strides=(1, 1))(x)
    x_val = layers.BatchNormalization()(x_val)
    x_val = layers.ReLU()(x_val)
    x_val = layers.Flatten()(x_val)

    x_val = layers.Dense(128, activation='relu')(x_val)
    val = layers.Dense(1, activation='tanh')(x_val)

    model = keras.Model(inputs=inputs, outputs=[pol, val], name='mini-zero')
    return model


if __name__ == '__main__':
    model = get_model((10, 10, 2), 100)
    print(model.summary())
    keras.utils.plot_model(model, 'mini-zero.png')