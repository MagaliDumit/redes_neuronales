import tensorflow as tf
from tensorflow.keras import layers, Model


def build_autoencoder(input_dim=35):

    encoder_input = layers.Input(shape=(input_dim,), name='encoder_input')

    x = layers.Dense(17, activation='relu', name='enc_h1')(encoder_input)
    x = layers.Dense(7, activation='relu', name='enc_h2')(x)

    latent = layers.Dense(2, activation='tanh', name='latent')(x)

    encoder = Model(encoder_input, latent, name='encoder')

    decoder_input = layers.Input(shape=(2,), name='decoder_input')

    x = layers.Dense(7, activation='relu', name='dec_h1')(decoder_input)
    x = layers.Dense(17, activation='relu', name='dec_h2')(x)

    decoder_output = layers.Dense(input_dim, activation='sigmoid', name='decoder_output')(x)

    decoder = Model(decoder_input, decoder_output, name='decoder')

    autoencoder_input = layers.Input(shape=(input_dim,), name='ae_input')
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = Model(autoencoder_input, decoded, name='autoencoder')

    return autoencoder, encoder, decoder


def build_mnist_autoencoder(latent_dim=2):
    encoder_input = layers.Input(shape=(784,), name='encoder_input')
    x = layers.Dense(256, activation='relu', name='enc_h1')(encoder_input)
    x = layers.Dense(64, activation='relu', name='enc_h2')(x)
    x = layers.BatchNormalization(name='enc_bn')(x)
    latent = layers.Dense(latent_dim, activation='tanh', name='latent')(x)

    encoder = Model(encoder_input, latent, name='encoder')

    decoder_input = layers.Input(shape=(latent_dim,), name='decoder_input')
    x = layers.Dense(64, activation='relu', name='dec_h1')(decoder_input)
    x = layers.Dense(256, activation='relu', name='dec_h2')(x)
    decoder_output = layers.Dense(784, activation='sigmoid', name='decoder_output')(x)

    decoder = Model(decoder_input, decoder_output, name='decoder')

    autoencoder_input = layers.Input(shape=(784,), name='ae_input')
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = Model(autoencoder_input, decoded, name='autoencoder')

    return autoencoder, encoder, decoder
