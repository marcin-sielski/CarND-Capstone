from keras.layers import Layer

class Normalization(Layer):
    def call(self, x):
        return x/127.5 - 1.0