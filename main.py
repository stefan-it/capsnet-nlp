from keras.layers import K, Activation
from keras.engine import Layer
from keras.layers import LeakyReLU, Dense, Input, Embedding, Dropout, Bidirectional, GRU, Flatten, SpatialDropout1D
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Model


gru_len = 256
Routings = 3
Num_capsule = 10
Dim_capsule = 16
dropout_p = 0.25
rate_drop_dense = 0.28

max_features = 20000
maxlen = 1000
embed_size = 256


def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale

# Notice: the base implementation is taken from
# https://www.kaggle.com/chongjiujjin/capsule-net-with-gru
# Modifications and refactoring are in progress.
class Capsule(Layer):
    def __init__(
            self,
            num_capsule=10,
            dim_capsule=16,
            routings=3,
            share_weights=True,
            activation='default',
            **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(
                name='capsule_kernel',
                shape=(
                    1,
                    input_dim_capsule,
                    self.num_capsule *
                    self.dim_capsule),
                initializer='he_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(
                name='capsule_kernel',
                shape=(
                    input_num_capsule,
                    input_dim_capsule,
                    self.num_capsule *
                    self.dim_capsule),
                initializer='he_uniform',
                trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(
            u_hat_vecs,
            (batch_size,
             input_num_capsule,
             self.num_capsule,
             self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule,
        # dim_capsule]

        # shape = [None, num_capsule, input_num_capsule]
        b = K.zeros_like(u_hat_vecs[:, :, :, 0])
        for i in range(self.routings):
            # shape = [None, input_num_capsule, num_capsule]
            b = K.permute_dimensions(b, (0, 2, 1))
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


def get_model():
    input1 = Input(shape=(maxlen,))
    embed_layer = Embedding(max_features,
                            embed_size,
                            input_length=maxlen)(input1)
    embed_layer = SpatialDropout1D(rate_drop_dense)(embed_layer)

    x = Bidirectional(GRU(gru_len,
                          activation='relu',
                          dropout=dropout_p,
                          recurrent_dropout=dropout_p,
                          return_sequences=True))(embed_layer)
    capsule = Capsule(
        num_capsule=Num_capsule,
        dim_capsule=Dim_capsule,
        routings=Routings,
        share_weights=True)(x)

    capsule = Flatten()(capsule)
    capsule = Dropout(dropout_p)(capsule)
    capsule = LeakyReLU()(capsule)

    x = Flatten()(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input1, outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()

    return model


def load_imdb(maxlen=1000):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=maxlen)
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    return x_train, y_train, x_test, y_test


def main():
    x_train, y_train, x_test, y_test = load_imdb()

    model = get_model()

    batch_size = 32
    epochs = 40

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test))


if __name__ == '__main__':
    main()
