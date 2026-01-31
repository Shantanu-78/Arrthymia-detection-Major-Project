import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Conv1D, LayerNormalization, MultiHeadAttention,
    Dropout, Input, GlobalAveragePooling1D, Lambda, Flatten, Reshape
)
from tensorflow.keras.models import Model

class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, window_size, ff_dim, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.attn = MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout
        )
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential(
            [
                Dense(ff_dim, activation="gelu"),
                Dense(embed_dim),
            ]
        )
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, training=False):
        x = self.norm1(inputs)

        attn_output = self.attn(x, x)
        attn_output = self.dropout1(attn_output, training=training)

        x = inputs + attn_output

        ffn_output = self.norm2(x)

        ffn_output = self.ffn(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)

        return x + ffn_output

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate,
        })
        return config

def create_model(input_shape, num_classes):
    embed_dim = 64
    num_heads = 2
    window_size = 4
    ff_dim = 128
    dropout_rate = 0.1

    inputs = Input(shape=input_shape, name="input_ecg")

    x = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same', name="cnn_conv1")(inputs)
    x = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same', name="cnn_conv2")(x)
    cnn_output = GlobalAveragePooling1D(name="cnn_global_avg_pool")(x)

    # Replaced Lambda layer with Reshape for better serialization compatibility
    x_swin_input = Reshape((1, 64), name="reshape_for_swin")(cnn_output)

    x = SwinTransformerBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        window_size=window_size,
        ff_dim=ff_dim,
        dropout=dropout_rate,
        name="swin_transformer_block_1"
    )(x_swin_input)

    x = Flatten(name="flatten_after_swin_block")(x)

    x = Dense(64, activation='relu', name="classifier_dense1")(x)
    x = Dropout(dropout_rate, name="classifier_dropout1")(x)

    outputs = Dense(num_classes, activation='softmax', name="output_softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="cnn_swin_transformer_model_optimized")
    return model
