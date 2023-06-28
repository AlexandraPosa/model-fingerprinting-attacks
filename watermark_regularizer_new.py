import tensorflow as tf
import numpy as np


class WatermarkRegularizer(tf.keras.regularizers.Regularizer):

    def __init__(self, strength, embed_dim, seed=0, apply_penalty=True):
        self.strength = strength
        self.embed_dim = embed_dim
        self.seed = seed
        self.matrix = None
        self.signature = None
        self.apply_penalty = apply_penalty

    def __call__(self, weights):
        self.weights = weights

        # define the watermark
        self.signature = np.ones((1, self.embed_dim))

        # set a seed
        np.random.seed(self.seed)

        # build the projection matrix for the watermark embedding
        mat_rows = np.prod(weights.shape[0:3])
        mat_cols = self.signature.shape[1]
        self.matrix = np.random.randn(mat_rows, mat_cols)

        # compute cross-entropy loss
        weights_mean = tf.reduce_mean(weights, axis=3)
        weights_flat = tf.reshape(weights_mean, (1, tf.size(weights_mean)))
        proj_matrix = tf.convert_to_tensor(self.matrix, dtype=tf.float32)

        regularized_loss = self.strength * tf.reduce_sum(
            tf.keras.losses.binary_crossentropy(
                tf.sigmoid(tf.matmul(weights_flat, proj_matrix)), self.signature))

        # apply a penalty to the loss function
        if self.apply_penalty:
            return regularized_loss

        return None

    def get_matrix(self):
        return self.matrix

    def get_signature(self):
        return self.signature

    def get_config(self):
        return {'strength': self.strength}

def get_watermark_regularizers(model):
    return_list = []

    for i, layer in enumerate(model.layers):
        try:
            if isinstance(layer.kernel_regularizer, WatermarkRegularizer):
                return_list.append((i, layer.kernel_regularizer))
        except AttributeError:
            continue

    return return_list


def show_encoded_watermark(model):
    for i, layer in enumerate(model.layers):
        try:
            if isinstance(layer.kernel_regularizer, WatermarkRegularizer):

                # retrieve the weights
                weights = layer.get_weights()[0]
                weights_mean = weights.mean(axis=3)
                weights_flat = weights_mean.reshape(1, weights_mean.size)

                # retrieve the projection matrix
                proj_matrix = layer.kernel_regularizer.get_matrix()

                # extract the watermark from the layer
                #watermark = tf.sigmoid(tf.matmul(tf.constant(weights_flat, dtype=tf.float32),
                #                                 tf.constant(proj_matrix, dtype=tf.float32)))
                watermark = 1 / (1 + np.exp(-np.dot(weights_flat, proj_matrix)))

                # print the watermark
                print('\nWatermark:')
                print('Layer Index = {} \nClass = {} \n{}\n'.format(i, layer.__class__.__name__, watermark))

                # compute confidence levels
                confidence_levels = [0.5, 0.7, 0.8, 0.9]

                for level in confidence_levels:
                    confidence = (watermark > level).astype(int)
                    ones = np.count_nonzero(confidence)
                    zeros = confidence.size - ones

                    print("Confidence level: {}%".format(int(level * 100)))
                    print("Number of ones: {}\nNumber of zeros: {}\n{}\n".format(ones, zeros, confidence))

        except AttributeError:
            continue  # Continue the loop if the layer has no regularizers
