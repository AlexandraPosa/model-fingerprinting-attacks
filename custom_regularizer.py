from keras.regularizers import Regularizer
import tensorflow as tf
import numpy as np

class CustomRegularizer(Regularizer):

    def __init__(self, strength, embed_dim, seed=0, apply_penalty=True):
        self.seed = seed
        self.strength = strength
        self.embed_dim = embed_dim
        self.apply_penalty = apply_penalty
        self.signature = None
        self.coefficient = None
        self.proj_matrix = None
        self.orthogonal_matrix = None

    def __call__(self, weights):
        self.weights = weights

        # set a seed
        np.random.seed(self.seed)

        # define the code vector
        self.signature = np.ones((self.embed_dim, 1))

        # compute the linear mapping
        self.coefficient = 2 * self.signature - 1

        # build the orthogonal matrix:
        aux_matrix = np.random.rand(self.embed_dim, self.embed_dim)
        self.orthogonal_matrix, _ = np.linalg.qr(aux_matrix)             # Perform QR decomposition

        # test the orthogonality of the matrix
        dot_product = np.dot(self.orthogonal_matrix, self.orthogonal_matrix.T)
        is_orthogonal = np.allclose(dot_product, np.identity(len(dot_product)))

        if not is_orthogonal:
            raise ValueError('Matrix is not orthogonal')

        # apply a basis transformation to the code vector
        fingerprint = np.dot(self.orthogonal_matrix, self.coefficient)

        # build the projection matrix for the watermark embedding
        mat_cols = np.prod(weights.shape[0:3])
        mat_rows = self.embed_dim
        self.proj_matrix = np.random.randn(mat_rows, mat_cols)

        # prepare the weights for computation
        weights_mean = tf.reduce_mean(weights, axis=3)
        weights_flat = tf.reshape(weights_mean, (tf.size(weights_mean), 1))

        tf_fingerprint = tf.constant(fingerprint, dtype=tf.float32)
        tf_proj_mat = tf.constant(self.proj_matrix, dtype=tf.float32)

        # compute the mean squared error
        regularized_loss = self.strength * tf.reduce_mean(tf.square(tf.subtract(
            tf_fingerprint, tf.matmul(tf_proj_mat, weights_flat))))

        # apply a penalty to the loss function
        if self.apply_penalty:
            return regularized_loss

        return None

    def get_matrix(self):
        return self.proj_matrix, self.orthogonal_matrix

    def get_signature(self):
        return self.signature.reshape(self.embed_dim,), self.coefficient.reshape(self.embed_dim,)

    def get_config(self):
        return {'strength': self.strength}



def show_encoded_signature(model):
    for i, layer in enumerate(model.layers):
        try:
            if isinstance(layer.kernel_regularizer, CustomRegularizer):

                # retrieve the weights
                weights = layer.get_weights()[0]
                weights_mean = weights.mean(axis=3)
                weights_flat = weights_mean.reshape(1, weights_mean.size)

                # retrieve the projection matrix
                proj_matrix = layer.kernel_regularizer.get_matrix()

                # extract the watermark from the layer
                watermark = 1 / (1 + np.exp(-np.dot(weights_flat, proj_matrix)))

                # print the signature
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
