from keras.regularizers import Regularizer
import tensorflow as tf
import numpy as np
import h5py

class FingerprintRegularizer(Regularizer):

    def __init__(self, strength, embed_dim, num_zeros, seed=0, apply_penalty=False):
        self.seed = seed
        self.strength = strength
        self.embed_dim = embed_dim
        self.num_zeros = num_zeros
        self.apply_penalty = apply_penalty
        self.signature = None
        self.fingerprint = None
        self.proj_matrix = None
        self.orthogonal_matrix = None

    def __call__(self, weights):
        self.weights = weights

        # set a seed
        np.random.seed(self.seed)

        # define the user-specific code vector
        self.signature = np.ones(self.embed_dim, dtype=int)
        zero_indices = np.random.choice(self.embed_dim, size=self.num_zeros, replace=False)
        self.signature[zero_indices] = 0
        self.signature = self.signature.reshape(self.embed_dim, 1)

        # compute the linear mapping
        coefficient = 2 * self.signature - 1

        # build the orthogonal matrix:
        aux_matrix = np.random.randn(self.embed_dim, self.embed_dim)
        self.orthogonal_matrix, _ = np.linalg.qr(aux_matrix)             # Perform QR decomposition

        # test the orthogonality of the matrix
        dot_product = np.dot(self.orthogonal_matrix, self.orthogonal_matrix.T)
        is_orthogonal = np.allclose(dot_product, np.identity(len(dot_product)))

        if not is_orthogonal:
            raise ValueError('Matrix is not orthogonal')

        # apply a basis transformation to the code vector
        self.fingerprint = np.dot(self.orthogonal_matrix, coefficient)

        # build the projection matrix for the watermark embedding
        mat_cols = np.prod(weights.shape[0:3])
        mat_rows = self.embed_dim
        self.proj_matrix = np.random.randn(mat_rows, mat_cols)

        # prepare the weights for computation
        weights_mean = tf.reduce_mean(weights, axis=3)
        weights_flat = tf.reshape(weights_mean, (tf.size(weights_mean), 1))

        tf_fingerprint = tf.constant(self.fingerprint, dtype=tf.float32)
        tf_proj_matrix = tf.constant(self.proj_matrix, dtype=tf.float32)

        # compute the mean squared error
        regularized_loss = self.strength * tf.reduce_mean(tf.square(tf.subtract(
            tf_fingerprint, tf.matmul(tf_proj_matrix, weights_flat))))

        # apply a penalty to the loss function
        if self.apply_penalty:
            return regularized_loss
        return None

    def get_matrix(self):
        return self.proj_matrix, self.orthogonal_matrix

    def get_signature(self):
        return np.squeeze(self.signature), np.squeeze(self.fingerprint)

    def get_config(self):
        return {
            'strength': self.strength,
            'embed_dim': self.embed_dim,
            'apply_penalty': self.apply_penalty,
            'seed': self.seed,
            'num_zeros': self.num_zeros
        }



def extract_fingerprint(model, extracted_fingerprint_fname):
    for i, layer in enumerate(model.layers):
        try:
            if isinstance(layer.kernel_regularizer, FingerprintRegularizer):

                # retrieve the weights
                weights = layer.get_weights()[0]
                weights_mean = weights.mean(axis=3)
                weights_flat = weights_mean.reshape(weights_mean.size, 1)

                # retrieve the projection matrix and the orthogonal matrix
                proj_matrix, ortho_matrix = layer.kernel_regularizer.get_matrix()

                # extract the fingerprint
                extract_fingerprint = np.dot(proj_matrix, weights_flat)

                # extract the signature
                extract_coefficient = np.dot(extract_fingerprint.T, ortho_matrix)
                extract_signature = 0.5 * (extract_coefficient + 1)

                # print the fingerprint embedding information
                print(f'Extracted fingerprint saved to {extracted_fingerprint_fname}')
                print('Fingerprinted Layer Index = {}  Class = {}'.format(i, layer.__class__.__name__))

                # save the extracted fingerprint to an HDF5 file
                with h5py.File(extracted_fingerprint_fname, 'w') as hf:
                    hf.create_dataset('fingerprint', data=np.squeeze(extract_fingerprint))
                    hf.create_dataset('signature', data=np.squeeze(extract_signature))

        except AttributeError:
            continue  # Continue the loop if the layer has no regularizers
