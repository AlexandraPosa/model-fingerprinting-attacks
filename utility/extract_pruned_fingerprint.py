# This script compares and visualizes fingerprint data from an original and pruned neural network model
# to analyze the impact of pruning on the fingerprinted layers.

# ------------------------------------- Import Libraries and Modules ---------------------------------------------------

import os
import h5py
import numpy as np
import tensorflow as tf
import keras.utils as tf_utils
import matplotlib.pyplot as plt

# import the embed_fingerprint module
from embed_fingerprint import FingerprintRegularizer

# register the custom regularizer to load the models
tf_utils.get_custom_objects()['FingerprintRegularizer'] = FingerprintRegularizer

# set paths
result_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "result_09"))
plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images", "pruned_fingerprint.png"))

# ---------------------------------------- Save and Load Functions -----------------------------------------------------

def load_from_hdf5(filename):
    with h5py.File(filename, 'r') as hdf5_file:
        proj_matrix = hdf5_file['proj_matrix'][:]
        ortho_matrix = hdf5_file['ortho_matrix'][:]
        signature = hdf5_file['signature'][:]
        fingerprint = hdf5_file['fingerprint'][:]
    return proj_matrix, ortho_matrix, signature, fingerprint

# --------------------------------------------- Load Data --------------------------------------------------------------

# load the fingerprint embedding information
fingerprint_filename = os.path.join(result_path, "embedding_keys.h5")
proj_matrix, ortho_matrix, signature, fingerprint = load_from_hdf5(fingerprint_filename)

# load the original model
model_path = os.path.join(result_path, "embedded_model.keras")
original_model = tf.keras.models.load_model(model_path)

# load the pruned model
pruned_model_path = os.path.join(result_path, "pruned_model_sparsity0.4_epoch2.keras")
pruned_model = tf.keras.models.load_model(pruned_model_path)

pruned_model.summary()

# ---------------------------------- Function: Find Embedded Layer -----------------------------------------------------

# find the layer where the fingerprint is embedded
def find_fingerprinted_layer(model):
    embedded_layer_name = None
    for layer in model.layers:
        try:
            if isinstance(layer.kernel_regularizer, FingerprintRegularizer):
                embedded_layer_name = layer.name
                break  # Break the loop once the fingerprinted layer is found
        except AttributeError:
            continue
    return embedded_layer_name

# -------------------------------- Function: Extract Embedded Fingerprint ----------------------------------------------

# extract the fingerprint from the embedded layer
def extract_fingerprint(model, layer_name, proj_matrix, ortho_matrix):

    # retrieve the layer by name
    layer = model.get_layer(layer_name)

    # retrieve the weights
    weights = layer.get_weights()[0]
    weights_mean = weights.mean(axis=3)
    weights_flat = weights_mean.reshape(weights_mean.size, 1)

    # extract the fingerprint
    extract_fingerprint = np.dot(proj_matrix, weights_flat)

    # extract the signature
    extract_coefficient = np.dot(extract_fingerprint.T, ortho_matrix)
    extract_signature = 0.5 * (extract_coefficient + 1)

    return extract_signature, extract_fingerprint

# -------------------------- Analyzing the Differences Between the Extracted Fingerprints ------------------------------

# extract the fingerprint and signature from the original model
original_model_layer_name = find_fingerprinted_layer(original_model)

orig_signature, orig_fingerprint = extract_fingerprint(original_model,
                                                       original_model_layer_name,
                                                       proj_matrix, ortho_matrix)

# extract the fingerprint and signature from the pruned model
pruned_model_layer_name = find_fingerprinted_layer(pruned_model)

prun_signature, prun_fingerprint = extract_fingerprint(pruned_model,
                                                       pruned_model_layer_name,
                                                       proj_matrix, ortho_matrix)

# ----------------------------- Visualizing Differences in Fingerprint Distributions -----------------------------------

# plot the histograms of the original and pruned signatures
plt.hist(np.squeeze(orig_signature), bins=30, alpha=0.5, label='Non-Pruned Values', color='gray')
plt.hist(np.squeeze(prun_signature), bins=50, alpha=0.5, label='Pruned Values', color='orange')
plt.xlabel('Fingerprint Signature')
plt.ylabel('Frequency')
#plt.xticks(np.arange(-0.1, 1.2, 0.1))
plt.title(' ')
plt.legend(loc='upper center')

# show the figure
#plt.show()

# save the plot to file
plt.savefig(plot_path, dpi=300, bbox_inches='tight')











