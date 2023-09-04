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

# add the parent directory of the script's location to the path
result_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "result"))

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
fingerprint_filename = os.path.join(result_path, "fingerprint_data.h5")
proj_matrix, ortho_matrix, signature, fingerprint = load_from_hdf5(fingerprint_filename)

# load the original model
model_path = os.path.join(result_path, "embed_model.keras")
original_model = tf.keras.models.load_model(model_path)

# load the pruned model
pruned_model_path = os.path.join(result_path, "pruned_model_sparsity0.01_epoch1.keras")
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

# Compute percentiles
diff_fingerprint = orig_fingerprint - prun_fingerprint
percentiles = [25, 50, 75]
quantiles = np.percentile(diff_fingerprint, percentiles)

for p, q in zip(percentiles, quantiles):
    print(f"{p}th percentile: {q}")

# use element-wise comparison to find differences
absolute_differences = np.abs(orig_fingerprint - prun_fingerprint)

# subtract the 50th percentile from the 75th percentile
threshold = quantiles[2] - quantiles[1]

# count the number of values that exceed the threshold
count_exceeding_threshold = np.count_nonzero(absolute_differences > threshold)

# Print the count
print("Number of values exceeding the threshold:", count_exceeding_threshold)

# ----------------------------- Visualizing Differences in Fingerprint Distributions -----------------------------------

# Calculate the difference fingerprint
diff_fingerprint = orig_fingerprint - prun_fingerprint

# Create a figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Plot the histogram of the differences
ax1.hist(np.squeeze(diff_fingerprint), bins=50, color='blue')
ax1.set_xlabel('Difference')
ax1.set_ylabel('Frequency')
ax1.set_title('Difference Between Original and Pruned Fingerprint')

# Plot the histograms of the original and pruned fingerprints
ax2.hist(np.squeeze(orig_fingerprint), bins=50, alpha=0.5, label='Original Fingerprint', color='blue')
ax2.hist(np.squeeze(prun_fingerprint), bins=50, alpha=0.5, label='Pruned Fingerprint', color='red')
ax2.set_xlabel('Value')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Fingerprint Values')
ax2.legend(loc='upper right')

# Plot the histograms of the original and pruned signatures
ax3.hist(np.squeeze(orig_signature), bins=50, alpha=0.5, label='Original Signature', color='green')
ax3.hist(np.squeeze(prun_signature), bins=50, alpha=0.5, label='Pruned Signature', color='orange')
ax3.set_xlabel('Value')
ax3.set_ylabel('Frequency')
ax3.set_title('Distribution of Signature Values')
ax3.legend(loc='upper right')

# Show the figure with all three subplots
plt.tight_layout()
plt.show()

# save plots to file
plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images", "fingerprint_plots.png"))
fig.savefig(plot_path, dpi=300, bbox_inches='tight')











