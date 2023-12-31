# This script compares and visualizes fingerprint data from the original model and the quantized TFLIte model
# to analyze the impact of quantization on the accuracy of the extracted fingerprint.

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
result_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "result"))
plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images", "quantized_fingerprint.png"))

# ---------------------------------------- Save and Load Functions -----------------------------------------------------

def load_from_hdf5(filename, *dataset_names):
    data = {}
    with h5py.File(filename, 'r') as hdf5_file:
        for dataset_name in dataset_names:
            data[dataset_name] = hdf5_file[dataset_name][:]
    return tuple(data[dataset_name] for dataset_name in dataset_names)

# --------------------------------------------- Load Data --------------------------------------------------------------

# load the original model
original_model_path = os.path.join(result_path, "embedded_model.keras")
original_model = tf.keras.models.load_model(original_model_path)

# load the quantized TFLite model
tflite_model_path = os.path.join(result_path, "quantized_model.tflite")
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# load the fingerprint embedding information
embedding_keys_filename = os.path.join(result_path, "embedding_keys.h5")
embedding_dataset_names = ['proj_matrix', 'ortho_matrix', 'signature', 'fingerprint']
proj_matrix, ortho_matrix, embedded_signature, embedded_fingerprint = load_from_hdf5(embedding_keys_filename,
                                                                                     *embedding_dataset_names)

# load the extracted fingerprint from the original model
fingerprint_dataset_names = ['signature', 'fingerprint']
fingerprint_filename = os.path.join(result_path, "extracted_fingerprint.h5")
original_signature, original_fingerprint = load_from_hdf5(fingerprint_filename,  *fingerprint_dataset_names)

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
def extract_fingerprint(fingerprinted_layer_weights, proj_matrix, ortho_matrix):

    # flatten the weights
    weights_mean = fingerprinted_layer_weights.mean(axis=3)
    weights_flat = weights_mean.reshape(weights_mean.size, )

    # extract the fingerprint
    extract_fingerprint = np.dot(proj_matrix, weights_flat)

    # extract the signature
    extract_coefficient = np.dot(extract_fingerprint.T, ortho_matrix)
    extract_signature = 0.5 * (extract_coefficient + 1)

    return extract_signature, extract_fingerprint

# -------------------------- Analyzing the Differences Between the Extracted Fingerprints ------------------------------

# find the fingerprinted layer's name
fingerprinted_layer_name = find_fingerprinted_layer(original_model)

# get all tensor details
tensor_details = interpreter.get_tensor_details()

# find the tensor corresponding to the fingerprinted layer by name
fingerprinted_tensor_detail = None
for tensor_detail in tensor_details:
    # find the tensor associated with the fingerprinted layer in the original model and exclude tensors
    # resulting from adding bias terms to the outputs of a previous layer before applying activation functions
    if fingerprinted_layer_name in tensor_detail["name"] and 'BiasAdd' not in tensor_detail["name"]:
        fingerprinted_tensor_detail = tensor_detail
        break

# retrieve the weights of the fingerprinted layer
if fingerprinted_tensor_detail is not None:

    # access the quantized tensor data
    quantized_data = interpreter.get_tensor(fingerprinted_tensor_detail["index"])     # ndarray (64, 3, 3, 64)
    quantized_data = np.transpose(quantized_data, (1, 2, 3, 0))                       # ndarray (3, 3, 64, 64)

    # access the quantization parameters
    scales = fingerprinted_tensor_detail['quantization_parameters']['scales']
    zero_points = fingerprinted_tensor_detail['quantization_parameters']['zero_points']

    # dequantize the tensor data to obtain the original weights
    fingerprinted_layer_weights = (quantized_data - zero_points) * scales

else:
    print(f"Layer '{fingerprinted_layer_name}' not found in the model.")

# extract the fingerprint and signature from the
quantized_signature, quantized_fingerprint = extract_fingerprint(fingerprinted_layer_weights,
                                                                 proj_matrix,
                                                                 ortho_matrix)

# ----------------------------- Visualizing Differences in Fingerprint Distributions -----------------------------------

# plot the histograms of the original and pruned signatures
plt.hist(original_signature, bins=25, alpha=0.5, label='Non-Quantized Values', color='gray')
plt.hist(quantized_signature, bins=30, alpha=0.5, label='Quantized Values', color='orange')
plt.xlabel('Fingerprint Signature')
plt.ylabel('Frequency')
plt.title(' ')
plt.legend(loc='upper center')

# show the figure
plt.show()

# save the plot to file
#plt.savefig(plot_path, dpi=300, bbox_inches='tight')