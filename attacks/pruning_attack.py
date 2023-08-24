# import libraries and modules
import os
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# import modules
import keras.utils as tf_utils
from keras.optimizers import SGD
import sklearn.metrics as metrics
from embed_fingerprint import FingerprintRegularizer

# set paths
model_path = os.path.join("..", "result", "embed_model.keras")
hdf5_filepath = os.path.join("..", "result", "validation_data.h5")

# load the validation data
with h5py.File(hdf5_filepath, 'r') as hf:
    test_input = hf['input_data'][:]
    test_output = hf['output_labels'][:]

# register the custom regularizer
tf_utils.get_custom_objects()['FingerprintRegularizer'] = FingerprintRegularizer

# load the pre-trained model
pretrained_model = tf.keras.models.load_model(model_path)

# print the model architecture
pretrained_model.summary()

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

fingerprinted_layer_name = find_fingerprinted_layer(pretrained_model)
print("Fingerprinted Layer:", fingerprinted_layer_name)

# Helper function uses `prune_low_magnitude` to make only the
# Dense layers train with pruning.
def apply_pruning_to_layer(layer):
  if layer.name == fingerprinted_layer_name:
    return tfmot.sparsity.keras.prune_low_magnitude(layer)
  return layer

# Use `tf.keras.models.clone_model` to apply `apply_pruning_to_dense`
# to the layers of the model.
pruned_model = tf.keras.models.clone_model(
    pretrained_model,
    clone_function=apply_pruning_to_layer
)

# compile the pruned model
#sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
#pruned_model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# print the model architecture
pruned_model.summary()

pruned_layer = pruned_model.get_layer("prune_low_magnitude_" + fingerprinted_layer_name)
pruned_weights = pruned_layer.get_weights()

# Count the number of non-zero elements in the weights
non_zero_weights = np.count_nonzero(np.concatenate([w.flatten() for w in pruned_weights]))

# Print the total number of parameters and the number of non-zero parameters
total_params = pruned_layer.count_params()
print("Fingerprinted layer:")
print("Total parameters:", total_params)
print("Number of non-zero weights:", non_zero_weights)

# make predictions using the pruned model
predictions_1 = pruned_model.predict(test_input)
predictions_2 = np.argmax(predictions_1, axis=1)
predictions_3 = tf_utils.np_utils.to_categorical(predictions_2, num_classes=10)

accuracy = metrics.accuracy_score(test_output, predictions_3) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)

