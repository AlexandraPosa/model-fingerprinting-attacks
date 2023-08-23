# import libraries and modules
import os
import h5py
import numpy as np
import tensorflow as tf
#import tensorflow_model_optimization as tfmot

# import modules
import sklearn.metrics as metrics
import keras.utils as kutils
from embed_fingerprint import FingerprintRegularizer

# set paths
model_path = os.path.join("..", "result", "embed_model.keras")
hdf5_filepath = os.path.join("..", "result", "validation_data.h5")

# load the validation data
with h5py.File(hdf5_filepath, 'r') as hf:
    test_input = hf['input_data'][:]
    test_output = hf['output_labels'][:]

# register the custom regularizer
kutils.get_custom_objects()['FingerprintRegularizer'] = FingerprintRegularizer

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

'''
# apply the pruning policy to the specific layer
pruned_model = tfmot.sparsity.keras.update_pruning(
    pretrained_model,
    pruning_policy={fingerprinted_layer_name: 0.01}
)

# compile the pruned model
pruned_model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# make predictions using the pruned model
predictions_1 = pruned_model.predict(test_input)
predictions_2 = np.argmax(predictions_1, axis=1)
predictions_3 = kutils.np_utils.to_categorical(predictions_2)

accuracy = metrics.accuracy_score(test_output, predictions_3) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)'''

