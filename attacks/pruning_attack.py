# import libraries and modules
import os
import sys
import json
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# import modules
import tempfile
import keras.utils as tf_utils
from keras.optimizers import SGD
import sklearn.metrics as metrics
from keras.datasets import cifar10
from embed_fingerprint import FingerprintRegularizer
tfmo = tfmot.sparsity.keras

# register the custom regularizer
tf_utils.get_custom_objects()['FingerprintRegularizer'] = FingerprintRegularizer

'''
# load the validation data
hdf5_filepath = os.path.join("..", "result", "validation_data.h5")
with h5py.File(hdf5_filepath, 'r') as hf:
    test_input = hf['input_data'][:]
    test_output = hf['output_labels'][:]
'''

# set configuration
config_fname = sys.argv[1]
fine_tune_settings = json.load(open(config_fname))

# read parameters
batch_size = fine_tune_settings['batch_size']
nb_epoch = fine_tune_settings['epoch']
dataset = cifar10 if fine_tune_settings['dataset'] == 'cifar10' else None
if dataset is None:
    print('not supported dataset "{}"'.format(fine_tune_settings['dataset']))
    exit(1)

# load data
(train_input, train_output), (test_input, test_output) = dataset.load_data()
train_input = train_input.astype('float32') / 255.0
test_input = test_input.astype('float32') / 255.0
train_output = tf_utils.to_categorical(train_output)
test_output = tf_utils.to_categorical(test_output)

# load the pre-trained model
model_path = os.path.join("result", "embed_model.keras")
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

# prune the target layer
def apply_pruning_to_layer(layer):
    if layer.name == fingerprinted_layer_name:
        return tfmo.prune_low_magnitude(layer, pruning_schedule=tfmo.ConstantSparsity(target_sparsity=0.1,
                                                                                      begin_step=0))
    return layer

# create the pruned model
pruned_model = tf.keras.models.clone_model(pretrained_model, clone_function=apply_pruning_to_layer)

# print the model architecture
pruned_model.summary()

# compile the pruned model
sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
pruned_model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
print("Finished compiling")

# fine-tuning the pruned model
logdir = tempfile.mkdtemp()

callbacks = [tfmo.UpdatePruningStep(),
             tfmo.PruningSummaries(log_dir=logdir)
             ]

pruned_model.fit(train_input,
                 train_output,
                 batch_size=batch_size,
                 epochs=nb_epoch,
                 validation_split=0.1,
                 callbacks=callbacks
                 )

'''
# count the number of pruned (0) and active (1) weights
pruned_layer = pruned_model.get_layer("prune_low_magnitude_" + fingerprinted_layer_name)
pruning_mask = pruned_layer.get_mask() ???
flattened_mask = np.concatenate([mask.numpy().flatten() for mask in pruning_mask])
num_pruned_weights = np.count_nonzero(pruning_mask == 0)
num_active_weights = np.count_nonzero(pruning_mask == 1)

# print the pruning results
print("Fingerprinted layer:")
print("Number of pruned weights:", num_pruned_weights)
print("Number of active weights:", num_active_weights)
'''

# make predictions using the pruned model
predictions_1 = pruned_model.predict(test_input)
predictions_2 = np.argmax(predictions_1, axis=1)
predictions_3 = tf_utils.np_utils.to_categorical(predictions_2, num_classes=10)

accuracy = metrics.accuracy_score(test_output, predictions_3) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)
