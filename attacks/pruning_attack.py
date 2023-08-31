# This script is designed to simulate a pruning attack on the fingerprinted layer
# while incorporating the process of fine-tuning.

# ------------------------------------- Import Libraries and Modules ---------------------------------------------------
import os
import sys
import json
import h5py
import tempfile
import numpy as np
import tensorflow as tf
import keras.utils as tf_utils
from keras.optimizers import SGD
import sklearn.metrics as metrics
from keras.datasets import cifar10
import tensorflow_model_optimization as tfmot
tfmo = tfmot.sparsity.keras

# add the parent directory of the script's location to the path to import the embed_fingerprint module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from embed_fingerprint import FingerprintRegularizer

'''
# load the validation data
hdf5_filepath = os.path.join("..", "result", "validation_data.h5")
with h5py.File(hdf5_filepath, 'r') as hf:
    test_input = hf['input_data'][:]
    test_output = hf['output_labels'][:]
'''
# -------------------------------------------- Load and Prepare Data ---------------------------------------------------
# set a seed
seed_value = 0
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)

# register the custom regularizer to load the pre-trained model
tf_utils.get_custom_objects()['FingerprintRegularizer'] = FingerprintRegularizer

# set configuration
config_fname = sys.argv[1]
fine_tune_settings = json.load(open(config_fname))

# read parameters
sparsity_level = fine_tune_settings['sparsity_level']
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
#pretrained_model.summary()

# -------------------------------------------- Prune the Target Layer --------------------------------------------------

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

# prune the target layer
def apply_pruning_to_layer(layer):
    if layer.name == fingerprinted_layer_name:
        return tfmo.prune_low_magnitude(layer, pruning_schedule=tfmo.ConstantSparsity(target_sparsity=sparsity_level,
                                                                                      begin_step=0))
    return layer

# create the pruned model
pruned_model = tf.keras.models.clone_model(pretrained_model, clone_function=apply_pruning_to_layer)

# print the model architecture
#pruned_model.summary()

# ---------------------------------------- Fine-Tune the Pruned Model --------------------------------------------------

# compile the pruned model
sgd = SGD(lr=0.0005, momentum=0.9, nesterov=True)
pruned_model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
print("Finished compiling")

# fine-tuning the pruned model
logdir = tempfile.mkdtemp()

callbacks = [tfmo.UpdatePruningStep(),
             tfmo.PruningSummaries(log_dir=logdir)]

print("\nFine-tuning the pruned model:")
pruned_model.fit(train_input,
                 train_output,
                 batch_size=batch_size,
                 epochs=nb_epoch,
                 validation_split=0.15,
                 callbacks=callbacks)

# ---------------------------------------- Evaluate the Pruned Model --------------------------------------------------

# check that the layer was correctly pruned:
print(f"\nAssessing the sparsity level within the fingerprinted layer: {fingerprinted_layer_name}")

def print_model_weights_sparsity(model, fingerprinted_layer_name):
    for layer in model.layers:
        if layer.name == fingerprinted_layer_name:
            if isinstance(layer, tf.keras.layers.Wrapper):
                weights = layer.trainable_weights
            else:
                weights = layer.weights
            for weight in weights:
                # ignore auxiliary quantization weights
                if "quantize_layer" in weight.name:
                    continue
                weight_size = weight.numpy().size
                zero_num = np.count_nonzero(weight == 0)
                print(
                    f"{weight.name}: {zero_num / weight_size:.2%} sparsity ",
                    f"({zero_num}/{weight_size})"
                )

# strip the pruning wrapper before checking
stripped_pruned_model = tfmo.strip_pruning(pruned_model)
print_model_weights_sparsity(stripped_pruned_model, fingerprinted_layer_name)

# print the pruned model architecture
#stripped_pruned_model.summary()

# make predictions using the pruned model
print("\nPerforming an evaluation on the pruned model:")
predictions_1 = stripped_pruned_model.predict(test_input)
predictions_2 = np.argmax(predictions_1, axis=1)
predictions_3 = tf_utils.np_utils.to_categorical(predictions_2, num_classes=10)

# compute the accuracy of the pruned model
accuracy = metrics.accuracy_score(test_output, predictions_3) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)