# This script is designed to simulate a pruning attack on the fingerprinted layer
# while incorporating the process of fine-tuning.

# ------------------------------------- Import Libraries and Modules ---------------------------------------------------
import os
import sys
import json
import tempfile
import numpy as np
import tensorflow as tf
import keras.utils as tf_utils
from keras.optimizers import SGD
from keras.datasets import cifar10
import tensorflow_model_optimization as tfmot
tfmo = tfmot.sparsity.keras

# add the parent directory of the script's location to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import the embed_fingerprint module
from embed_fingerprint import FingerprintRegularizer

# ------------------------------------- Initialization and Configuration -----------------------------------------------

# set a seed
seed_value = 0
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)

# register the custom regularizer to load the pre-trained model
tf_utils.get_custom_objects()['FingerprintRegularizer'] = FingerprintRegularizer

# load the pre-trained model
model_path = os.path.join("result", "embedded_model.keras")
base_model = tf.keras.models.load_model(model_path)

# set configuration
config_fname = sys.argv[1]
fine_tune_settings = json.load(open(config_fname))

# -------------------------------------------- Load and Prepare Data ---------------------------------------------------

# read parameters
nb_epoch = fine_tune_settings['epoch']
batch_size = fine_tune_settings['batch_size']
sparsity_level = fine_tune_settings['sparsity_level']
validation_split = fine_tune_settings['validation_split']
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

# calculate the sizes for fine-tuning and evaluation of the model
total_samples = len(test_input)
train_size = int(0.8 * total_samples)  # 80% for training
validation_size = total_samples - train_size  # 20% for model evaluation

# split the training data
training_input = test_input[:train_size]
training_output = test_output[:train_size]

# split the validation data
validation_input = test_input[train_size:]
validation_output = test_output[train_size:]

# -------------------------------------------- Prune the Target Layer --------------------------------------------------

# assessing the performance of the base model before pruning
_, base_model_accuracy = base_model.evaluate(validation_input,
                                             validation_output,
                                             verbose=0)

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

fingerprinted_layer_name = find_fingerprinted_layer(base_model)

# prune the target layer
def apply_pruning_to_layer(layer):
    if layer.name == fingerprinted_layer_name:
        return tfmo.prune_low_magnitude(layer, pruning_schedule=tfmo.ConstantSparsity(target_sparsity=sparsity_level,
                                                                                      begin_step=0))
    return layer

# create the pruned model
pruned_model = tf.keras.models.clone_model(base_model, clone_function=apply_pruning_to_layer)

# ---------------------------------------- Fine-Tune the Pruned Model --------------------------------------------------

# compile the pruned model
sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
pruned_model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
print("Finished compiling")

# fine-tuning the pruned model
logdir = tempfile.mkdtemp()

callbacks = [tfmo.UpdatePruningStep(),
             tfmo.PruningSummaries(log_dir=logdir)]

print("\nFine-tuning the pruned model:")
pruned_model.fit(training_input,
                 training_output,
                 batch_size=batch_size,
                 epochs=nb_epoch,
                 validation_split=validation_split,
                 callbacks=callbacks)

# ----------------------------------- Check the Accuracy of the Pruning ------------------------------------------------

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

# ---------------------------- Validate Training Accuracy and Save Model -----------------------------------------------

# compare the accuracy of the pruned model to the base model
print("Assessing the performance of the model...")

stripped_pruned_model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
_, pruned_model_accuracy = stripped_pruned_model.evaluate(validation_input,
                                                          validation_output,
                                                          verbose=0)

print("Base model accuracy: {:.2f}%".format(base_model_accuracy * 100))
print("Pruned model accuracy: {:.2f}%".format(pruned_model_accuracy * 100))

# save model
pruned_model_fname = os.path.join('result', f'pruned_model_sparsity{sparsity_level}_epoch{nb_epoch}.keras')
stripped_pruned_model.save(pruned_model_fname)