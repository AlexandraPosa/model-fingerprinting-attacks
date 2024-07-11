# This script simulates an attack by pruning less active filters that contribute minimally to the outcome,
# using a global threshold derived from the lowest percentage across all filters.

# ---------------------------------------- Import Libraries and Modules ------------------------------------------------
import os
import sys
import numpy as np
import tensorflow as tf
import keras.utils as tf_utils
from keras.datasets import cifar10
from keras.models import Model

# add the parent directory of the script's location to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import the embed_fingerprint module
from embed_fingerprint import FingerprintRegularizer

# ------------------------------------------ Load and Prepare Data -----------------------------------------------------

# set a seed for reproducibility
seed_value = 33
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)

# register the custom regularizer to load the trained model
tf_utils.get_custom_objects()['FingerprintRegularizer'] = FingerprintRegularizer

# load the trained model
model_path = os.path.join("result", "embedded_model.keras")
base_model = tf.keras.models.load_model(model_path)
pruned_model = tf.keras.models.load_model(model_path)

# load the CIFAR-10 dataset
(train_input, train_output), (test_input, test_output) = cifar10.load_data()

# normalize the data
train_input = train_input.astype('float32') / 255.0
test_input = test_input.astype('float32') / 255.0

# convert labels to categorical format
train_output = tf_utils.to_categorical(train_output)
test_output = tf_utils.to_categorical(test_output)

# reduce the size of the training data by half
half_size = len(train_input) // 50
train_input = train_input[:half_size]

# -------------------------------------- Configuration and Initialization ----------------------------------------------

# determine the batch size and number of samples
batch_size = 64
num_samples = len(train_input)

# define the threshold for the lowest percentage of the activations
sparsity_level = 10

# specify the names of the activation layers and the respective convolutional layers
activation_layer_names = ['activation', 'activation_1', 'activation_2', 'activation_3', 'activation_4', 'activation_5',
                          'activation_6']
convolutional_layer_names = ['conv2d', 'conv2d_2', 'conv2d_3', 'conv2d_5', 'conv2d_6', 'conv2d_8', 'conv2d_9']

# set the file path for saving the pruned model
pruned_model_fname = os.path.join('result', f'global_activation_pruned_model_sparsity{sparsity_level}.keras')

# ----------------------------------------- Activation Based Pruning ---------------------------------------------------

# get the output tensors of the activation layers
activation_layers = [pruned_model.get_layer(name).output for name in activation_layer_names]
activation_model = Model(inputs=pruned_model.input, outputs=activation_layers)

# initialize a list to store mean activations per filter across batches
mean_activations_per_layer = [[] for _ in activation_layer_names]

# iterate over the training data in batches
for i in range(0, num_samples, batch_size):
    batch_input = train_input[i:i + batch_size]
    batch_activations = activation_model.predict(batch_input)

    # collect mean activations for each layer in the batch
    for layer_idx, activation in enumerate(batch_activations):
        mean_activations = np.mean(np.abs(activation), axis=(0, 1, 2))
        mean_activations_per_layer[layer_idx].append(mean_activations)

# aggregate mean activations across all batches
aggregate_activations = [np.mean(layer_activations, axis=0) for layer_activations in mean_activations_per_layer]

# determine the threshold for the lowest percentage of the activations across all layers
flatten_aggregate_activations = np.concatenate(aggregate_activations)
global_pruning_threshold = np.percentile(flatten_aggregate_activations, sparsity_level)

def prune_filters(convolutional_layer, mean_activations):
    # identify filters to prune
    filters_to_prune = mean_activations < global_pruning_threshold

    # get the number of filters in the convolutional layer
    total_filters = mean_activations.shape[0]

    # calculate the number of pruned filters
    pruned_filters = np.sum(filters_to_prune)

    # calculate the percentage of pruned filters
    percentage_pruned = (pruned_filters / total_filters) * 100

    # print the percentage of pruned filters
    print(f"Percentage of pruned filters in layer {convolutional_layer.name}: {percentage_pruned:.2f}%")

    # get current weights and biases of the convolutional layer
    weights, biases = convolutional_layer.get_weights()

    # zero out the weights of the filters to be pruned
    weights[:, :, :, filters_to_prune] = 0

    # set the new weights back to the convolutional layer
    convolutional_layer.set_weights([weights, biases])

# prune corresponding convolutional layers
for i, conv_layer_name in enumerate(convolutional_layer_names):
    conv_layer = pruned_model.get_layer(conv_layer_name)
    prune_filters(conv_layer, aggregate_activations[i])

# -------------------------------------- Evaluating Model Performance --------------------------------------------------

# compare the accuracy of the pruned model to the base model
print("\nAssessing the performance of the model:")

_, base_model_accuracy = base_model.evaluate(test_input, test_output, verbose=0)
_, pruned_model_accuracy = pruned_model.evaluate(test_input, test_output, verbose=0)

print("Base model accuracy: {:.2f}%".format(base_model_accuracy * 100))
print("Pruned model accuracy: {:.2f}%".format(pruned_model_accuracy * 100))

# save the pruned model
pruned_model.save(pruned_model_fname)