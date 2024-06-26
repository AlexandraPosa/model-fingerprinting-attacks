# This script simulates an attack by pruning less active filters that contribute minimally to the outcome.

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
seed_value = 0
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)

# register the custom regularizer to load the trained model
tf_utils.get_custom_objects()['FingerprintRegularizer'] = FingerprintRegularizer

# load the trained model
model_path = os.path.join("result", "embedded_model.keras")
base_model = tf.keras.models.load_model(model_path)
pruned_model = tf.keras.models.load_model(model_path)

# load data
(train_input, train_output), (test_input, test_output) = cifar10.load_data()
train_input = train_input.astype('float32') / 255.0
test_input = test_input.astype('float32') / 255.0
train_output = tf_utils.to_categorical(train_output)
test_output = tf_utils.to_categorical(test_output)

# -------------------------------------- Configuration and Initialization ----------------------------------------------

# define the threshold for the lowest percentage of the activations
sparsity_level = 10

# define the names of the activation layers and the respective convolutional layers
activation_layer_names = ['activation', 'activation_1', 'activation_2', 'activation_3', 'activation_4', 'activation_5',
                          'activation_6']
convolutional_layer_names = ['conv2d', 'conv2d_2', 'conv2d_3', 'conv2d_5', 'conv2d_6', 'conv2d_8', 'conv2d_9']

# ----------------------------------------- Activation Based Pruning ---------------------------------------------------

# get the output tensors of the activation layers
activation_layers = [pruned_model.get_layer(name).output for name in activation_layer_names]
activation_model = Model(inputs=pruned_model.input, outputs=activation_layers)

# predict activations using a subset of the training data
activations = activation_model.predict(train_input[:len(train_input) // 10])

def prune_filters(convolutional_layer, activation_layer):
    # calculate mean activation per filter
    mean_activations = np.mean(activation_layer, axis=(0, 1, 2))

    # determine the threshold for the lowest percentage of the activations
    pruning_threshold = np.percentile(mean_activations, sparsity_level)

    # identify filters to prune
    filters_to_prune = mean_activations < pruning_threshold

    # get current weights and biases of the convolutional layer
    weights, biases = convolutional_layer.get_weights()

    # zero out the weights of the filters to be pruned
    weights[:, :, :, filters_to_prune] = 0

    # set the new weights back to the convolutional layer
    convolutional_layer.set_weights([weights, biases])

# prune corresponding convolutional layers
for i, conv_layer_name in enumerate(convolutional_layer_names):
    conv_layer = pruned_model.get_layer(conv_layer_name)
    prune_filters(conv_layer, activations[i])

# -------------------------------------- Evaluating Model Performance --------------------------------------------------

# compare the accuracy of the pruned model to the base model
print("\nAssessing the performance of the model:")

_, base_model_accuracy = base_model.evaluate(test_input, test_output, verbose=0)
_, pruned_model_accuracy = pruned_model.evaluate(test_input, test_output, verbose=0)

print("Base model accuracy: {:.2f}%".format(base_model_accuracy * 100))
print("Pruned model accuracy: {:.2f}%".format(pruned_model_accuracy * 100))
