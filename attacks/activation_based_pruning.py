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

# -------------------------------------------- Load and Prepare Data ---------------------------------------------------

# set a seed
seed_value = 0
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)

# determine the threshold for the lowest percentage of the activations
sparsity_level = 20

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

# ------------------------------------------ Activation Based Pruning --------------------------------------------------

layer_names = ['activation', 'activation_1', 'activation_2', 'activation_3', 'activation_4', 'activation_5', 'activation_6']
activation_layers = [pruned_model.get_layer(name).output for name in layer_names]
activation_model = Model(inputs=pruned_model.input, outputs=activation_layers)
activations = activation_model.predict(test_input)

def prune_filters(conv_layer, activation):
    # Calculate mean activation per filter
    mean_activations = np.mean(activation, axis=(0, 1, 2))

    # Determine the threshold for the lowest 20% of the activations
    pruning_threshold = np.percentile(mean_activations, sparsity_level)

    # Identify filters to prune
    filters_to_prune = mean_activations < pruning_threshold

    # Get current weights and biases of the convolutional layer
    weights, biases = conv_layer.get_weights()

    # Zero out the weights of the filters to be pruned
    weights[:, :, :, filters_to_prune] = 0

    # Set the new weights back to the convolutional layer
    conv_layer.set_weights([weights, biases])

# Prune corresponding conv layers
conv_layers = ['conv2d', 'conv2d_2', 'conv2d_3', 'conv2d_5', 'conv2d_6', 'conv2d_8', 'conv2d_9']
for i, conv_layer_name in enumerate(conv_layers):
    conv_layer = pruned_model.get_layer(conv_layer_name)
    prune_filters(conv_layer, activations[i])

# check performance of the pruned model:
_, pruned_model_accuracy = pruned_model.evaluate(test_input,
                                                 test_output,
                                                 verbose=0)

print("Pruned model accuracy: {:.2f}%".format(pruned_model_accuracy * 100))