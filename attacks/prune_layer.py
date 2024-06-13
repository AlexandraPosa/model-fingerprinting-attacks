import os
import sys
import numpy as np
import tensorflow as tf
import keras.utils as tf_utils
from keras.datasets import cifar10

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

# set sparsity level for the target layer
sparsity_level = 0.4

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

fingerprinted_layer_name = find_fingerprinted_layer(base_model)

print(f"\nPruning the fingerprinted layer.")

# manually set a percentage of weights to zero for the fingerprinted layer
def prune_layer(model, fingerprinted_layer_name, sparsity_level):
    for layer in model.layers:
        if layer.name == fingerprinted_layer_name:
            # check if the target layer is a convolutional layer
            if isinstance(layer, tf.keras.layers.Conv2D):
                weights, biases = layer.get_weights()

                # calculate the number of weights to set to zero based on the percentage
                num_weights_to_zero = int(sparsity_level * np.prod(weights.shape))

                # flatten the weights
                flat_weights = weights.ravel()

                # get the indices of the smallest weights
                sorted_indices = np.argsort(np.abs(flat_weights))  # Sort by absolute value
                zero_indices = sorted_indices[:num_weights_to_zero]

                # set the smallest weights to zero
                flat_weights[zero_indices] = 0

                # reshape back to the original shape
                weights = flat_weights.reshape(weights.shape)

                # set the modified weights back to the layer
                layer.set_weights([weights, biases])

prune_layer(pruned_model, fingerprinted_layer_name, sparsity_level)

# ----------------------------------- Check the Accuracy of the Pruning ------------------------------------------------

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

print_model_weights_sparsity(pruned_model, fingerprinted_layer_name)

# ---------------------------- Validate Training Accuracy and Save Model -----------------------------------------------

# compare the accuracy of the pruned model to the base model
print("\nAssessing the performance of the model...")
'''
_, base_model_accuracy = base_model.evaluate(test_input,
                                             test_output,
                                             verbose=0)'''

_, pruned_model_accuracy = pruned_model.evaluate(test_input,
                                                 test_output,
                                                 verbose=0)

#print("Base model accuracy: {:.2f}%".format(base_model_accuracy * 100))
print("Pruned model accuracy: {:.2f}%".format(pruned_model_accuracy * 100))

# save model
pruned_model_fname = os.path.join('result', f'pruned_layer_sparsity{sparsity_level}.keras')
#pruned_model.save(pruned_model_fname)