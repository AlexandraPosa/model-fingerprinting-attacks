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
sparsity_level = 0.7

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

# -------------------------------------------- Prune the Model ---------------------------------------------------------

def prune_model(model, prune_percentage):
    # Collect all weights from the model
    all_weights = []
    layer_shapes = []
    layer_sizes = []
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            weights, biases = layer.get_weights()
            all_weights.append(weights.ravel())
            layer_shapes.append(weights.shape)
            layer_sizes.append(weights.size)

    # Concatenate all weights into a single array
    all_weights = np.concatenate(all_weights)

    # Calculate the total number of weights and the number of weights to prune
    total_weights = all_weights.size
    num_weights_to_prune = int(prune_percentage * total_weights)

    # Get the indices of the smallest weights to prune
    sorted_indices = np.argsort(np.abs(all_weights))
    prune_indices = sorted_indices[:num_weights_to_prune]

    # Create a mask to zero out the smallest weights
    prune_mask = np.ones_like(all_weights, dtype=bool)
    prune_mask[prune_indices] = False

    # Apply the mask to prune weights in the model
    start = 0
    pruned_weights = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            weights, biases = layer.get_weights()
            shape = weights.shape
            size = weights.size
            flat_weights = weights.ravel()

            # Prune the weights
            flat_weights[~prune_mask[start:start + size]] = 0
            weights = flat_weights.reshape(shape)
            layer.set_weights([weights, biases])

            # Save pruned weights for overall sparsity calculation
            pruned_weights.append(flat_weights)

            # Calculate and print the sparsity level for this layer
            num_pruned_weights = np.count_nonzero(flat_weights == 0)
            sparsity = num_pruned_weights / size
            print(f"Layer '{layer.name}': Sparsity: {sparsity:.2%}")

            start += size

    # Concatenate all pruned weights into a single array
    pruned_weights = np.concatenate(pruned_weights)

    # Calculate overall sparsity level
    overall_sparsity = np.count_nonzero(pruned_weights == 0) / total_weights
    print(f"\nOverall model sparsity: {overall_sparsity:.2%}")

prune_model(pruned_model, sparsity_level)

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
pruned_model_fname = os.path.join('result', f'pruned_model_sparsity{sparsity_level}.keras')
#pruned_model.save(pruned_model_fname)
