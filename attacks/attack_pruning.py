# import libraries
import os
import h5py
import numpy as np
import tensorflow as tf
import sklearn.metrics as metrics
import keras.utils.np_utils as kutils
from embed_fingerprint import CustomRegularizer

# set paths
model_path = os.path.join("..", "result", "wrn_model.keras")
hdf5_filepath = os.path.join("..", "result", "validation_data.h5")

# load the validation data
with h5py.File(hdf5_filepath, 'r') as hf:
    test_input = hf['input_data'][:]
    test_output = hf['output_labels'][:]

# register the custom regularizer
tf.keras.utils.get_custom_objects()['CustomRegularizer'] = CustomRegularizer

# load the pre-trained model
pretrained_model = tf.keras.models.load_model(model_path)

# print the model architecture
pretrained_model.summary()

'''
# Define a custom pruning policy for the specific layer
custom_policy = tfmot.sparsity.keras.PolynomialDecayPruningPolicy(
    target_sparsity=0.5,
    initial_sparsity=0.2,
    begin_step=0,
    end_step=1000
)

# Apply the pruning policy to the specific layer
pruned_model = tfmot.sparsity.keras.update_pruning(
    pretrained_model,
    pruning_policy={
        'conv2d_layer_name': custom_policy
    }
)

# Compile the pruned model
pruned_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])'''

# Make predictions using the pruned model
predictions_1 = pretrained_model.predict(test_input)
predictions_2 = np.argmax(predictions_1, axis=1)
predictions_3 = kutils.to_categorical(predictions_2)

accuracy = metrics.accuracy_score(test_output, predictions_3) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)

