

# ------------------------------------- Import Libraries and Modules ---------------------------------------------------
import os
import sys
import json
import numpy as np
import tensorflow as tf
import keras.utils as tf_utils
from keras.optimizers import SGD
from keras.datasets import cifar10
import tensorflow_model_optimization as tfmot
quantize_model = tfmot.quantization.keras.quantize_model

# add the parent directory to the module search path
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
model_path = os.path.join("result_09", "embedded_model.keras")
pretrained_model = tf.keras.models.load_model(model_path)

# set configuration
config_fname = sys.argv[1]
fine_tune_settings = json.load(open(config_fname))

# -------------------------------------------- Load and Prepare Data ---------------------------------------------------

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

# ---------------------------------------- Quantization Aware Training -------------------------------------------------

# quantization aware
quantized_model = quantize_model(pretrained_model)

# `quantize_model` requires a recompile.
sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
quantized_model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
quantized_model.summary()

# fine-tuning the quantized model
print("\nPerforming quantization aware training:")
quantized_model.fit(training_input,
                    training_output,
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    validation_split=0.2)

# check if weights are int8 instead of float32
for layer in quantized_model.layers:
    for weight in layer.weights:
        if weight.dtype != 'float32':
            print(f"Layer {layer.name} has weights with data type {weight.dtype}")

# ----------------------------------------- Quantized Model Evaluation -------------------------------------------------

_, baseline_model_accuracy = pretrained_model.evaluate(validation_input,
                                                       validation_output,
                                                       verbose=0)

_, quantized_model_accuracy = quantized_model.evaluate(validation_input,
                                                       validation_output,
                                                       verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy)
print('Quantized test accuracy:', quantized_model_accuracy)

