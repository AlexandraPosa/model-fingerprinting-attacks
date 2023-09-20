# Fine-tune the model by applying the quantization aware training API
# to test the resilience of the embedded fingerprint.

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

# create a quantization aware model
quant_aware_model = quantize_model(pretrained_model)

# `quantize_model` requires a recompile.
sgd = SGD(lr=0.001,
          momentum=0.9,
          nesterov=True)
quant_aware_model.compile(loss="categorical_crossentropy",
                          optimizer=sgd,
                          metrics=["accuracy"])
quant_aware_model.summary()

# fine-tuning the quantized model
print("\nPerforming quantization aware training:")
quant_aware_model.fit(training_input,
                      training_output,
                      batch_size=500,
                      epochs=1,
                      validation_split=0.2)
'''
# check the data type of the quantized weights
for layer in quant_aware_model.layers:
    for weight in layer.weights:
        if weight.dtype != 'float32':
            print(f"Layer {layer.name} has weights with data type {weight.dtype}")
'''

# check quantized model accuracy
_, baseline_model_accuracy = pretrained_model.evaluate(validation_input,
                                                       validation_output,
                                                       verbose=0)

_, quant_aware_model_accuracy = quant_aware_model.evaluate(validation_input,
                                                           validation_output,
                                                           verbose=0)

print('Baseline model accuracy:', baseline_model_accuracy * 100)
print('Quantization aware model accuracy:', quant_aware_model_accuracy * 100)

# -------------------------------------- TF Lite Model Evaluation ------------------------------------------------------

# create quantized model for TFLite backend
converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter.convert()

# load the TFLite model
interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
interpreter.allocate_tensors()
'''
# check the data type of each tensor in the model
for tensor_details in interpreter.get_tensor_details():
    tensor_name = tensor_details["name"]
    tensor_dtype = tensor_details["dtype"]
    print(f"Tensor {tensor_name} has dtype {tensor_dtype}.")
'''
# TFLite model evaluation function
def evaluate_model(interpreter):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # run predictions on every image in the validation dataset
    prediction_digits = []
    for i, test_image in enumerate(validation_input):
        if i % 1000 == 0:
          print('Evaluated on {n} results so far.'.format(n=i))

        # pre-processing: add batch dimension and convert to float32 to match with the model's input data format
        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, test_image)

        # run inference
        interpreter.invoke()

        # post-processing: remove batch dimension and find the digit with the highest probability
        output = interpreter.tensor(output_index)
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)

    # compare prediction results with ground truth labels to calculate accuracy
    predicted_labels = np.array(prediction_digits)
    ground_truth_labels = np.argmax(validation_output, axis=1)  # convert one-hot encoded labels to single labels
    accuracy = (predicted_labels == ground_truth_labels).astype(np.float32).mean()

    return accuracy

test_accuracy = evaluate_model(interpreter)

print('Quantized TFLite model accuracy:', test_accuracy * 100)
print('Quantization aware model accuracy:', quant_aware_model_accuracy * 100)

# save TFLite model to file
tflite_model_path = os.path.join('result_09', 'quantized_model.tflite')
with open(tflite_model_path, 'wb') as f:
    f.write(quantized_tflite_model)




