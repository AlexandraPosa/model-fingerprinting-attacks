# ------------------------------------- Import Libraries and Modules ---------------------------------------------------
import os
import sys
import numpy as np
import tensorflow as tf
import keras.utils as tf_utils
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

# load the trained model
model_path = os.path.join("result", "embedded_model.keras")
base_model = tf.keras.models.load_model(model_path)

# load data
(train_input, train_output), (test_input, test_output) = cifar10.load_data()
train_input = train_input.astype('float32') / 255.0
test_input = test_input.astype('float32') / 255.0
train_output = tf_utils.to_categorical(train_output)
test_output = tf_utils.to_categorical(test_output)

# ---------------------------------------- Quantization using Tensorflow Lite ------------------------------------------

# Define a function to generate a representative dataset
def representative_dataset_gen():
    num_calibration_steps = 50
    batch_size = 64
    for i in range(num_calibration_steps):
        # Select a random batch of images from the training set
        batch_images = train_input[i * batch_size:(i + 1) * batch_size]
        yield [batch_images]

# convert the model to TensorFlow Lite format with float16 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()

# initialize the TensorFlow Lite interpreter
interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)
interpreter.allocate_tensors()

# check the data type of each tensor in the model
for tensor_details in interpreter.get_tensor_details():
    tensor_name = tensor_details["name"]
    tensor_dtype = tensor_details["dtype"]
    print(f"Tensor {tensor_name} has dtype {tensor_dtype}.")

# -------------------------------------- TF Lite Model Evaluation ------------------------------------------------------

print("\nAssessing the performance of the model...")

# TFLite model evaluation function
def evaluate_model(interpreter):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # run predictions on every image in the validation dataset
    prediction_digits = []
    for i, test_image in enumerate(test_input):

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
    ground_truth_labels = np.argmax(test_output, axis=1)  # convert one-hot encoded labels to single labels
    accuracy = (predicted_labels == ground_truth_labels).astype(np.float32).mean()

    return accuracy

test_accuracy = evaluate_model(interpreter)

# check base model accuracy
_, baseline_model_accuracy = base_model.evaluate(test_input,
                                                 test_output,
                                                 verbose=0)

print('Baseline model accuracy:', baseline_model_accuracy * 100)
print('Quantized TFLite model accuracy:', test_accuracy * 100)

# save TFLite model to file
tflite_model_path = os.path.join('result', 'quantized_model_int8.tflite')
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_quant_model)

