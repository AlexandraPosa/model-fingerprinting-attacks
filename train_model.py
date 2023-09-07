# This script performs the training of a wide residual network using the CIFAR-10 dataset
# while providing an optional feature to embed a fingerprint into a specified target layer.

# ------------------------------------- Import Libraries and Modules ---------------------------------------------------

import numpy as np
import random
import sys
import json
import os
import h5py
import tensorflow as tf
import sklearn.metrics as metrics
import wide_residual_network as wrn
import keras.utils.np_utils as kutils
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from embed_fingerprint import FingerprintRegularizer
from embed_fingerprint import show_encoded_signature

# ------------------------------------- Initialization and Configuration -----------------------------------------------

# set seed
seed_value = 0
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)

# set configuration
settings_json_fname = sys.argv[1]
train_settings = json.load(open(settings_json_fname))

# set paths
result_path = './result'
os.makedirs(result_path) if not os.path.isdir(result_path) else None
model_checkpoint_fname = os.path.join(result_path, 'model_checkpoint.h5')
fingerprint_fname = os.path.join(result_path, 'fingerprint_data.h5')
model_filepath = os.path.join(result_path, 'embed_model.keras') \
    if train_settings['embed_flag'] == True \
    else os.path.join(result_path, 'non_embed_model.keras')

# ---------------------------------------- Save and Load Functions -----------------------------------------------------

# Save the keys used for the fingerprint embedding to an HDF5 file
def save_to_hdf5(proj_matrix, ortho_matrix, fingerprint, signature, fingerprint_fname):
    with h5py.File(fingerprint_fname, 'w') as hdf5_file:
        hdf5_file.create_dataset('proj_matrix', data=proj_matrix)
        hdf5_file.create_dataset('ortho_matrix', data=ortho_matrix)
        hdf5_file.create_dataset('signature', data=signature)
        hdf5_file.create_dataset('fingerprint', data=fingerprint)

# --------------------------------------------- Load Dataset -----------------------------------------------------------

# load dataset
if train_settings['dataset'] == 'cifar10':
    dataset = cifar10
    nb_classes = 10
else:
    print('not supported dataset "{}"'.format(train_settings['dataset']))
    exit(1)

# ------------------------------------------- Read Parameters ----------------------------------------------------------

# read parameters
batch_size = train_settings['batch_size']
nb_epoch = train_settings['epoch']
scale = train_settings['scale']
embed_dim = train_settings['embed_dim']
embed_flag = train_settings['embed_flag']
N = train_settings['N']
k = train_settings['k']
target_blk_id = train_settings['target_blk_id']

# set learning rate decay ratio to 0.2
lr_schedule = [60, 120, 160]

def schedule(epoch_idx):
    if (epoch_idx + 1) < lr_schedule[0]:
        return 0.05
    elif (epoch_idx + 1) < lr_schedule[1]:
        return 0.01
    elif (epoch_idx + 1) < lr_schedule[2]:
        return 0.002
    return 0.0004

# ------------------------------- Data Preprocessing and Augmentation --------------------------------------------------

# load data
(train_input, train_output), (test_input, test_output) = dataset.load_data()
train_input = train_input.astype('float32') / 255.0
test_input = test_input.astype('float32') / 255.0
train_output = kutils.to_categorical(train_output)
test_output = kutils.to_categorical(test_output)

# split the data for training and validation
total_samples = len(train_input)
train_size = int(0.8 * total_samples)  # 80% for training
validation_size = total_samples - train_size  # 20% for validation

# Split the training data
training_input = train_input[:train_size]
training_output = train_output[:train_size]

# Split the validation data
validation_input = train_input[train_size:]
validation_output = train_output[train_size:]

# fitting data for learning
generator = ImageDataGenerator(rotation_range=10,
                               width_shift_range=5./32,
                               height_shift_range=5./32,
                               horizontal_flip=True)

generator.fit(training_input, augment=True)

# --------------------------------------------- Create Model -----------------------------------------------------------

# create model
init_shape = (3, 32, 32) if K.image_data_format() == "channels_first" else (32, 32, 3)

fingerprint_embedding = FingerprintRegularizer(strength=scale, embed_dim=embed_dim, seed=seed_value,
                                               apply_penalty=embed_flag)

model = wrn.create_wide_residual_network(init_shape, nb_classes=nb_classes, N=N, k=k, dropout=0.00,
                                         custom_regularizer=fingerprint_embedding, target_blk_num=target_blk_id)
model.summary()

# ------------------------------------------ Training Process ----------------------------------------------------------

# model compilation
model.compile(loss="categorical_crossentropy",
              optimizer=SGD(lr=0.05, momentum=0.9, nesterov=True),
              metrics=["accuracy"])

print("Finished compiling")

# model training
callbacks = [ModelCheckpoint(model_checkpoint_fname,
                             monitor="val_accuracy",
                             save_best_only=True),
             LearningRateScheduler(schedule=schedule)]

hist = \
model.fit(generator.flow(training_input, training_output, batch_size=batch_size),
          steps_per_epoch=np.ceil(len(training_input)/batch_size),
          epochs=nb_epoch,
          callbacks=callbacks,
          validation_data=(validation_input, validation_output),
          validation_steps=np.ceil(len(validation_input)/batch_size))

# -------------------------------- Print and Save the Fingerprint Information ------------------------------------------

# print the keys used for the embedding
proj_matrix, ortho_matrix = fingerprint_embedding.get_matrix()
print('\nProjection matrix.\n{}\n \nOrthogonal matrix:\n{}\n'.format(proj_matrix, ortho_matrix))
signature, fingerprint = fingerprint_embedding.get_signature()
print('\nSignature:\n{}\n \nEmbedded fingerprint:\n{}\n'.format(signature, fingerprint))

# save the keys to file
save_to_hdf5(proj_matrix, ortho_matrix, fingerprint, signature, fingerprint_fname)
print(f'Data saved to {fingerprint_fname}')

# print the extracted fingerprint
show_encoded_signature(model)

# ---------------------------- Validate Training Accuracy and Save Model -----------------------------------------------

# make predictions using the pruned model
predictions_1 = model.predict(test_input)
predictions_2 = np.argmax(predictions_1, axis=1)
predictions_3 = kutils.to_categorical(predictions_2)

# compute the accuracy of the pruned model
accuracy = metrics.accuracy_score(test_output, predictions_3) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)

# save model
model.save(model_filepath)

'''
import matplotlib.pyplot as plt

# plot the results
training_accuracy = hist.history['accuracy']
validation_accuracy = hist.history['val_accuracy']

# Create a range of epochs for the x-axis
epochs = range(1, len(training_accuracy) + 1)

# Plot training and validation accuracy
plt.plot(epochs, training_accuracy, label='Training Accuracy')
plt.plot(epochs, validation_accuracy, label='Validation Accuracy')

# Add labels and title
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Show the plot
plt.show()
'''