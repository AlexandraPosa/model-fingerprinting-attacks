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
from embed_fingerprint import extract_fingerprint

# ------------------------------------- Initialization and Configuration -----------------------------------------------

# set seed
seed_value = 2
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

extracted_fingerprint_fname = os.path.join(result_path, 'extracted_fingerprint.h5')
embedding_keys_fname = os.path.join(result_path, 'embedding_keys.h5')

training_history_fname = os.path.join(result_path, 'model_training_history.h5')
model_checkpoint_fname = os.path.join(result_path, 'model_checkpoint.h5')

model_filepath = os.path.join(result_path, 'embedded_model.keras') \
    if train_settings['embed_flag'] == True \
    else os.path.join(result_path, 'non_embedded_model.keras')

# ---------------------------------------- Save and Load Functions -----------------------------------------------------

# save the keys used for the fingerprint embedding to an HDF5 file
def save_embedding_keys(proj_matrix, ortho_matrix, fingerprint, signature, fingerprint_fname):
    with h5py.File(fingerprint_fname, 'w') as hdf5_file:
        hdf5_file.create_dataset('proj_matrix', data=proj_matrix)
        hdf5_file.create_dataset('ortho_matrix', data=ortho_matrix)
        hdf5_file.create_dataset('signature', data=signature)
        hdf5_file.create_dataset('fingerprint', data=fingerprint)

# save the model training history to an HDF5 file
def save_training_history(history, filename):
    with h5py.File(filename, 'w') as hf:
        for key, value in history.history.items():
            hf.create_dataset(key, data=value)

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
        return 0.1
    elif (epoch_idx + 1) < lr_schedule[1]:
        return 0.02
    elif (epoch_idx + 1) < lr_schedule[2]:
        return 0.004
    return 0.0008

# ------------------------------- Data Preprocessing and Augmentation --------------------------------------------------

# load data
(train_input, train_output), (test_input, test_output) = dataset.load_data()
train_input = train_input.astype('float32') / 255.0
test_input = test_input.astype('float32') / 255.0
train_output = kutils.to_categorical(train_output)
test_output = kutils.to_categorical(test_output)

# calculate the sizes for training and validation
total_samples = len(train_input)
train_size = int(0.8 * total_samples)  # 80% for training
validation_size = total_samples - train_size  # 20% for validation during training

# split the training data
training_input = train_input[:train_size]
training_output = train_output[:train_size]

# split the validation data
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
              optimizer=SGD(lr=0.1, momentum=0.87, nesterov=True),
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

# save training history to file
save_training_history(hist, training_history_fname)
print(f'Model training history saved to {training_history_fname}')

# ------------------------------------- Save Fingerprint Information ---------------------------------------------------

# access the keys used for the embedding
proj_matrix, ortho_matrix = fingerprint_embedding.get_matrix()
signature, fingerprint = fingerprint_embedding.get_signature()

# save the keys used for the embedding to an HDF5 file
save_embedding_keys(proj_matrix, ortho_matrix, fingerprint, signature, embedding_keys_fname)
print(f'Embedding information saved to {embedding_keys_fname}')

# extract the encoded fingerprint and save it to an HDF5 file
extract_fingerprint(model, extracted_fingerprint_fname)

# ---------------------------- Validate Training Accuracy and Save Model -----------------------------------------------

# make predictions using the pruned model
print("\nAssessing the performance on the model:")
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
print(f'Model saved to {model_filepath}')


