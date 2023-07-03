# import libraries
import numpy as np
import random
import sys
import json
import os
import tensorflow as tf
import sklearn.metrics as metrics
import wide_residual_network as wrn
import keras.utils.np_utils as kutils
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from custom_regularizer import CustomRegularizer
from custom_regularizer import show_encoded_signature

# set a seed
seed_value = 0
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)

# set path
RESULT_PATH = './result'
MODEL_CHKPOINT_FNAME = os.path.join(RESULT_PATH, 'WRN-Weights.h5')

# set learning rate
lr_schedule = [60, 120, 160]  # epoch_step

def schedule(epoch_idx):
    if (epoch_idx + 1) < lr_schedule[0]:
        return 0.08
    elif (epoch_idx + 1) < lr_schedule[1]:   # lr_decay_ratio = 0.2
        return 0.016
    elif (epoch_idx + 1) < lr_schedule[2]:
        return 0.0032
    return 0.00064

if __name__ == '__main__':
    settings_json_fname = sys.argv[1]
    train_settings = json.load(open(settings_json_fname))
    
    if not os.path.isdir(RESULT_PATH):
        os.makedirs(RESULT_PATH)
        
    # load dataset and fitting data for learning
    if train_settings['dataset'] == 'cifar10':
        dataset = cifar10
        nb_classes = 10
    else:
        print('not supported dataset "{}"'.format(train_settings['dataset']))
        exit(1)

    (trainX, trainY), (testX, testY) = dataset.load_data()
    trainX = trainX.astype('float32')
    trainX /= 255.0
    testX = testX.astype('float32')
    testX /= 255.0
    trainY = kutils.to_categorical(trainY)
    testY = kutils.to_categorical(testY)

    generator = ImageDataGenerator(rotation_range=10,
                                   width_shift_range=5./32,
                                   height_shift_range=5./32,
                                   horizontal_flip=True)
    generator.fit(trainX, augment=True)

    if 'replace_train_y' in train_settings and len(train_settings['replace_train_y']) > 0:
        print('trainY was replaced from "{}"'.format(train_settings['replace_train_y']))
        trainY = np.load(train_settings['replace_train_y'])

    # read parameters
    batch_size = train_settings['batch_size']
    nb_epoch = train_settings['epoch']
    scale = train_settings['scale']
    embed_dim = train_settings['embed_dim']
    apply_penalty = train_settings['apply_penalty']
    N = train_settings['N']
    k = train_settings['k']
    target_blk_id = train_settings['target_blk_id']

    # create model
    fingerprint_embedding = CustomRegularizer(strength=scale, embed_dim=embed_dim, seed=seed_value,
                                              apply_penalty=apply_penalty)
    init_shape = (3, 32, 32) if K.image_data_format() == "channels_first" else (32, 32, 3)
    model = wrn.create_wide_residual_network(init_shape, nb_classes=nb_classes, N=N, k=k, dropout=0.00,
                                             custom_regularizer=fingerprint_embedding, target_blk_num=target_blk_id)
    model.summary()

    # training process
    sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    print("Finished compiling")

    hist = \
    model.fit(generator.flow(trainX, trainY, batch_size=batch_size),
              steps_per_epoch=np.ceil(len(trainX)/batch_size), epochs=nb_epoch,
              callbacks=[ModelCheckpoint(MODEL_CHKPOINT_FNAME, monitor="val_accuracy", save_best_only=True),
                         LearningRateScheduler(schedule=schedule)],
              validation_data=(testX, testY),
              validation_steps=np.ceil(len(testX)/batch_size))

    # print the keys used for the embedding

    proj_matrix, ortho_matrix = fingerprint_embedding.get_matrix()
    print('\nProjection matrix.\n{}\n \nOrthogonal matrix:\n{}\n'.format(proj_matrix, ortho_matrix))

    signature, coefficient = fingerprint_embedding.get_signature()
    print('\nSignature:\n{}\n \nCoeffients:\n{}\n'.format(signature, coefficient))

    show_encoded_signature(model)

    # validate training accuracy
    yPreds = model.predict(testX)
    yPred = np.argmax(yPreds, axis=1)
    yPred = kutils.to_categorical(yPred)
    yTrue = testY

    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    error = 100 - accuracy
    print("Accuracy : ", accuracy)
    print("Error : ", error)
