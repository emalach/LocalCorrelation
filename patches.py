import math, json, os, sys
import tensorflow as tf
import keras
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Lambda, LocallyConnected2D, Reshape, GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.datasets import cifar10


import pickle

import numpy as np

def pooled_categorical_crossentropy(y_true,y_pred):
    h = y_pred.shape.as_list()[1]
    w = y_pred.shape.as_list()[2]
    return float(w*h)*K.categorical_crossentropy(y_true, y_pred)

def pooled_top_k_categorical_accuracy(y_true, y_pred, k=5):
    h = y_pred.shape.as_list()[1]
    w = y_pred.shape.as_list()[2]
    c = y_pred.shape.as_list()[3]
    yt = K.tile(y_true, (1,h,w,1))
    yt = K.reshape(yt, (-1, c))
    yp = K.reshape(y_pred, (-1,c))
    in_top_k = tf.cast(tf.nn.in_top_k(yp, tf.argmax(yt, axis=-1), k), K.floatx())
    return in_top_k


def multiply_labels(image_generator, num_classes=10, names=None):
    for x, y in image_generator:
        y_ = y.reshape((-1,1,1,num_classes))
        if names is None:
            yield x, y_
        else:
            yield x, {n: y_ for n in names}


if __name__ == "__main__":
    dataset = 'imagenet'
    model_type = 'linear'
    BATCH_SIZE = 16

    if dataset == 'imagenet':
        DATA_DIR = '/mnt/ssd/VD/imagenet/'
        TRAIN_DIR = os.path.join(DATA_DIR, 'train')
        VALID_DIR = os.path.join(DATA_DIR, 'valid')
        width = height = 224
        SIZE = (height, width)

        num_train_samples = 1281167 #sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
        num_valid_samples = 50000 #sum([len(files) for r, d, files in os.walk(VALID_DIR)])

        num_train_steps = math.floor(num_train_samples/BATCH_SIZE)
        num_valid_steps = math.floor(num_valid_samples/BATCH_SIZE)

        gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=keras.applications.resnet50.preprocess_input)
        val_gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=keras.applications.resnet50.preprocess_input)

        batches = gen.flow_from_directory(TRAIN_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
        val_batches = val_gen.flow_from_directory(VALID_DIR, target_size=SIZE, class_mode='categorical', shuffle=False, batch_size=BATCH_SIZE)
        
        classes = list(iter(batches.class_indices))
        num_classes = len(classes)

        val_data = multiply_labels(val_batches, num_classes=num_classes)
        grid_size = 7
        epochs = 10
    elif dataset == 'cifar':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        num_classes = 10
        
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        y_train = y_train[:,None,None,:]
        y_test = y_test[:,None,None,:]
        
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        
        width = height = x_train.shape[1]
        SIZE = (height, width)
        epochs = 10
        
        val_data = zip([x_test], [y_test])
        num_valid_steps = 1
        grid_size = 8
 
    strides = (int(SIZE[0]/grid_size), int(SIZE[1]/grid_size))
    model = Sequential()
    if model_type == 'linear':
        model.add(LocallyConnected2D(num_classes, (3, 3), strides=strides, activation="softmax", input_shape=(height,width,3)))
    elif model_type == 'nn':
        model.add(LocallyConnected2D(128, (3, 3), strides=strides, activation="relu", input_shape=(height,width,3)))
        model.add(LocallyConnected2D(num_classes, (1, 1), strides=(1,1), activation="softmax"))

    opt = keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=opt, loss=pooled_categorical_crossentropy, metrics=['accuracy', pooled_top_k_categorical_accuracy])

    checkpointer = ModelCheckpoint('patches_best.h5', verbose=1, save_best_only=True)
    if dataset == 'imagenet':
        model.fit_generator(multiply_labels(batches, num_classes=num_classes), steps_per_epoch=num_train_steps, epochs=epochs, callbacks=[checkpointer])
    else:
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=epochs, shuffle=True)
    
    stats = {}
    stats['count'] = 0
    for x_batch, y_batch in val_data:
        if stats['count'] == num_valid_steps:
            break
        if stats['count'] % 10 == 0:
            print(stats['count'])
        pred = model.predict(x_batch)
        mean_acc = np.mean((np.argmax(pred, axis=-1)==np.argmax(y_batch, axis=-1)), axis=0)
        mean_top_k = np.mean(np.any(np.argsort(pred, axis=-1)[...,-1:-6:-1] == np.argmax(y_batch), axis=-1), axis=0)
        if 'mean_acc' not in stats:
            stats['mean_acc'] = mean_acc
            stats['mean_top_k'] = mean_top_k
        else:
            stats['mean_acc'] += mean_acc
            stats['mean_top_k'] += mean_top_k

        if stats['count'] == 0:
            stats['sample_x'] = x_batch[:100]
            stats['sample_y'] = y_batch[:100]
            stats['sample_pred'] = pred[:100]

        stats['count'] += 1

    pickle.dump(stats, open('patches_experiment_%s_%d_%s_epochs.pickle' % (dataset, epochs, model_type), 'wb'))