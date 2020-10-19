import math, json, os, sys
import tensorflow as tf
import keras
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Lambda, LocallyConnected2D, Reshape, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image

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



DATA_DIR = '/mnt/ssd/VD/imagenet/'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')
SIZE = (224, 224)
BATCH_SIZE = 16

def multiply_labels(image_generator, num_classes=10, names=None):
    for x, y in image_generator:
        y_ = y.reshape((-1,1,1,num_classes))
        if names is None:
            yield x, y_
        else:
            yield x, {n: y_ for n in names}


if __name__ == "__main__":
    num_train_samples = 1281167 #sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
    num_valid_samples = 50000 #sum([len(files) for r, d, files in os.walk(VALID_DIR)])

    num_train_steps = math.floor(num_train_samples/BATCH_SIZE)
    num_valid_steps = math.floor(num_valid_samples/BATCH_SIZE)

    gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=keras.applications.resnet50.preprocess_input)
    val_gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=keras.applications.resnet50.preprocess_input)

    batches = gen.flow_from_directory(TRAIN_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
    val_batches = val_gen.flow_from_directory(VALID_DIR, target_size=SIZE, class_mode='categorical', shuffle=False, batch_size=BATCH_SIZE)

    model = keras.applications.resnet50.ResNet50()

    classes = list(iter(batches.class_indices))

    for layer in model.layers:
        layer.trainable=False

    outputs = []
    names = []
    losses = {}
    layers_to_use = [l for l in model.layers if 'activation' in l.name]
    for layer_to_use in layers_to_use[::4]:
        cur_output = layer_to_use.output
        cur_stride = int(cur_output.shape.as_list()[1]/7)
        cur_name = '%s_readout' % layer_to_use.name
        x = LocallyConnected2D(len(classes), (1, 1), strides=(cur_stride,cur_stride),
                                activation="softmax", name=cur_name)(cur_output)
        outputs.append(x)
        names.append(cur_name)
        losses[cur_name] = pooled_categorical_crossentropy
    
    opt = keras.optimizers.Adam(lr=0.0001)
    finetuned_model = Model(model.input, outputs=outputs)
    metrics = {o.name: 'accuracy' for o in outputs}
    finetuned_model.compile(optimizer=opt, loss=pooled_categorical_crossentropy, metrics=['accuracy', pooled_top_k_categorical_accuracy])
    for c in batches.class_indices:
        classes[batches.class_indices[c]] = c
    finetuned_model.classes = classes

    early_stopping = EarlyStopping(patience=10)
    checkpointer = ModelCheckpoint('resnet50_best.h5', verbose=1, save_best_only=True)
    finetuned_model.fit_generator(multiply_labels(batches, num_classes=len(classes), names=names), steps_per_epoch=num_train_steps, epochs=5, callbacks=[checkpointer])
    stats = {}
    stats['count'] = 0
    for x_batch, y_batch in multiply_labels(val_batches, num_classes=len(classes)):
        if stats['count'] == num_valid_steps:
            break
        if stats['count'] % 10 == 0:
            print(stats['count'])
        pred = finetuned_model.predict(x_batch)
        for ni, name in enumerate(names):
            if name not in stats:
                stats[name] = {}

            mean_acc = np.mean((np.argmax(pred[ni], axis=-1)==np.argmax(y_batch, axis=-1)), axis=0)
            mean_top_k = np.mean(np.any(np.argsort(pred[ni], axis=-1)[...,-1:-6:-1] == np.argmax(y_batch), axis=-1), axis=0)
            if 'mean_acc' not in stats[name]:
                stats[name]['mean_acc'] = mean_acc
                stats[name]['mean_top_k'] = mean_top_k
            else:
                stats[name]['mean_acc'] += mean_acc
                stats[name]['mean_top_k'] += mean_top_k
        stats['count'] += 1

    pickle.dump(stats, open('experiment.pickle', 'wb'))