import numpy as np
import pdb
import tensorflow as tf
import time
import matplotlib.pyplot as plt

def parity_data(m, d, k, bias):
    X = np.sign(np.random.uniform(size=(m,d)) - bias)
    Y = np.prod(X[:,:k],axis=1)
    return X, Y

def Affine(input_tensor, out_channels, relu=True):
    input_shape = input_tensor.get_shape().as_list()
    input_channels = input_shape[-1]
    weights = tf.Variable(
        tf.truncated_normal([input_channels, out_channels],
                            stddev=1.0 / np.sqrt(float(input_channels))))
    biases = tf.Variable(tf.zeros([out_channels]))
    if relu:
        return tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    else:
        return tf.matmul(input_tensor, weights) + biases

def train_and_test(X, Y, Xtest, Ytest, k, lr, batch_size, max_iters):
    m, d = X.shape

    with tf.Graph().as_default():
        X_placeholder = tf.placeholder(tf.float32, shape=(None, d))

        h = X_placeholder
        for i, out_channels in enumerate(k):
            h = Affine(h, out_channels, relu=(i != len(k) - 1))

        Y_placeholder = tf.placeholder(tf.float32, shape=(None, 1))
        loss = tf.reduce_mean(tf.nn.relu(1 - h*Y_placeholder))
        accuracy = 1 - 0.5*tf.reduce_mean(tf.abs(tf.sign(h)-Y_placeholder))

        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        for step in xrange(max_iters+1):
            examples = np.random.randint(X.shape[0], size=batch_size)
            feed_dict = {X_placeholder: X[examples, :], Y_placeholder: Y[examples, np.newaxis]}
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            if step % (max_iters/20) == 0 or step == max_iters:
                train_loss, train_accuracy = sess.run([loss, accuracy], feed_dict={X_placeholder: X, Y_placeholder: Y[:, np.newaxis]})
                test_loss, test_accuracy = sess.run([loss, accuracy], feed_dict={X_placeholder: Xtest, Y_placeholder: Ytest[:, np.newaxis]})
                print('%d\t%.2f' % (step, test_accuracy))
                if step == max_iters - 1:
                    return train_loss, test_loss, train_accuracy, test_accuracy


if __name__ == '__main__':
    print('WITH BIAS:')
    m, mtest, d, k, bias = 1000000, 1000, 128, 5, 0.4
    X, Y = parity_data(m, d, k, bias)
    Xtest, Ytest = parity_data(mtest, d, k, bias)
    train_and_test(X/np.sqrt(d), Y, Xtest/np.sqrt(d), Ytest, [128, 1], 0.01, 64, 20000)

    print('WITHOUT BIAS:')
    bias = 0.5
    X, Y = parity_data(m, d, k, bias)
    Xtest, Ytest = parity_data(mtest, d, k, bias)
    train_and_test(X/np.sqrt(d), Y, Xtest/np.sqrt(d), Ytest, [128, 1], 0.01, 64, 20000)
