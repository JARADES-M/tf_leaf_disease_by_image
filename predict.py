from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
from scipy import ndimage
import sys

#import matplotlib.image as mpimg

if __name__ == "__main__":
    image_file = sys.argv[1]
    pickle_file = 'data.pickle'

    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        labels = save['labels']
        del save

    image_size = 28
    num_channels = 3

    image_data = ndimage.imread(image_file).astype(float)
    #image_data = mpimg.imread(image_file)
    image_data.resize(image_size, image_size, 3)
    image_data = image_data.reshape(
    (1, image_size, image_size, num_channels)).astype(np.float32)

    batch_size = 1
    patch_size = 5
    depth = 16
    num_hidden = 64
    num_labels = 6

    graph = tf.Graph()

    with graph.as_default():

        # Dados de entrada
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))

        # Variaveis
        layer1_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, num_channels, depth], stddev=0.1), name="w1")
        layer1_biases = tf.Variable(tf.zeros([depth]),name='b1')
        layer2_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, depth, depth], stddev=0.1),'w2')
        layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]),'b2')
        layer3_weights = tf.Variable(tf.truncated_normal(
            [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1),'w3')
        layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]),'b3')
        layer4_weights = tf.Variable(tf.truncated_normal(
            [num_hidden, num_labels], stddev=0.1),'w4')
        layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]),'b4')

        # Modelo
        def model(data):
            conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer1_biases)
            conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer2_biases)
            shape = hidden.get_shape().as_list()
            reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
            hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
            return tf.matmul(hidden, layer4_weights) + layer4_biases

        # Inferencia
        logits = model(tf_train_dataset)
        train_prediction = tf.nn.softmax(logits)

        # opc salvar e restaurar variaveis
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as session:
        # Restaura variaveis salvas
        saver.restore(session, "/tmp/model.ckpt")
        print("Model restored.")
        # Verifica valores das variaveis
        feed_dict = {tf_train_dataset : image_data}
        predictions = session.run( [train_prediction], feed_dict=feed_dict)
        pred_index = predictions[0].argmax()

        print(labels[pred_index])


# Pra rodar 'python predict.py ImagePath'
