# export TF_CPP_MIN_LOG_LEVEL=2 # ocultar mensagem de erro em compilacao executar antes de executar Train.py
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

if __name__ == "__main__":
    pickle_file = 'data.pickle'

    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # liberar memoria
        # print('Training set', train_dataset.shape, train_labels.shape)
        # print('Test set', test_dataset.shape, test_labels.shape)

    image_size = 28
    num_labels = 6
    num_channels = 3 # RGB

    # print("After formatting")
    train_dataset, train_labels = reformat(train_dataset, train_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)
    # print('Training set', train_dataset.shape, train_labels.shape)
    # print('Test set', test_dataset.shape, test_labels.shape)

    #TensorFlow
    batch_size = 16
    patch_size = 5
    depth = 16
    num_hidden = 64

    graph = tf.Graph()

    with graph.as_default():

        # Dados de entrada.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_test_dataset = tf.constant(test_dataset)

        # Variaveis.
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

        # Model
        # input - conv - conv - linear(fc) - linear(fc)
        def model(data): # input Layer

        	# 1 conv Layer
            conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer1_biases) # Activation function

            # 1 conv Layer
            conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer2_biases) # Activation function
            
            # not a layer ( just reshape)
            shape = hidden.get_shape().as_list()
            reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

            # 1 linear fully connected layer + relu
            hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
            
            # 1 linear fully connected layer
            return tf.matmul(hidden, layer4_weights) + layer4_biases

        # Training computation.
        logits = model(tf_train_dataset)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))


        # Otimizador
        optimizer = tf.train.AdamOptimizer(0.00001).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        test_prediction = tf.nn.softmax(model(tf_test_dataset))
        saver = tf.train.Saver()

    # num_steps = 2001
    num_steps = 2201
    # plt graph
    step_arr = []
    accur_arr = []

    with tf.Session(graph=graph) as session:
        # tf.initialize_all_variables().run()
        tf.global_variables_initializer().run()
        print('Treinamento iniciado..')
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
            accur = accuracy(predictions, batch_labels)
            step_arr.append(step)
            accur_arr.append(accur)
            if (step % 150 == 0):
                # print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy step %d: %.2f%%' % (step, accur))
        plt.plot(step_arr, accur_arr, linewidth=1)
        plt.title('Acurácia x Iteração')
        plt.ylabel('Acurácia %')
        plt.xlabel('Iteração')
        plt.show()

        print('Validação em dataset de teste..')
        print('Test accuracy: %.2f%%' %  accuracy(test_prediction.eval(), test_labels))

        # Salvar variaveis em disco
        save_path = saver.save(session, "/tmp/model.ckpt")
        print("Model saved in file: %s" % save_path)
