import cPickle as pickle

from numpy.distutils.system_info import numarray_info

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.dtypes import float32
from tensorflow.python.ops import rnn, rnn_cell



def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

embedding_file = 'embedding.pickle';
with open(embedding_file, 'rb') as f:
    save = pickle.load(f);
    dictionary = save['dictionary'];
    reverse_dictionary = save['reverse_dictionary'];
    embeddings = save['embedding'];
    sentences = save['sentences']
    max_sentence = save['max_sentence']
    del save;


pickle_file = 'movie_review.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  #print('Training set', train_dataset.shape, train_labels.shape)
  #print('Validation set', valid_dataset.shape, valid_labels.shape)
  #print('Test set', test_dataset.shape, test_labels.shape)

num_labels = 2
num_words = max_sentence;
embed_size = 128;


def make_arrays(num_sentence, num_words, embed_size):
    dataset = np.ndarray((num_sentence, num_words, embed_size), dtype=np.float32)
    labels = np.ndarray(num_sentence, dtype=np.int32)
    return dataset, labels

def	Reshape(labels):
  labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
  return labels

def     make_data_valid(data, label):
    lent = len(data)
    valid_data, valid_label = make_arrays(lent, num_words, embed_size)
    for index in xrange(lent):
        indexSen = data[index]
        listWord = sentences[indexSen]
        cnt = 0;
        lent = len(listWord)
        for word in listWord:
            lowWord = word.lower();
            indexW = dictionary[lowWord]
            valid_data[index, num_words - lent + cnt, :] = embeddings[indexW]
            cnt += 1
        for i in xrange(num_words - lent):
            valid_data[index, i, :] = np.zeros(shape=(1, embed_size))
        valid_label[index] = label[index]
    valid_label = Reshape(valid_label)
    return valid_data, valid_label;

def     genarate_batch(offset, batch_size, data, labels):
    batch_data, batch_label = make_arrays(batch_size, num_words, embed_size)
    for index in xrange(batch_size):
        indexSen = data[offset + index];
        listWord = sentences[indexSen]
        cnt = 0;
        lent = len(listWord)
        if (lent == 0):
            print ("FALSE")
        for word in listWord:
            lowWord = word.lower();
            indexW = dictionary[lowWord]
            batch_data[index, num_words - lent + cnt, :] = embeddings[indexW]
            cnt += 1
        for i in xrange(num_words - lent):
            batch_data[index, i, :] = np.zeros(shape=(1, embed_size))
        batch_label[index] = labels[offset + index]
    batch_label = Reshape(batch_label)
    return batch_data, batch_label

#valid_data, valid_lb = make_data_valid(valid_dataset, valid_labels)
test_data, test_lb = make_data_valid(test_dataset, test_labels)


#print (test_data.shape, len(test_data))

batch_size = 128

# Network Parameters
n_input = 128
n_steps = max_sentence
n_hidden = 128
n_classes = 2


# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
istate = tf.placeholder("float", [None, 2*n_hidden]) #state & cell => 2x n_hidden
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(_X, _istate, _weights, _biases):

    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) # (n_steps*batch_size, n_input)
    # Linear activation
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(0, n_steps, _X) # n_steps * (batch_size, n_hidden)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, _X, initial_state=_istate)

    # Linear activation
    # Get inner loop last output
    return tf.matmul(outputs[-1], _weights['out']) + _biases['out']

def   accuracy(predictions, labels):
    return 100 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

num_steps = 2000


logits = RNN(x, istate, weights, biases)
prediction = tf.nn.softmax(logits)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print('initilize')
    for step in xrange(num_steps):
        if (step % 10 == 0):
            print (step)
        if (step % 60 == 0):
            train_dataset, train_labels = randomize(train_dataset, train_labels)
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data, batch_labels = genarate_batch(offset, batch_size, train_dataset, train_labels)
        feed_dict = {x : batch_data, y : batch_labels, istate: np.zeros((batch_size, 2*n_hidden))}
        _, c, predic = sess.run([optimizer, cost, prediction], feed_dict=feed_dict)

        if (step % 300 == 0):
            print('Loss at step %d: %f' % (step, c))
            print('Training Accuracy:', accuracy.eval({x : batch_data, y : batch_labels, istate: np.zeros((batch_size, 2*n_hidden))}))
        if (step % 500):
            print ('Testing Accuracy', accuracy.eval({x : test_data, y : test_lb, istate: np.zeros((len(test_lb), 2*n_hidden))}))

