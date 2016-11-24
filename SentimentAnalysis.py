import cPickle as pickle

from numpy.distutils.system_info import numarray_info

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.dtypes import float32

num_labels = 5
num_words = 60;
embed_size = 128;

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
  sentences = save['sentences']
  del save  # hint to help gc free up memory
  #print('Training set', train_dataset.shape, train_labels.shape)
  #print('Validation set', valid_dataset.shape, valid_labels.shape)
  #print('Test set', test_dataset.shape, test_labels.shape)

num_labels = 5
num_words = 60;
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

def     genarate_batch(offset, batch_size):
    batch_data, batch_label = make_arrays(batch_size, num_words, embed_size)
    for index in xrange(batch_size):
        indexSen = train_dataset[offset + index];
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
        batch_label[index] = train_labels[offset + index]
    batch_label = Reshape(batch_label)
    return batch_data, batch_label

#valid_data, valid_lb = make_data_valid(valid_dataset, valid_labels)
#test_data, test_lb = make_data_valid(valid_dataset, valid_labels)
#print (valid_data.shape, valid_lb.shape)

print (train_labels.shape[0])


batch_size = 64
hidden_unit = 64
out_init = 5;

graph = tf.Graph()
with graph.as_default():
    def   weight_variable(shape):
        init_weight = tf.truncated_normal(shape, -1.0, 1.0)
        return tf.Variable(init_weight)

    def   bias_variable(shape):
        init_bias = tf.zeros(shape=shape)
        return tf.Variable(init_bias)

    tf_train_datasets = tf.placeholder(float32, shape=(batch_size, num_words, embed_size))
    tf_train_labels = tf.placeholder(float32, shape=(batch_size, num_labels))

    #tf_valid_datasets = tf.constant(valid_data)
    #tf_test_datasets = tf.constant(test_data)


    weight_W = weight_variable(shape=[embed_size, hidden_unit])
    weight_U = weight_variable(shape=[hidden_unit, hidden_unit])
    weight_V = weight_variable(shape=[hidden_unit, out_init])

    bias_1 = bias_variable(shape=[hidden_unit])
    bias_2 = bias_variable(shape=[out_init])

    activation = tf.zeros((1, hidden_unit))
    for iter in xrange(num_words):
        tmp = tf.matmul(tf_train_datasets[:,iter,:], weight_W)
        tmp += tf.matmul(activation, weight_U);
        activation = tf.tanh(tmp) + bias_1

    logits = tf.matmul(activation, weight_V) + bias_2

  #weight_decay = tf.constant(5.0) * (tf.nn.l2_loss(weight_1) + tf.nn.l2_loss(weight_2))/ (train_labels.shape[0])

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    optimizer = tf.train.AdagradOptimizer(0.01).minimize(loss)
    train_prediction = tf.nn.softmax(logits)


def   accuracy(predictions, labels):
    return 100 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

num_steps = 10001


with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print ("Initialized")
    for step in xrange(num_steps):
        if (step % 2000 == 0):
            train_dataset, train_labels = randomize(train_dataset, train_labels)
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data, batch_labels = genarate_batch(offset, batch_size)
        feed_dict = {tf_train_datasets : batch_data, tf_train_labels : batch_labels}

        _, costFunction, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step < 10):
            print (predictions)
            print('Loss at step %d: %f' % (step, costFunction))
            print('Training accuracy: %.2f%%' % accuracy(predictions, batch_labels))
      #print('Validation accuracy: %.2f%%' % accuracy(valid_prediction.eval(), valid_labels))
      #print("Test accuracy: %.2f%%" % accuracy(test_prediction.eval(), test_labels))
