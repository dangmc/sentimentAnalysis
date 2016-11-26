import csv;
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import nltk.tokenize
from nltk.corpus import stopwords
import pandas as pd
import re


import cPickle as pickle

trainDataset = "Data/train.txt";

class   Word2Vec:

    def     __init__(self):
        self.maxSentence = 0;
        self.numberSentence = 0;
        self.sentences = [];
        self.sentenceId = 0;
        self.index = 0;
        self.dictionary = dict();
        self.label = []

    def     readFileImDb(self, filename):
            reader  = pd.read_table(filename, names=['sentiment', 'comment'], header=None, delimiter="\t", quoting=3)
            print (reader.sentiment.value_counts())
            stop_words = set(stopwords.words('english'))
            for text in reader.comment:
                word_tokens = re.sub("[^a-zA-Z]", " ", text)
                word_tokens = nltk.word_tokenize(word_tokens.lower())
                sentence = [w for w in word_tokens if not w in stop_words]
                #print (len(word_tokens), len(sentence))
                self.maxSentence = max(self.maxSentence, len(sentence));
                if (len(sentence) > 0):
                    self.sentences.append(sentence)
                    self.numberSentence += 1
                for word in sentence:
                    if (word not in self.dictionary):
                       self.dictionary[word] = len(self.dictionary)
            for label in reader.sentiment:
                self.label.append(label)
            self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))

    def generate_batch(self, batch_size, num_skips, skip_window):
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1 # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            #print (self.sentenceId, len(self.sentences[self.sentenceId]), self.index);
            word = self.sentences[self.sentenceId][self.index].lower();
            index = self.dictionary[word];
            buffer.append(index);
            if (self.index + 1 < len(self.sentences[self.sentenceId])):
                self.index += 1;
            else:
                self.index = 0;
                self.sentenceId = (self.sentenceId + 1) % self.numberSentence;
        for i in range(batch_size // num_skips):
            target = skip_window  # target label at the center of the buffer
            targets_to_avoid = [ skip_window ]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]
            #print (self.sentenceId, len(self.sentences[self.sentenceId]), self.index);
            word = self.sentences[self.sentenceId][self.index].lower();
            index = self.dictionary[word];

            buffer.append(index);
            if (self.index + 1 < len(self.sentences[self.sentenceId])):
                self.index += 1;
            else:
                self.index = 0;
                self.sentenceId = (self.sentenceId + 1) % self.numberSentence;
        return batch, labels
    def     Training(self):
        vocabulary_size = len(self.dictionary);
        batch_size = 128
        embedding_size = 128 # Dimension of the embedding vector.
        skip_window = 2 # How many words to consider left and right.
        num_skips = 4 # How many times to reuse an input to generate a label.
        # We pick a random validation set to sample nearest neighbors. here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent.
        valid_size = 16 # Random set of words to evaluate similarity on.
        valid_window = 100 # Only pick dev samples in the head of the distribution.
        valid_examples = np.array(random.sample(range(valid_window), valid_size))
        num_sampled = 64 # Number of negative examples to sample.

        graph = tf.Graph()

        with graph.as_default(), tf.device('/cpu:0'):

              # Input data.
              train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
              train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
              valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

              # Variables.
              embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
              softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],stddev=1.0 / math.sqrt(embedding_size)))
              softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

              # Model.
              # Look up embeddings for inputs.
              embed = tf.nn.embedding_lookup(embeddings, train_dataset)
              # Compute the softmax loss, using a sample of the negative labels each time.
              loss = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed,train_labels, num_sampled, vocabulary_size))

              # Optimizer
              optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

              # Compute the similarity between minibatch examples and all embeddings.
              # We use the cosine distance:
              norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
              normalized_embeddings = embeddings / norm
              valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
              similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))
        num_steps = 80001

        with tf.Session(graph=graph) as session:
              tf.initialize_all_variables().run()
              print('Initialized')
              average_loss = 0
              for step in range(num_steps):
                    batch_data, batch_labels = self.generate_batch(batch_size, num_skips, skip_window)
                    feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
                    _, l = session.run([optimizer, loss], feed_dict=feed_dict)
                    average_loss += l
                    if step % 2000 == 0:
                          if step > 0:
                            average_loss = average_loss / 2000
                          # The average loss is an estimate of the loss over the last 2000 batches.
                          print('Average loss at step %d: %f' % (step, average_loss))
                          average_loss = 0
                    # note that this is expensive (~20% slowdown if computed every 500 steps)
                    if step % 10000 == 0:
                          sim = similarity.eval()
                          for i in range(valid_size):
                                valid_word = self.reverse_dictionary[valid_examples[i]]
                                top_k = 8 # number of nearest neighbors
                                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                                log = 'Nearest to %s:' % valid_word
                                for k in range(top_k):
                                      close_word = self.reverse_dictionary[nearest[k]]
                                      log = '%s %s,' % (log, close_word)
                                print(log)
              self.final_embeddings = normalized_embeddings.eval()
    def     run(self):
        self.readFileImDb(trainDataset)
        self.Training();
        embedding_file = 'embedding.pickle'
        try:
          f = open(embedding_file, 'wb')
          save = {
            'dictionary' : self.dictionary,
            'reverse_dictionary' : self.reverse_dictionary,
            'embedding' : self.final_embeddings,
            'sentences' : self.sentences,
            'labels' : self.label,
            'max_sentence' : self.maxSentence
            }
          pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
          f.close()
        except Exception as e:
          print('Unable to save data to', embedding_file, ':', e)
          raise
        statinfo = os.stat(embedding_file)
        print('Compressed pickle size:', statinfo.st_size)


word2vec = Word2Vec();
word2vec.run()




        #for key in dictionary:
         #   print(key, dictionary[key]);