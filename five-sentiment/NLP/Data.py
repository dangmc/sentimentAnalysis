from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from urllib import urlretrieve
import cPickle as pickle
import csv;
import tensorflow as tf;

# Config the matlotlib backend as plotting inline in IPython
num_words = 60;
embed_size = 128;
np.random.seed(133)

training_file = 'Data/train.tsv';

embedding_file = 'embedding.pickle';
with open(embedding_file, 'rb') as f:
    save = pickle.load(f);
    dictionary = save['dictionary'];
    reverse_dictionary = save['reverse_dictionary'];
    embeddings = save['embedding'];
f.close();


def make_arrays(num_sentence, num_words, embed_size):
    dataset = np.ndarray(num_sentence, dtype=np.float32)
    labels = np.ndarray(num_sentence, dtype=np.int32)
    return dataset, labels

def     readFile(filename):
    with open(filename) as file:
        reader = csv.reader(file, delimiter='\t');
        numberRow = 0;
        cntRow = 156060;
        datasets, labels = make_arrays(cntRow - 1, num_words, embed_size)
        sentences = np.ndarray(cntRow - 1, dtype=list)
        for row in reader:
            if (numberRow > 0):
                listWord = row[2].split();
                sentences[numberRow - 1] = listWord
                datasets[numberRow - 1] = numberRow - 1;
                labels[numberRow - 1] = row[3]
            numberRow += 1;
        return datasets, labels, sentences;

datasets, labels, sentences = readFile(training_file)

def     randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels;

datasets, labels = randomize(datasets, labels)

train_size = 120000
valid_size = 20000


training_datasets = datasets[0:train_size];
training_labels = labels[0:train_size]

test_datasets = datasets[train_size:]
test_labels = labels[train_size:]

training_datasets, training_labels = randomize(training_datasets, training_labels)

validation_dataset = training_datasets[0:valid_size]
validation_labels = training_labels[0:valid_size]

training_datasets, training_labels = randomize(training_datasets, training_labels)
validation_dataset, validation_labels = randomize(validation_dataset, validation_labels)
test_datasets, test_labels = randomize(test_datasets, test_labels)

print (sentences[0])

pickle_file = 'movie_review.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': training_datasets,
    'train_labels': training_labels,
    'valid_dataset': validation_dataset,
    'valid_labels': validation_labels,
    'test_dataset': test_datasets,
    'test_labels': test_labels,
    'sentences' : sentences
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)


def     readFile_old(filename):
    with open(filename) as file:
        reader = csv.reader(file, delimiter='\t');
        numberRow = 0;
        cntRow = 156060;
        datasets, labels = make_arrays(cntRow - 1, num_words, embed_size)

        for row in reader:
            if (numberRow > 0):
                listWord = row[2].split();
                cnt = 0;
                lent = len(listWord)
                for word in listWord:
                    lowWord = word.lower();
                    index = dictionary[lowWord]
                    datasets[numberRow - 1, num_words - lent + cnt, :] = embeddings[index]
                    cnt += 1
                for i in xrange(num_words - lent):
                    datasets[numberRow - 1, i, :] = np.zeros(shape=(1, embed_size))
                labels[numberRow - 1] = row[3]
            numberRow += 1;
        return datasets, labels;