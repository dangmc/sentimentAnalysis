from __future__ import print_function

import numpy as np
import os
import cPickle as pickle


# Config the matlotlib backend as plotting inline in IPython
np.random.seed(133)

training_file = 'Data/train.tsv';

embedding_file = 'embedding.pickle';
with open(embedding_file, 'rb') as f:
    save = pickle.load(f);
    dictionary = save['dictionary'];
    reverse_dictionary = save['reverse_dictionary'];
    embeddings = save['embedding'];
    max_sentence = save['max_sentence']
    label = save['labels']
    sentences = save['sentences']
f.close();


def make_arrays(num_sentence):
    dataset = np.ndarray(num_sentence, dtype=np.int32)
    labels = np.ndarray(num_sentence, dtype=np.int32)
    return dataset, labels

numRow = len(label)
print (numRow)
datasets, labels = make_arrays(numRow)
for i in xrange(numRow):
    datasets[i] = i;
    labels[i] = label[i]

def     randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels;

datasets, labels = randomize(datasets, labels)

train_size = 5000
valid_size = 2000


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
    'test_labels': test_labels
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

