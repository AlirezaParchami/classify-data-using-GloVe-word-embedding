from __future__ import print_function
import argparse
import pprint

from glove import Glove
from glove import Corpus


def read_corpus(filename):

    with open(filename, 'r') as datafile:
        for line in datafile:
            #print(line.lower().split(' '))
            yield line.lower().split(' ')





# Build the corpus dictionary and the cooccurrence matrix.
print('Pre-processing corpus')

get_data = read_corpus("text7.txt")

corpus_model = Corpus()
corpus_model.fit(get_data, window=10)
corpus_model.save('corpus.model')

print('Dict size: %s' % len(corpus_model.dictionary))
print('Collocations: %s' % corpus_model.matrix.nnz)


        # Train the GloVe model and save it to disk.

        # if not args.create:
        #     # Try to load a corpus from disk.
        #     print('Reading corpus statistics')
        #     corpus_model = Corpus.load('corpus.model')
        #
        #     print('Dict size: %s' % len(corpus_model.dictionary))
        #     print('Collocations: %s' % corpus_model.matrix.nnz)

print('Training the GloVe model')

glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus_model.matrix, epochs=int(10),
          no_threads=4 ,verbose=True)
glove.add_dictionary(corpus_model.dictionary)
glove.save('glove.model')

        # Finally, query the model for most similar words.

        # if not args.train:
        #     print('Loading pre-trained GloVe model')
        #     glove = Glove.load('glove.model')

#print('Querying for %s' % args.query)
#pprint.pprint(glove.most_similar(args.query, number=10))