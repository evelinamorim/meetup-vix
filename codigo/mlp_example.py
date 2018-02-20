import os
import numpy as np
from nltk import word_tokenize

from gensim.models import KeyedVectors
from sklearn.neural_network import MLPClassifier

word_vectors = KeyedVectors.load_word2vec_format('/Users/evelin.amorim/Documents/UFMG/word2vec/vectors-pt-toy.bin', binary=True)  # C binary format

# lendo os dados

dir_data = '/Users/evelin.amorim/Dropbox/UFMG/Seminarios2018/dados/train_news/'

y = []
npos = 5 # numero de documentos que sao esportes
doc_dim = 10
word_dim = 5


data_files = os.listdir(dir_data)

ninstances = len(data_files)
X = np.zeros(shape = (ninstances, word_dim * doc_dim))

ndoc = 0
for f in data_files:
    full_name = os.path.join(dir_data, f)
    tok_lst = word_tokenize(open(full_name, 'r').read(), language='portuguese')[:doc_dim]

    doc = None
    for i in range(doc_dim):
        try:
            if i != 0:
                doc = np.hstack((doc, word_vectors[tok_lst[i]]))
            else:
                doc = np.array(word_vectors[tok_lst[i]])
        except KeyError: # a palavra nao existe no modelo
            if i != 0:
                doc = np.hstack((doc, np.zeros(shape = (word_dim, ))))
            else:
                doc = np.zeros(shape = (word_dim, ))


    X[ndoc, :] = doc

    if len(y)  < npos:
       y.append(1)
    else:
       y.append(0)

    ndoc = ndoc + 1

y = np.array(y)


clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(X, y)

test_data = '/Users/evelin.amorim/Dropbox/UFMG/Seminarios2018/dados/test.txt'

tok_lst = word_tokenize(open(test_data, 'r').read(), language='portuguese')[:doc_dim]


X_test = np.zeros(shape = (1, word_dim * doc_dim))
doc = None
for i in range(doc_dim):
    try:
        if i != 0:
            doc = np.hstack((doc, word_vectors[tok_lst[i]]))
        else:
            doc = np.array(word_vectors[tok_lst[i]])
    except KeyError: # a palavra nao existe no modelo
        if i != 0:
            doc = np.hstack((doc, np.zeros(shape = word_dim)))
        else:
            doc = np.zeros(shape = word_dim)

X_test[0,:] = doc

print(clf.predict(X_test))
