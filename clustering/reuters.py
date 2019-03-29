from bs4 import BeautifulSoup
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import pyLDAvis
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pyLDAvis.sklearn
from nltk.corpus import reuters
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

n_classes = 90
labels = reuters.categories()

def load_data(config={}):
    """
    Load the Reuters dataset.

    Returns
    -------
    Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    stop_words = stopwords.words("english")
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    mlb = MultiLabelBinarizer()

    documents = reuters.fileids()
    test = [d for d in documents if d.startswith('test/')]
    train = [d for d in documents if d.startswith('training/')]

    docs = {}
    docs['train'] = [reuters.raw(doc_id) for doc_id in train]
    docs['test'] = [reuters.raw(doc_id) for doc_id in test]
    xs = {'train': [], 'test': []}
    xs['train'] = vectorizer.fit_transform(docs['train']).toarray()
    xs['test'] = vectorizer.transform(docs['test']).toarray()
    ys = {'train': [], 'test': []}
    ys['train'] = mlb.fit_transform([reuters.categories(doc_id)
                                     for doc_id in train])
    ys['test'] = mlb.transform([reuters.categories(doc_id)
                                for doc_id in test])
    data = {'x_train': xs['train'], 'y_train': ys['train'],
            'x_test': xs['test'], 'y_test': ys['test'],
            'labels': globals()["labels"]}
    return data

d = load_data()
X=d["x_train"]

lab = d["y_train"]

#changer le max_iter pour de meilleurs performances.
clf = LDA(n_components=n_classes,max_iter=10,topic_word_prior=0.9)
clf.fit(X)


#calcul de pureté du premier cluster
pred_cluster1=[]
for k in range(len(X)):
    pred = np.argmax(clf.transform([X[k]]))
    if(pred==1):
        pred_cluster1.append(k)
lab_clust1=[]
for p in pred_cluster1:
    lab_clust1.append(np.argmax(lab[p]))
c = Counter(lab_clust1)
somme = sum(c.values())
classe_majoritaire = np.max(list(c.values()))
print("Pureté de la classe numero 1 : ")
print(classe_majoritaire/somme)
