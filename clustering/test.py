from bs4 import BeautifulSoup
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import pyLDAvis
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pyLDAvis.sklearn

file = open("ap/ap.txt")
soup = BeautifulSoup(file, 'html.parser')

txt = soup.find_all("text")

txt = [str(t).replace("<text>","").replace("</text>","") for t in txt ]

vectorizer = CountVectorizer(strip_accents = 'unicode',
                                stop_words = 'english',
                                lowercase = True,
                                max_df = 0.5,
                                min_df = 10)
X = vectorizer.fit_transform(txt)

#changer le max_iter pour de meilleurs performances.
clf = LDA(n_components=20,max_iter=10)
clf.fit(X)

print(clf.components_)
print(clf.components_.shape)

parametre_classe = clf.components_

#arguments: 'doc_topic_dists', 'doc_lengths', 'vocab', and 'term_frequency'
print(len(parametre_classe))


#Visualisation
"""
tfidf_vectorizer = TfidfVectorizer(**vectorizer.get_params())
dtm_tfidf = tfidf_vectorizer.fit_transform(txt)

movies_vis_data = pyLDAvis.sklearn.prepare(clf, X, vectorizer)

pyLDAvis.show(movies_vis_data)
for k in range(20):
    p_c = parametre_classe[k]
    indices_mots = p_c.argsort()[-10:]
    mots=[]
    for i in indices_mots:
        tmp=[]
        tmp.append(vectorizer.get_feature_names()[i])
        tmp.append(p_c[i])
        mots.append(tmp)
    print(mots)
    plt.bar([m[0] for m in mots], [m[1] for m in mots], color="blue")
    plt.show()
"""
while(1):
    doc_a_test = input("taper votre document :")
    x_a_test= vectorizer.transform([doc_a_test])
    print(clf.transform(x_a_test))
