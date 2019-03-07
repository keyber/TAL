from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import numpy as np


stemmer = SnowballStemmer("french")
stop_words = stopwords.words("french")
print("stop words:", len(stop_words))

CM = {'C':1, 'M':-1}
CM_inv = {1:'C', -1:'M'}

def load_data(fileName):
    with open(fileName) as file:
        labels = []
        data = []
        all_lines = file.readlines()
        for line in all_lines:
            tab = line.split(" ")
            lab = tab[0][1:-1].split(":")[2]
            labels.append(CM[lab])
            phrase = simplifie_phrase(tab[1:])
            data.append(" ".join(phrase))
        return data, labels


def load_test(fileName):
    with open(fileName) as file:
        data = []
        all_lines = file.readlines()
        for line in all_lines:
            tab = line.split(" ")
            phrase = simplifie_phrase(tab[1:])
            data.append(" ".join(phrase))
        return data


def simplifie_phrase(p):
    mots = [s.lower() for s in p]
    return [stemmer.stem(s) for s in mots if s not in stop_words]

    
def writeTofile(tab):
    with open("result.txt", "w") as res:
        for t in tab:
            res.write(CM_inv[t] + "\n")


def main():
    x_train, y_train = load_data("corpus.tache1.learn.utf8")
    
    print("taille train:", len(x_train))
    
    # n = len(x_train)
    n = 100
    x_train = x_train[:n]
    y_train = y_train[:n]
    
    vectorizer = CountVectorizer()
    x_train = vectorizer.fit_transform(x_train)
    print("taille du dictionnaire:", len(vectorizer.get_feature_names()))
    
    print("fit en cours\n")
    clf = SVC(kernel="linear", verbose=1, max_iter=1e3)
    clf.fit(x_train, y_train)
    
    
    print("génération réponse")
    x_test = load_test("corpus.tache1.test.utf8")
    
    x_test = vectorizer.transform(x_test)
    
    y_pred = clf.predict(x_test)
    
    writeTofile(y_pred)

if __name__ == '__main__':
    main()