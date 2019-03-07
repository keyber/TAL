from sklearn.linear_model import Perceptron
import nltk
import numpy as np
from nltk.stem.porter import *
import numpy as np


stemmer = PorterStemmer()


def load_data(fileName):
    with open(fileName) as file:
        labels = []
        data = []
        all_lines = file.readlines()
        for line in all_lines:
            tab = line.split(" ")
            lab = tab[0][1:-1].split(":")[2]
            labels.append(1 if lab == 'C' else -1)
            donnee = simplifieData(tab[1:])
            data.append(donnee)
        return data, labels


def load_test(fileName):
    with open(fileName) as file:
        data = []
        all_lines = file.readlines()
        for line in all_lines:
            tab = line.split(" ")
            donnee = simplifieData(tab[1:])
            data.append(donnee)
        return data


def simplifieData(listeMot):
    newList = []
    for mot in listeMot:
        nouveauMot = mot.lower().replace("\n", "").replace(",", "").replace(".", "").replace(":", "")
        #newList.append(nouveauMot)
        newList.append(stemmer.stem(nouveauMot))
    return newList


def createDict(data):
    voc = set()
    for phrase in data:
        for mot in phrase:
            voc.add(mot)
    return voc


def encodage_data(data, vocabDict):
    new_data = []
    for d in data:
        l = np.zeros((len(vocabDict)))
        for mot in d:
            if mot in vocabDict:
                l[vocabDict[mot]] += 1
        new_data.append(l)
    return new_data


def writeTofile(tab):
    with open("result.txt", "w") as res:
        for t in tab:
            res.write(t + "\n")


def main():
    x_train, y_train = load_data("corpus.tache1.learn.utf8")
    print(len(x_train))
    x_train = x_train
    y_train = y_train

    vocab = createDict(x_train)
    vocabDict = dict(zip(vocab, range(len(vocab))))

    x_train_vectorized = encodage_data(x_train, vocabDict)
    clf = Perceptron(tol=1e-1, max_iter=100, shuffle=True)
    clf.fit(x_train_vectorized, y_train)

    x_test_vectorized = encodage_data(load_test("corpus.tache1.test.utf8"), vocabDict)
    pred = clf.predict(x_test_vectorized)

    pred = np.where(pred == -1, 'C', 'M')
    writeTofile(pred)

if __name__ == '__main__':
    main()
