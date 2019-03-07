from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import numpy as np
from sklearn.model_selection import StratifiedKFold


#Attention il rest \n et . dans le texte
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
            data.append(" ".join(tab[1:]))
        return data, labels


def load_test(fileName):
    with open(fileName) as file:
        data = []
        all_lines = file.readlines()
        for line in all_lines:
            tab = line.split(" ")
            data.append(" ".join(tab[1:]))
        return data


def writeTofile(tab):
    with open("result.txt", "w") as res:
        for t in tab:
            res.write(CM_inv[t] + "\n")


def main():
    x_train, y_train = load_data("corpus.tache1.learn.utf8")

    stop_words = set(stopwords.words("french"))
    print("stop words:", len(stop_words))
    print(stop_words)

    print("taille test:", len(x_train))

    n = len(x_train)
    # n = 100
    x_train = x_train[:n]
    y_train = y_train[:n]

    vectorizer = CountVectorizer(stop_words=stop_words)
    x_train = vectorizer.fit_transform(x_train)
    print(x_train[:10])
    y_train=np.array(y_train)
    print("taille du dictionnaire:", len(vectorizer.get_feature_names()))

    clf = SVC(kernel="linear", verbose=1, max_iter=1e3)

    skf = StratifiedKFold(n_splits=5)
    skf.get_n_splits(x_train, y_train)


    moyenne=0
    for train_index, test_index in skf.split(x_train, y_train):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_trainTmp, X_testTmp = x_train[train_index], x_train[test_index]
        y_trainTmp, y_testTmp = y_train[train_index], y_train[test_index]
        print(len(X_trainTmp))
        print(len(y_trainTmp))
        clf.fit(X_trainTmp,y_trainTmp)
        moyenne+=clf.score(X_testTmp,y_testTmp)
    print(moyenne/5)

    x_test = load_test("corpus.tache1.test.utf8")

    x_test = vectorizer.transform(x_test)

    y_pred = clf.predict(x_test)

    writeTofile(y_pred)

if __name__ == '__main__':
    main()
