from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pickle

CM = {'C': 1, 'M': -1}
CM_inv = {1: 'C', -1: 'M'}

class Parser:
    from nltk.corpus import stopwords
    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("french")
    stop_words = stopwords.words("french")


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

def main():
    x_train, y_train = load_pickle()

    print("taille train:", len(x_train))

    # n = len(x_train)
    n = 100
    x_train = np.array(x_train[:n])
    y_train = np.array(y_train[:n])

    vectorizer = CountVectorizer()
    vectorizer.fit(x_train)
    #x_train = vectorizer.fit_transform(x_train)
    print("taille du dictionnaire:", len(vectorizer.get_feature_names()))

    import itertools
    print("cross valisation en cours")
    max_iter = [1e3]
    c = [2 ** (2 - i) for i in range(6)]
    g = [2 ** (-i) for i in range(6)]
    kernel = ["linear", "poly", "rbf"]

    param_name = ["max_iter", "kernel", "C", "gamma"]
    p_max, val_max = None, float("-inf")
    for p in itertools.product(max_iter, kernel, c, g):
        p = {name: val for name, val in zip(param_name, p)}
        val = cross_val(vectorizer, x_train, y_train, p)
        print(p, val)
        if val > val_max:
            val_max = val
            p_max = p

    print("optimal trouvé")
    print(p_max)
    print(val_max)

    print("fit en cours\n")
    p_max["max_iter"] = 1e6
    clf = SVC(**p_max)

    print("génération réponse")

    x_test = load_test("corpus.tache1.test.utf8")
    
    x_test = vectorizer.transform(x_test)

    y_pred = clf.predict(x_test)

    writeTofile(y_pred)


if __name__ == '__main__':
    main()
