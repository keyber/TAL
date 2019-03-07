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


def cross_val(vectorizer, x, y, param):
    clf = SVC(**param)
    
    skf = StratifiedKFold(n_splits=5)
    
    moyenne = 0
    for train_index, test_index in skf.split(x, y):
        fold_x_train, fold_x_test = x[train_index], x[test_index]
        fold_y_train, fold_y_test = y[train_index], y[test_index]
        
        fold_x_train = vectorizer.transform(fold_x_train)
        fold_x_test = vectorizer.transform(fold_x_test)
        
        clf.fit(fold_x_train, fold_y_train)
        
        moyenne += clf.score(fold_x_test, fold_y_test)
    
    return moyenne / skf.get_n_splits(x, y)


def load_pickle():
    try:
        pickle_in = open("x_train", "rb")
        x_train = pickle.load(pickle_in)
        pickle_in.close()
        
        pickle_in = open("y_train", "rb")
        y_train = pickle.load(pickle_in)
        pickle_in.close()
        
        print("lecture train effectuée")
        
    except FileNotFoundError:
        print("lecture train échouée")
        x_train, y_train = load_data("corpus.tache1.learn.utf8")
        
        pickle_out = open("x_train", "wb")
        pickle.dump(x_train, pickle_out)
        pickle_out.close()
        
        pickle_out = open("y_train", "wb")
        pickle.dump(y_train, pickle_out)
        pickle_out.close()
        print("train écrit")
    
    return x_train, y_train

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
