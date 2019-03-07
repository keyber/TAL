from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pickle
from sklearn.metrics import classification_report

CM = {'C': 1, 'M': -1}
CM_inv = {1: 'C', -1: 'M'}

class Parser:
    from nltk.corpus import stopwords
    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("french")
    stop_words = stopwords.words("french")
    
    @classmethod
    def load_data(cls, fileName):
        with open(fileName) as file:
            labels = []
            data = []
            all_lines = file.readlines()
            for line in all_lines:
                tab = line.split(" ")
                lab = tab[0][1:-1].split(":")[2]
                labels.append(CM[lab])
                phrase = cls.simplifie_phrase(tab[1:])
                data.append(" ".join(phrase))
            return np.array(data), np.array(labels)
    
    @classmethod
    def load_test(cls, fileName):
        with open(fileName) as file:
            data = []
            all_lines = file.readlines()
            for line in all_lines:
                tab = line.split(" ")
                phrase = cls.simplifie_phrase(tab[1:])
                data.append(" ".join(phrase))
            return np.array(data)
    
    
    @classmethod
    def simplifie_phrase(cls, p):
        mots = [s.lower() for s in p]
        return [cls.stemmer.stem(s) for s in mots if s not in cls.stop_words]


def writeTofile(tab):
    with open("result.txt", "w") as res:
        for t in tab:
            res.write(CM_inv[t] + "\n")


def load_pickled_train():
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
        x_train, y_train = Parser.load_data("corpus.tache1.learn.utf8")
        
        pickle_out = open("x_train", "wb")
        pickle.dump(x_train, pickle_out)
        pickle_out.close()
        
        pickle_out = open("y_train", "wb")
        pickle.dump(y_train, pickle_out)
        pickle_out.close()
        print("train écrit")
    
    return x_train, y_train


def load_pickled_test():
    try:
        pickle_in = open("x_test", "rb")
        x_test = pickle.load(pickle_in)
        pickle_in.close()
        
        print("lecture test effectuée")
        
    except FileNotFoundError:
        print("lecture test échouée")
        x_test = Parser.load_test("corpus.tache1.test.utf8")
        
        pickle_out = open("x_test", "wb")
        pickle.dump(x_test, pickle_out)
        pickle_out.close()
        
        print("test écrit")
    
    return x_test


def cross_val(vectorizer, x, y, param):
    clf = LinearSVC(**param)
    
    skf = StratifiedKFold(n_splits=5)
    
    moyenne_f1 = 0
    for train_index, test_index in skf.split(x, y):
        fold_x_train, fold_x_test = x[train_index], x[test_index]
        fold_y_train, fold_y_test = y[train_index], y[test_index]
        
        fold_x_train = vectorizer.transform(fold_x_train)
        fold_x_test = vectorizer.transform(fold_x_test)
        
        clf.fit(fold_x_train, fold_y_train)
        
        y_pred = clf.predict(fold_x_test)
        rapport = classification_report(fold_y_test, y_pred, output_dict=True)
        
        moyenne_f1 += rapport['-1']["f1-score"]
        # moyenne += clf.score(fold_x_test, fold_y_test)
    
    return moyenne_f1 / skf.get_n_splits(x, y)


# def F1():
#     pass
# def qte():
#     return F1()
# def quali():
#     print(WordCloud(SVM.poids associés aux mots))

def optimize_cross_val(x_train, y_train):
    import itertools
    print("cross validation en cours")
    
    max_iter = [1e5]
    class_weight = ["balanced"]
    c = [2 ** (-3-i) for i in range(4)]
    # g = [2 ** (-i) for i in range(6)]
    # kernel = ["linear", "poly", "rbf"]
    # param_name = ["max_iter", "kernel", "C", "gamma"]
    params_model = [max_iter, class_weight, c]
    params_model_name = ["max_iter", "class_weight", "C"]
    
    
    ngram_range = [(1,1)]#, (1,2), (2,2)]
    max_df = np.linspace(.5, .8, 8)
    min_df = np.linspace(.0, .0, 1)
    
    params_vec = [ngram_range, max_df, min_df]
    params_vec_name = ["ngram_range", "max_df", 'min_df']
    
    p_max_mod, p_max_vec, val_max = None, None, float("-inf")
    for p_mod in itertools.product(*params_model):
        p_mod = {name: val for name, val in zip(params_model_name, p_mod)}
        for p_vec in itertools.product(*params_vec):
            p_vec = {name: val for name, val in zip(params_vec_name, p_vec)}
            
            vectorizer = CountVectorizer(**p_vec)
            vectorizer.fit(x_train)
            
            val = cross_val(vectorizer, x_train, y_train, p_mod)
            
            aff = [str(name) + ":" + str(p_mod[name]) for name in params_model_name]
            aff += [str(name) + ":" + str(p_vec[name]) for name in params_vec_name]
            print(aff, val)
            
            if val > val_max:
                val_max = val
                p_max_vec = p_vec
                p_max_mod = p_mod
                
        print(p_max_vec)
    
    print("cross val, optimal trouvé :")
    print(p_max_mod, p_max_vec, val_max)
    return p_max_mod, params_vec, val_max


def main():
    x_train, y_train = load_pickled_train()
    
    # n = len(x_train)
    n = 10000
    
    print("taille train:", n)
    
    p_max_vectorizer, p_max_model, val_max = optimize_cross_val(x_train[:n], y_train[:n])
    
    input("génération réponse ?")
    vectorizer = CountVectorizer(p_max_vectorizer)
    vectorizer.fit(x_train)
    print("taille du dictionnaire:", len(vectorizer.get_feature_names()))

    print("fit en cours\n")
    p_max_vectorizer["max_iter"] = 1e3
    clf = LinearSVC(**p_max_vectorizer)
    x_train_vec = vectorizer.fit_transform(x_train)
    clf.fit(x_train_vec, y_train)
    
    x_test = load_pickled_test()
    
    x_test = vectorizer.transform(x_test)
    
    y_pred = clf.predict(x_test)
    
    writeTofile(y_pred)


if __name__ == '__main__':
    main()
