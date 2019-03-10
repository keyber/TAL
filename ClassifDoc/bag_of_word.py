from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pickle
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import random


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
        """enlève les stopwords avant et après stemming
        sépare su apostrophe"""
        mots = " ".join([s.lower() for s in p])
        import re
        mots = re.sub("\d|\.|,|:|;|!|\?|\"|'", " ", mots)
        mots = mots.split()
        mots = [cls.stemmer.stem(s) for s in mots if s not in cls.stop_words]
        mots = [s for s in mots if s not in cls.stop_words]
        return mots


def writeTofile(tab):
    with open("result.txt", "w") as res:
        for t in tab:
            res.write(CM_inv[t] + "\n")


def load_pickled_train():
    try:
        pickle_in = open("data/x_train", "rb")
        x_train = pickle.load(pickle_in)
        pickle_in.close()
        
        pickle_in = open("data/y_train", "rb")
        y_train = pickle.load(pickle_in)
        pickle_in.close()
        
        print("lecture train effectuée")
        
    except FileNotFoundError:
        print("lecture train échouée")
        x_train, y_train = Parser.load_data("corpus.tache1.learn.utf8")
        
        pickle_out = open("data/x_train", "wb")
        pickle.dump(x_train, pickle_out)
        pickle_out.close()
        
        pickle_out = open("data/y_train", "wb")
        pickle.dump(y_train, pickle_out)
        pickle_out.close()
        print("train écrit")
    
    return x_train, y_train


def load_pickled_test():
    try:
        pickle_in = open("data/x_test", "rb")
        x_test = pickle.load(pickle_in)
        pickle_in.close()
        
        print("lecture test effectuée")
        
    except FileNotFoundError:
        print("lecture test échouée")
        x_test = Parser.load_test("data/corpus.tache1.test.utf8")
        
        pickle_out = open("data/x_test", "wb")
        pickle.dump(x_test, pickle_out)
        pickle_out.close()
        
        print("test écrit")
    
    return x_test

def plot_coefficients(classifier, feature_names, top_features=20):
    coef = classifier.coef_.ravel()
    ind = np.argsort(coef)
    top_positive_coefficients = ind[-top_features:]
    top_negative_coefficients = ind[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ["red" if c < 0 else "blue" for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha="right")
    plt.show()


def cross_val(x, y, p_mod, p_vec):
    clf = LinearSVC(**p_mod)
    vectorizer = CountVectorizer(**p_vec)
    skf = StratifiedKFold(n_splits=3)
    
    score_f1 = []
    taille_dict = []
    for train_index, test_index in skf.split(x, y):
        fold_x_train, fold_x_test = x[train_index], x[test_index]
        fold_y_train, fold_y_test = y[train_index], y[test_index]
        
        vectorizer.fit(fold_x_train)
        
        # print(len(vectorizer.get_feature_names()))
        # import collections
        # buf = [set(x.split()) for x in fold_x_train]
        # mots = []
        # for b in buf:
        #     mots+=b
        # c = collections.Counter(mots)
        # print(min([c[m] for m in vectorizer.get_feature_names() if m in c]))
        # print(max([c[m] for m in vectorizer.get_feature_names() if m in c]))
        
        taille_dict.append(len(vectorizer.get_feature_names()))
        
        fold_x_train = vectorizer.transform(fold_x_train)
        fold_x_test = vectorizer.transform(fold_x_test)
        
        clf.fit(fold_x_train, fold_y_train)
        
        y_pred = clf.predict(fold_x_test)
        rapport = classification_report(fold_y_test, y_pred, output_dict=True)
        
        score_f1.append(rapport['-1']["f1-score"])
        # moyenne += clf.score(fold_x_test, fold_y_test)
    
    return (np.mean(score_f1), np.std(score_f1)), (np.mean(taille_dict), np.std(taille_dict))


# def quali():
#     print(WordCloud(SVM.poids associés aux mots))

def optimize_cross_val(x_train, y_train):
    import itertools
    print("cross validation en cours")
    
    max_iter = [1e5]
    class_weight = ["balanced"]
    c = [2 ** (-3-i) for i in range(6)]
    
    # g = [2 ** (-i) for i in range(6)]
    # kernel = ["linear", "poly", "rbf"]
    # param_name = ["max_iter", "kernel", "C", "gamma"]
    params_model = [max_iter, class_weight, c]
    params_model_name = ["max_iter", "class_weight", "C"]
    
    
    ngram_range = [(1,1), (1,2), (2,2)]
    max_df = [1.0]#[.01, .02, .04, .08, 0.16, 10.0]#np.linspace(.01, .2, 5)
    min_df = range(0, 1, 5)
    
    params_vec = [ngram_range, max_df, min_df]
    params_vec_name = ["ngram_range", "max_df", 'min_df']
    
    p_max_mod, p_max_vec = None, None
    val_max, std_max = float("-inf"), None
    list_score = [[], []]
    list_taille = [[], []]
    for p_mod in itertools.product(*params_model):
        p_mod = {name: val for name, val in zip(params_model_name, p_mod)}
        
        for p_vec in itertools.product(*params_vec):
            p_vec = {name: val for name, val in zip(params_vec_name, p_vec)}
            
            (val, val_std), (taille, taille_std) = cross_val(x_train, y_train, p_mod, p_vec)
            
            aff = [str(name) + ":" + str(p_mod[name]) for name in params_model_name]
            aff += [str(name) + ":" + str(p_vec[name]) for name in params_vec_name]
            list_score[0].append(val)
            list_score[1].append(val_std)
            list_taille[0].append(taille)
            list_taille[1].append(taille_std)
            print(aff, val, taille)
            
            if val > val_max:
                val_max = val
                std_max = val_std
                p_max_vec = p_vec
                p_max_mod = p_mod
    list_score = np.array(list_score)
    list_taille = np.array(list_taille)
    ax1 = plt.gca()
    ax1.errorbar(c, list_score[0], 2*list_score[1], capsize=5)
    #ax2 = ax1.twinx()
    #ax2.errorbar(c, list_taille[0], 2*list_taille[1], color='orange', capsize=5)
    plt.show()
    print("cross val, optimal trouvé :")
    print(p_max_mod, p_max_vec, val_max, std_max)
    return p_max_mod, p_max_vec, val_max, std_max


def main():
    x_train, y_train = load_pickled_train()
    
    # import collections
    # buf = [set(x.split()) for x in x_train]
    # mots = []
    # for b in buf:
    #     mots += b
    # c = collections.Counter(mots)
    #
    # print("nb doc:", len(x_train))
    # print("nb mot tot:", sum(c.values()))
    # print("nb mot diff:", len(c))
    # plt.hist(np.log2(list(c.values())),  bins=20)
    # plt.show()
    
    indices = list(range(len(x_train)))
    random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]
    
    # n = len(x_train)
    n = 10000
    
    print("taille train:", n)
    
    p_max_model, p_max_vectorizer, _, _ = optimize_cross_val(x_train[:n], y_train[:n])

    #input("génération réponse ?")
    vectorizer = CountVectorizer(**p_max_vectorizer)
    vectorizer.fit(x_train)
    print("taille du dictionnaire:", len(vectorizer.get_feature_names()))

    print("fit en cours\n")
    p_max_vectorizer["max_iter"] = 1e3
    clf = LinearSVC(**p_max_model)
    x_train_vec = vectorizer.transform(x_train)
    clf.fit(x_train_vec, y_train)
    
    plot_coefficients(clf, vectorizer.get_feature_names())
    
    x_test = load_pickled_test()
    
    x_test = vectorizer.transform(x_test)
    
    y_pred = clf.predict(x_test)
    
    writeTofile(y_pred)


if __name__ == '__main__':
    main()
