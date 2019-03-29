import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from sklearn.metrics import classification_report
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


def simplifie_phrase(p):
    stemmer = SnowballStemmer("english")
    stop_words = stopwords.words("english")
    phrase = p.lower().split(" ")
    #phraseDecomp = [stemmer.stem(s) for s in phrase if s not in stop_words]
    phraseDecomp = [s for s in phrase]
    return ' '.join(word for word in phraseDecomp)


def loadData(fileList):
    data=[]
    for fileName in fileList:
        file = open(fileName,"r")
        all_lines=file.readlines()
        all_lines = ' '.join(word for word in all_lines)
        data.append(simplifie_phrase(all_lines))
    return data

def loadData_test(fileName):
    data=[]
    file = open(fileName,"r")
    for document in file:
        data.append(simplifie_phrase(document))
    return data


def writeTofile(tab):
    with open("result_sentiment_annalysis.txt", "w") as res:
        for t in tab:
            res.write(t + "\n")

def convertOutput(pred):
    new_pred=[]
    for p in pred:
        if(p==-1):
            new_pred.append('C')
        else:
            new_pred.append('M')
    return new_pred



listFilePos = getListOfFiles("AFDmovies/movies1000/pos/")
listFileNeg = getListOfFiles("AFDmovies/movies1000/neg/")


dataPos = loadData(listFilePos)
print(len(dataPos))
print("je viens de load les data pos")
dataNeg = loadData(listFilePos)
print(len(dataNeg))
print("je viens de load les data neg")
labelsPos = [1 for _ in range(len(dataPos))]
labelsNeg = [-1 for _ in range(len(dataNeg))]


############A decommenter############
data = dataPos+dataNeg
labels = labelsPos+labelsNeg
##################################
indiceShuffle = np.random.permutation(np.array([i for i in range(len(data))]))

new_data=[]
new_labels=[]
for indice in indiceShuffle:
    new_data.append(data[indice])
    new_labels.append(labels[indice])
data = new_data
labels = new_labels


vectorizer = CountVectorizer(max_df=0.99,min_df=0.01)
vectorizer.fit(data)
data = vectorizer.transform(data)

"""
#On créer un ensemble de validation pour faire des test
data = data[:-100]
labels = labels[:-100]

dataVal = data[-100:]
labels_val = labels[-100:]
"""

#clf = LinearSVC(verbose=1,max_iter=1e8,tol=1e-1)
#clf = linear_model.SGDClassifier(max_iter=1e4, tol=-1e3,verbose=1)
clf = KNeighborsClassifier(n_neighbors=3)
#clf = tree.DecisionTreeClassifier()
#clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial',max_iter=1e6,tol=1e-18, verbose=1)
"""
clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
"""
print("je commence le fit")
clf.fit(data,labels)

"""
##On test sur l'ensemble de validation
y_pred_val = clf.predict(dataVal)
rapport = classification_report(labels_val, y_pred_val, output_dict=True)
print(rapport)
"""


#######On genere les labels sur le jeux de test et on ecrit dans un fichier############
donnees_test = loadData_test("testSentiment.txt")
donnees_test =  vectorizer.transform(donnees_test)
print(donnees_test[0])

y_pred = clf.predict(donnees_test)
print(y_pred)
print(len(y_pred))
y_pred = convertOutput(y_pred)

writeTofile(y_pred)
