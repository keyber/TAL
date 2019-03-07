import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import numpy as np


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
    phraseDecomp = [stemmer.stem(s) for s in phrase if s not in stop_words]
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
            new_pred.append('M')#C
        else:
            new_pred.append('C')#M
    return new_pred



listFilePos = getListOfFiles("movies1000/pos/")
listFileNeg = getListOfFiles("movies1000/neg/")


dataPos = loadData(listFilePos)
print(len(dataPos))
print("je viens de load les data pos")
dataNeg = loadData(listFilePos)
print("je viens de load les data neg")
labelsPos = [1 for _ in range(len(dataPos))]
labelsNeg = [-1 for _ in range(len(dataNeg))]


############A decommenter############
data = dataPos+dataNeg
labels = labelsPos+labelsNeg
##################################


vectorizer = CountVectorizer()
vectorizer.fit(data)
data = vectorizer.transform(data)

donnees_test = loadData_test("testSentiment.txt")
donnees_test =  vectorizer.transform(donnees_test)
print(donnees_test[0])

clf = LinearSVC(verbose=1,max_iter=1e6)
print("je commence le fit")
clf.fit(data,labels)


y_pred = clf.predict(donnees_test)
y_pred = convertOutput(y_pred)

writeTofile(y_pred)
