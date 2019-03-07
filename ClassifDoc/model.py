from sklearn.linear_model import Perceptron
import nltk
from nltk.stem.porter import *
import numpy as np


stemmer = PorterStemmer()

def load_data(fileName):
    with open(fileName) as file :
        labels=[]
        data=[]
        all_lines=file.readlines()
        for line in all_lines:
            tab = line.split(" ")
            lab = tab[0][1:-1].split(":")[2]
            if(lab == "C"):
                labels.append(-1)
            else:
                labels.append(1)
            donnee = simplifieData(tab[1:])
            data.append(donnee)
        return data,labels

def load_test(fileName):
    with open(fileName) as file :
        data=[]
        all_lines=file.readlines()
        for line in all_lines:
            tab = line.split(" ")
            donnee = simplifieData(tab[1:])
            data.append(donnee)
        return data

def simplifieData(listeMot):
    newList=[]
    for mot in listeMot:
        nouveauMot = mot.lower().replace("\n","").replace(",","").replace(".","").replace(":","")
        newList.append(stemmer.stem(nouveauMot))
    return newList
def createDict(data):
    voc = set()
    for phrase in data:
        for mot in phrase:
            voc.add(mot)
    return voc

def encodage_data(data,vocabDict):
    new_data=[]
    for d in data :
        list=np.zeros((len(vocabDict)))
        for mot in d:
            list[vocabDict[mot]]+=1
        new_data.append(list)
    return new_data

def writeTofile(tab):
    with open("result.txt","w") as res:
        for t in tab:
            if(t==1):
                res.write("C\n")
            else:
                res.write("M\n")

data,labels = load_data("corpus.tache1.learn.utf8")
print(data[0])
print(len(data))
vocab = createDict(data)
vocabDict = dict(zip(vocab,range(len(vocab))))
data_format = encodage_data(data,vocabDict)
#print(data_format)
clf =Perceptron(tol=1e-3, random_state=0)
print(len(data_format))
print(len(labels))
clf.fit(data_format,labels)
data_format_test = encodage_data(load_test("corpus.tache1.test.utf8"),vocabDict)
pred = clf.predict(data_format)
print(pred)
writeTofile(pred)
