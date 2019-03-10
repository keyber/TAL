from sklearn.linear_model import Perceptron
import nltk
from nltk.stem.porter import *
import numpy as np
from gensim.models import Word2Vec
from sklearn.svm import LinearSVC


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
    #sentences,_ = load_data("corpus.tache1.learn.utf8")
    model = Word2Vec(data, min_count=1)
    print("fin entraienement model")
    new_data=[]
    for d in data:
        tmp=[]
        for mot in d:
            if(len(tmp)==0):
                tmp=np.array(model.wv[mot])
            else:
                tmp+=np.array(model.wv[mot])
        tmp=tmp/len(d)
        #print("tmp")
        #print(tmp)
        new_data.append(tmp)
    return new_data

def writeTofile(tab):
    print("jecrit dans result")
    with open("result.txt","w") as res:
        for t in tab:
            if(t==1):
                res.write("C\n")
            else:
                res.write("M\n")

data,labels = load_data("corpus.tache1.learn.utf8")

argC1 = np.argwhere(labels==1).reshape((-1))
argC2 = np.argwhere(labels==2).reshape((-1))
for _ in range(5):
    self.labels = np.concatenate((self.labels,self.labels[argC2]))
    self.data = np.concatenate( ( self.data,self.data[argC2] ) )



print(data[0])
print(len(data))
vocab = createDict(data)
vocabDict = dict(zip(vocab,range(len(vocab))))
data_format = encodage_data(data,vocabDict)
#print(data_format)
clf =Perceptron(random_state=0)
#clf = LinearSVC(verbose=1)
print(data_format[0])
print(labels[0])
print(len(data_format))
print(len(data_format[0]))
print(len(labels))
print("je commence a entrer mon modele")
clf.fit(data_format,labels)
data_format_test = encodage_data(load_test("corpus.tache1.test.utf8"),vocabDict)
pred = clf.predict(data_format_test)
print(pred)
writeTofile(pred)
