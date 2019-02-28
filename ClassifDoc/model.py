from sklearn.linear_model import SGDClassifier
import nltk
from nltk.stem.porter import *
stemmer = PorterStemmer()

def load_data(fileName):
    with open(fileName) as file :
        labels=[]
        data=[]
        all_lines=file.readlines()
        for line in all_lines:
            tab = line.split(" ")
            lab = tab[0][1:-1].split(":")[2]
            labels.append(lab)
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
        list=[]
        for mot in d:
            list.append(vocabDict[mot])
        new_data.append(list)
    return new_data

def writeTofile(tab):
    with open("result.txt","w") as res:
        for t in tab:
            res.write(t+"\n")

data,labels = load_data("corpus.tache1.learn.utf8")
print(data[0])
print(len(data))
vocab = createDict(data)
vocabDict = dict(zip(vocab,range(len(vocab))))
data_format = encodage_data(data,vocabDict)
#print(data_format)
clf = SGDClassifier(loss="perceptron", eta0=1e-4, learning_rate="constant", penalty=None,tol=1e-1,max_iter=10000,shuffle=True)
clf.fit(data_format,labels)
data_format_test = encodage_data(load_test("corpus.tache1.test.utf8"))
pred = clf.predict(data_format)
print(pred)
writeTofile(pred)
