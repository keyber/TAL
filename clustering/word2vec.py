from gensim.models import Word2Vec
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
            if(lab == "C"):
                labels.append(-1)
            else:
                labels.append(1)
            donnee = simplifieData(tab[1:])
            data.append(donnee)
        return data,labels

def simplifieData(listeMot):
    newList=[]
    for mot in listeMot:
        nouveauMot = mot.lower().replace("\n","").replace(",","").replace(".","").replace(":","")
        newList.append(stemmer.stem(nouveauMot))
    return newList

sentences,_ = load_data("corpus.tache1.learn.utf8")
model = Word2Vec(sentences, min_count=1)
print("fin entraienement model")
print(model.wv["citoyen"])


model = gensim.models.Word2Vec(iter=1)  # an empty model, no training yet
model.build_vocab(some_sentences)  # can be a non-repeatable, 1-pass generator
model.train(other_sentences)  # can be a non-repeatable, 1-pass generator

model = Word2Vec(sentences, min_count=10, size=200)  
