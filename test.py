import numpy as np
import nltk
from nltk.stem.porter import *
stemmer = PorterStemmer()




def load(filename):
    listeDoc = list()
    with open(filename, "r") as f:
        doc = list()
        for ligne in f:
            #print "l : ",len(ligne)," ",ligne
            if len(ligne) < 2: # fin de doc
                listeDoc.append(doc)
                doc = list()
                continue
            mots = ligne.split(" ")
            doc.append((mots[0],mots[1]))
    return listeDoc

def createDict(alldocs):
    dico = {}
    for doc in alldocs:
        for mot in doc:
            print(stemmer.stem(mot[0]))
            dico[stemmer.stem(mot[0])]=mot[1]
    return dico

def evalPerf(dico,docsTest):
    occClass = {}
    for value in dico.values():
        occClass[value]=occClass.get(value,0)+1
    argmax = max(occClass.items(),key=lambda x:x[1])
    compteur=0
    motMalClasse=[]
    taille_totale=0
    for doc in docsTest:
        for mot in doc:
            motNew=stemmer.stem(mot[0])
            if(motNew not in dico):
                if(argmax[0] == mot[1]):#renvoi la classe majoritaire
                    compteur+=1
                else:
                    motMalClasse.append(motNew)
            elif(dico[motNew] == mot[1]):
                compteur+=1
            else:
                motMalClasse.append(motNew)
            taille_totale+=1
    print(compteur)
    return compteur/taille_totale,motMalClasse
if __name__ == '__main__' :
    # =============== chargement ============
    filename = "data/wapiti/chtrain.txt" # a modifier
    filenameT = "data/wapiti/chtest.txt" # a modifier

    alldocs = load(filename)
    alldocsT = load(filenameT)

    dicoTrain = createDict(alldocs)

    acc,motMalClasse = evalPerf(dicoTrain,alldocsT)
    print(acc)
    print(len(alldocs)," docs read")
    print(len(alldocsT)," docs (T) read")
