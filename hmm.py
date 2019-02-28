import numpy as np
import matplotlib.pyplot as plt
# allx: liste de séquences d'observations
# allq: liste de séquences d'états
# N: nb états
# K: nb observation

def learnHMM(allx, allq, N, K, initTo1=True):
    if initTo1:
        eps = 1e-5
        A = np.ones((N,N))*eps
        B = np.ones((N,K))*eps
        Pi = np.ones(N)*eps
    else:
        A = np.zeros((N,N))
        B = np.zeros((N,K))
        Pi = np.zeros(N)
    for x,q in zip(allx,allq):
        Pi[int(q[0])] += 1
        for i in range(len(q)-1):
            A[int(q[i]),int(q[i+1])] += 1
            B[int(q[i]),int(x[i])] += 1
        B[int(q[-1]),int(x[-1])] += 1 # derniere transition
    A = A/np.maximum(A.sum(1).reshape(N,1),1) # normalisation
    B = B/np.maximum(B.sum(1).reshape(N,1),1) # normalisation
    Pi = Pi/Pi.sum()
    return Pi , A, B

def viterbi(x,Pi,A,B):
    T = len(x)
    N = len(Pi)
    logA = np.log(A)
    logB = np.log(B)
    logdelta = np.zeros((N,T))
    psi = np.zeros((N,T), dtype=int)
    S = np.zeros(T,dtype=int)
    logdelta[:,0] = np.log(Pi) + logB[:,x[0]]
    #forward
    for t in range(1,T):
        logdelta[:,t] = (logdelta[:,t-1].reshape(N,1) + logA).max(0) + logB[:,x[t]]
        psi[:,t] = (logdelta[:,t-1].reshape(N,1) + logA).argmax(0)
    # backward
    logp = logdelta[:,-1].max()
    S[T-1] = logdelta[:,-1].argmax()
    for i in range(2,T+1):
        S[T-i] = psi[S[T-i+1],T-i+1]
    return S, logp #, delta, psi


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

def decodeSortie(indList,ind2cle):
    new_list=[]
    for i in indList:
        new_list.append(ind2cle[i])
    return new_list

def evalHmm(Pi,A,B,allxT,allqT):
    compteur=0
    taille=0
    confMat = np.zeros((len(A),len(A)))
    for k in range(len(allxT)):
        S, _ = viterbi(allxT[k],Pi,A,B)
        for i in range(len(S)):
            if(S[i] == allqT[k][i]):
                compteur+=1
            confMat[allqT[k][i]-1][S[i]-1]+=1

            taille+=1
    print(compteur)
    confMat= [l/sum(l) for l in confMat]
    return compteur/taille,confMat

def affichage(A,posTag):
    filename="test"
    plt.figure()
    plt.imshow(A, interpolation='nearest')
    localLabs = posTag # liste des POS-TAG
    plt.yticks(range(len(localLabs)),localLabs) # affichage sur l'image
    plt.xticks(range(len(localLabs)),localLabs)
    if filename != None:
        plt.savefig(filename)


if __name__ == '__main__' :
    filename = "data/wapiti/chtrain.txt" # a modifier
    filenameT = "data/wapiti/chtest.txt" # a modifier

    alldocs = load(filename)
    alldocsT = load(filenameT)

    # alldocs etant issu du chargement des données

    buf = [[m for m,c in d ] for d in alldocs]
    mots = []
    [mots.extend(b) for b in buf]
    mots = np.unique(np.array(mots))
    nMots = len(mots)+1 # mot inconnu

    mots2ind = dict(zip(mots,range(len(mots))))
    mots2ind["UUUUUUUU"] = len(mots)

    buf2 = [[c for m,c in d ] for d in alldocs]
    cles = []
    [cles.extend(b) for b in buf2]
    cles = np.unique(np.array(cles))
    cles2ind = dict(zip(cles,range(len(cles))))

    ind2cle = dict(zip(range(len(cles)),cles))

    nCles = len(cles)

    print(nMots,nCles," in the dictionary")

    # mise en forme des données
    allx  = [[mots2ind[m] for m,c in d] for d in alldocs]
    allxT = [[mots2ind.get(m,len(mots)) for m,c in d] for d in alldocsT]

    allq  = [[cles2ind[c] for m,c in d] for d in alldocs]
    allqT = [[cles2ind.get(c,len(cles)) for m,c in d] for d in alldocsT]


    Pi,A,B = learnHMM(allx,allq,nCles,nMots)

    acc,confMat = evalHmm(Pi,A,B,allxT,allqT)
    print(acc)

    plt.figure()
    plt.imshow(confMat, interpolation='nearest')
    localLabs = cles # liste des POS-TAG
    plt.yticks(range(len(localLabs)),localLabs) # affichage sur l'image
    plt.xticks(range(len(localLabs)),localLabs)
    plt.show()
    X= allxT[0]
    S , logP = viterbi(X,Pi,A,B)
    print(decodeSortie(S,ind2cle))
    affichage(A,cles)
