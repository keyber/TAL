import numpy as np
import collections


def writeTofile(tab):
    with open("nouveau_result_sentiment_annalysis.txt", "w") as res:
        for t in tab:
            res.write(t + "\n")


file = open("result_sentiment_annalysis.txt")
lines = file.readlines()
new_line=[]
for l in lines:
    new_line.append(l.replace("\n",""))


print(collections.Counter(new_line))

moyenne=1
for k in range(moyenne,len(new_line)-moyenne):
    co = collections.Counter(new_line[k-moyenne:k+moyenne+1])
    """
    if(new_line[k]=='C' and co['M']==2):
        new_line[k]='M'
    """
    if(new_line[k]=='M' and co['C']==2):
        new_line[k]='C'


for k in range(0,len(new_line)-5):
    if(new_line[k]=='C' and new_line[k+2]=='C'):
        new_line[k+1]='C'
    if(new_line[k]=='C' and new_line[k+3]=='C'):
        new_line[k+1]='C'
        new_line[k+2]='C'
    if(new_line[k]=='C' and new_line[k+4]=='C'):
        new_line[k+1]='C'
        new_line[k+2]='C'
        new_line[k+3]='C'
    if(new_line[k]=='C' and new_line[k+5]=='C'):
        new_line[k+1]='C'
        new_line[k+2]='C'
        new_line[k+3]='C'
        new_line[k+4]='C'

print(collections.Counter(new_line))
writeTofile(new_line)
