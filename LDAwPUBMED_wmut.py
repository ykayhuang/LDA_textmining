# -*- coding: utf-8 -*-
"""
Created on Thu Jan 04 11:41:31 2018

@author: YK Huang
"""

from Bio import Entrez
Entrez.email = "yhuan25@uic.edu"     

findkey="air no2 health"

handle = Entrez.egquery(term=findkey)
record = Entrez.read(handle)

for row in record["eGQueryResult"]:
  if row["DbName"]=="pubmed":
     print(row["Count"])
     nofref=row["Count"]

#in later
from Bio import Medline
import pandas as pd
### manually broke into different yeay
y11,y12=2013,2030
y21,y22=2009,2013
y31,y32=2005,2008
y41,y42=2000,2004
####D-1
handle = Entrez.esearch(db="pubmed", term=findkey, retmax=nofref,
                        mindate=str(y11),maxdate=str(y12) )
record = Entrez.read(handle)
handle.close()
idlist = record["IdList"]

handle = Entrez.efetch(db="pubmed", id=idlist, rettype="medline",retmode="text")
records = Medline.parse(handle)

records = list(records)

w, h = 5, len(records);
maa = [[None for x in range(w)] for y in range(h)] 

for i in range(0, len(records)):
    maa[i][0]=records[i].get("PMID","?")
    maa[i][1]=records[i].get("TI","?")
    maa[i][2]=records[i].get("AB","?")
    maa[i][3]=records[i].get("DP")
    maa[i][4]=records[i].get("PT","?")
    
ma1=pd.DataFrame(maa)
ma1.columns = ['PMID', 'TI','AB','DP','PT']
#### D-1

#### D-2
handle = Entrez.esearch(db="pubmed", term=findkey, retmax=nofref,
                        mindate=str(y21),maxdate=str(y22) )
record = Entrez.read(handle)
handle.close()
idlist = record["IdList"]

handle = Entrez.efetch(db="pubmed", id=idlist, rettype="medline",retmode="text")
records = Medline.parse(handle)

records = list(records)

w, h = 5, len(records);
maa = [[None for x in range(w)] for y in range(h)] 

for i in range(0, len(records)):
    maa[i][0]=records[i].get("PMID","?")
    maa[i][1]=records[i].get("TI","?")
    maa[i][2]=records[i].get("AB","?")
    maa[i][3]=records[i].get("DP")
    maa[i][4]=records[i].get("PT","?")
    
ma2=pd.DataFrame(maa)
ma2.columns = ['PMID', 'TI','AB','DP','PT']
#### D-1    

#### D-3
handle = Entrez.esearch(db="pubmed", term=findkey, retmax=nofref,
                        mindate=str(y31),maxdate=str(y32) )
record = Entrez.read(handle)
handle.close()
idlist = record["IdList"]

handle = Entrez.efetch(db="pubmed", id=idlist, rettype="medline",retmode="text")
records = Medline.parse(handle)

records = list(records)

w, h = 5, len(records);
maa = [[None for x in range(w)] for y in range(h)] 

for i in range(0, len(records)):
    maa[i][0]=records[i].get("PMID","?")
    maa[i][1]=records[i].get("TI","?")
    maa[i][2]=records[i].get("AB","?")
    maa[i][3]=records[i].get("DP")
    maa[i][4]=records[i].get("PT","?")
    
ma3=pd.DataFrame(maa)
ma3.columns = ['PMID', 'TI','AB','DP','PT']
#### D-3    

#### D-4
handle = Entrez.esearch(db="pubmed", term=findkey, retmax=nofref,
                        mindate=str(y41),maxdate=str(y42) )
record = Entrez.read(handle)
handle.close()
idlist = record["IdList"]

handle = Entrez.efetch(db="pubmed", id=idlist, rettype="medline",retmode="text")
records = Medline.parse(handle)

records = list(records)

w, h = 5, len(records);
maa = [[None for x in range(w)] for y in range(h)] 

for i in range(0, len(records)):
    maa[i][0]=records[i].get("PMID","?")
    maa[i][1]=records[i].get("TI","?")
    maa[i][2]=records[i].get("AB","?")
    maa[i][3]=records[i].get("DP")
    maa[i][4]=records[i].get("PT","?")
    
ma4=pd.DataFrame(maa)
ma4.columns = ['PMID', 'TI','AB','DP','PT']
#### D-4    
makk=ma1.append([ma2, ma3, ma4])
makk=makk.drop_duplicates(subset=['PMID','AB'], keep='first')
akk=makk["AB"]

##topic
#####another
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import numpy as np

documents=akk

tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

lda_model = LatentDirichletAllocation(n_topics=10, max_iter=10, random_state=0).fit(tf)

lda_W = lda_model.transform(tf) # returns document-topic matrix
lda_H = lda_model.components_ 

##adding
lwdf=pd.DataFrame(lda_W)
r22 = pd.concat([makk, lwdf], axis=1, join_axes=[makk.index])
##end of adding

def display_topics(H, W, feature_names, documents, no_top_words, no_top_documents):
    for topic_idx, topic in enumerate(H):
        print ("Topic #%d: " % topic_idx)
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_documents]
        for doc_index in top_doc_indices:
            print (documents[doc_index])


# specify number of top words and top documents
no_top_words = 10
no_top_documents = 0

display_topics(lda_H, lda_W, tf_feature_names, documents, no_top_words, no_top_documents)

#topic:
r22["PMID"].count()

for i in range(0, len(lwdf.columns)):
    r23=r22.sort_values(by=[i], ascending=[0])
    r23 = r23.reset_index(drop=True)
    print("Topic #",i)
    print("Likely num of references:     ",r23[r23[i] >= 0.1][i].count() ,
          "  ~",round(r23[r23[i] >= 0.1][i].count()/float(r22["PMID"].count()),3))
    print(r23["TI"][0])
    print(r23["TI"][1])
    print(r23["TI"][2])
    print(r23["TI"][3])
    print(r23["TI"][4])
    print(r23["TI"][5])
    print("")
    
    
    