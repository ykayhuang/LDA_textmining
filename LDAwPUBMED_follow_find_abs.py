# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 22:02:47 2018

@author: YK Huang
"""
#followed

#enter selection key
    sk=(5,5) #have to be a list, can be duplicates
    a6=list(r22)
    r63 = pd.DataFrame(columns = a6.append("TOPIC#"))
    
for i in sk:
    r61=r22.sort_values(by=[i], ascending=[0])
    r61 = r61.reset_index(drop=True)
    r62 = r61[r61[i]> 0.1]
    r62["TOPIC#"]=i
     # Try to append temporary DF to master DF
    r63 = r63.append(r62,ignore_index=True)
    
##remove duplicates
r65=r63.drop_duplicates(subset=['PMID','AB'], keep='first')
    
r65.to_csv( 'a5.csv' )