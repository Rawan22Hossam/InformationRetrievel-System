import numpy as np
import math
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
# ------------------------First Part -----------------------------------
stopword = stopwords.words('english')
all_files = os.listdir("D:\ProjectIR\doc")
document_terms = []
docID = 1
# ------------------1)Read 10 files--------------------------------
print("#========================Read 10 files=======================================\n")

for files in all_files:
    with open(f'D:\ProjectIR\doc/{files}', 'r') as F:
        document = F.read()
        print('doc', docID, ":", document, '\n')
        docID += 1
    # ----------------2)Apply tokenization -----------------------------
    tokenized_Documents = word_tokenize(document)
    terms = []
    for word in tokenized_Documents:
        # -----------------3)Remove Stopwords--------------------------------
        if word not in stopword or word == "in" or word == "to" or word == "where":
            terms.append(word)
    document_terms.append(terms)
print("#========================Apply tokenization and stopwords====================\n")
print(document_terms)
print("\n")

# ----------------------------- Second Part-----------------------------
# ----------------------1) Positional index------------------------------
document_number = 1
positionalIndex = {}
for document in document_terms:
    for positional, term in enumerate(document):
        if term in positionalIndex:
            positionalIndex[term][0] = positionalIndex[term][0] + 1
            if document_number in positionalIndex[term][1]:
                positionalIndex[term][1][document_number].append[positional]

            else:
                positionalIndex[term][1][document_number] = [positional]

        else:
            positionalIndex[term] = []
            positionalIndex[term].append(1)
            positionalIndex[term].append({})
            positionalIndex[term][1][document_number] = [positional]

        document_number += 1

print("#===========================Positional Index===========================\n")
print('<', positionalIndex, '>')
print("\n")

# ----------------------2)Phrasel query -------------------------------------
print("#=======================Phrasel Query===================================\n")
query = input("please enter the query you wish to process \n")

phrasel_List = [[] for i in range(document_number)]

for word in query.split():
    for k in positionalIndex[word][1].keys():
        if word not in positionalIndex.keys():
            print("these terms not in the documents\n")
        else:
            for key in positionalIndex[word][1].keys():
                if phrasel_List[key] != []:
                    if phrasel_List[key-1][0] == positionalIndex[word][1][key][0]-1:
                        phrasel_List[key -
                                     1].append(positionalIndex[word][1][key][0])

                else:
                    phrasel_List[key -
                                 1].append(positionalIndex[word][1][key][0])


print("==========================This is the phrasel list=============================\n")
print(phrasel_List)
print("\n")

for pos, List in enumerate(phrasel_List, start=1):
    if len(List) == len(query.split()):
        print(pos)

# -------------------Part 3 --------------------------------------
# ----------------------1) Tf ------------------------------------

files_list = []
for file in document_terms:
    for everyfile in file:
        files_list.append(everyfile)


def apply_term_frequency(file):
    wordsfound = dict.fromkeys(files_list, 0)
    for everyfile in file:
        wordsfound[everyfile] += 1
    return wordsfound


term_frequency = pd.DataFrame(apply_term_frequency(
    document_terms[0]).values(), index=apply_term_frequency(document_terms[0]).keys())  # type: ignore


# for i in range(1, len(document_terms)):
for i in range(10):
    term_frequency[i] = apply_term_frequency(document_terms[i]).values()

term_frequency.columns = ['doc'+str(i)
                          for i in range(10)]  # type: ignore


def apply_weightedTF(x):
    if x > 0:
        return math.log(x) + 1
    return 0


for i in range(1):
    term_frequency["doc"+str(i)].apply(apply_weightedTF)
    print("\n======================Term Frequency  TF================================")
    print("\n")
    print(term_frequency)

idf = pd.DataFrame(columns=['freq', 'idf'])
# term frequency : no of term in document
# document freaquency : the doc how many times the term exist
# idf : informaativeness of term
#idf : log10* (N/df)
for i in range(len(term_frequency)):

    frequency = term_frequency.iloc[i].values.sum()  # type: ignore
    # sum : sum of term in all documents
    idf.loc[i, 'freq'] = frequency

    idf.loc[i, 'idf'] = math.log10(10 / float((frequency)))

idf.index = term_frequency.index

print("==============================IDF================================================")
print(idf)

#tf.idf :matrix
# doc 1 = 50 word , doc2 = 300 word
# score(doc1,query)= [1+log10 3(tf)]*idf
# score(doc2,query)= [1+log10 10(tf)]*idf

# =-----------------tf.idf============================================
term_freq_inve_doc_freq = term_frequency.multiply(idf['idf'], axis=0)
document_length = pd.DataFrame()


def get_docs_length(col):
    return np.sqrt(term_freq_inve_doc_freq[col].apply(lambda x: x**2).sum())


# vector similarity = difference =  sqrt (3-2)**2 +
# d1 = 3 [, 5, 10]
#d2 = [2,7,9]
for column in term_freq_inve_doc_freq.columns:
    document_length.loc[0, column+'_len'] = get_docs_length(column)

normalized_term_freq_idf = pd.DataFrame()


def get_normalized(col, x):
    try:
        return x / document_length[col+'_len'].values[0]
    except:
        return 0


for column in term_freq_inve_doc_freq.columns:
    normalized_term_freq_idf[column] = term_freq_inve_doc_freq[column].apply(
        lambda x: get_normalized(column, x))

print("========================this is normalized TF_IDF================================")
print("\n")
print(normalized_term_freq_idf)


def get_w_tf(x):
    try:
        return math.log10(x) + 1
    except:
        return 0


print("===========================Isert query matching=============================== ")


# def insert_query(q):
#    d = q.split()
#    for g in d:
#        quer = pd.DataFrame(index=query=pd.DataFrame(index=normalized_tf_idf.index)

#        quer['tf'] = [1 if x in document_terms else 0 for x in list(
#            normalized_term_freq_idf.index)]  # type: ignore
#        quer['w_tf'] = quer['tf'].apply(lambda x: get_w_tf(x))
#        product = normalized_term_freq_idf.multiply(quer['w_tf'], axis=0)
#        quer['idf'] = idf['idf'] * quer['w_tf']
#        quer['tf_idf'] = quer['w_tf'] * quer['idf']
#
#         quer['normalized'] = 0
#        for i in range(len(quer)):
#            quer['normalized'].iloc[i] = float(
#                quer['idf'].iloc[i]) / math.sqrt(sum(quer['idf'].values**2))  # type: ignore
#        print('Query Details')
#        print(quer.loc[document_terms])  # type: ignore
#        product2 = product.multiply(quer['normalized'], axis=0)
#        scores = {}
#        for col in product2.columns:
#            if 0 in product2[col].loc[document_terms].values:  # type: ignore
#                pass
#            else:
#                scores[col] = product2[col].sum()
#        product_result = product2[list(
#            scores.keys())].loc[document_terms]  # type: ignore
#        print()
#        print('Product (query*matched doc)')
#        print(product_result)
#        print()
#        print('product sum')
#        print(product_result.sum())
#        print()
#        print('Query Length')
#        q_len = math.sqrt(
#            sum([x**2 for x in query['idf'].loc[document_terms]]))  # type: ignore
#        print(q_len)
#        print()
#        print('Cosine Simliarity')
#        print(product_result.sum())
#        print()
#        sorted_scores = sorted(
#            scores.items(), key=lambda x: x[1], reverse=True)
#        print('Returned docs')
#        for typle in sorted_scores:
#            print(typle[0])


#q = input('Input Query for print Query details and matched document\n')
# insert_query(q)
q = input("pleaes enter query")
q.split()
query = pd.DataFrame(index=normalized_term_freq_idf.index)
query['tf'] = [1 if x in q.split() else 0 for x in list(
    normalized_term_freq_idf.index)]
query['w_tf'] = query['tf'].apply(lambda x: get_w_tf(x))
product = normalized_term_freq_idf.multiply(query['w_tf'], axis=0)
query['idf'] = tfd['idf']*query['w_tf']
query['tf-idf'] = query['w_tf']*query['idf']
query['norm'] = 0
for i in range(len(query)):
    query['norm'].iloc[i] = float(
        query['idf'].iloc[i])/math.sqrt(sum(query['idf'].values**2))  # type: ignore

print(query)
print('_______step3query_length____________')
query_length = math.sqrt(sum([x**2 for x in query['idf'].loc[q.split()]]))
print('query_length:'+str(query_length))

print('_______step3cosine_similarity____________')
product2 = product.multiply(query['norm'], axis=0)
scores = {}
for col in product2.columns:
    if 0 in product2[col].loc[q.split()].values:
        pass
    else:
        scores[col] = product2[col].sum()
print('cosine_similarity'+str(scores))
print('_______step3product=(query*matched docs)____________')
prod_res = product2[list(scores.keys())].loc[q.split()]
print(prod_res)
print('sum '+str(list(prod_res.sum())))
print('_______step3ranked____________')
final_score = sorted(scores.items(), key=lambda x: x[1], reverse=True)
print('returned doc ')
for doc in final_score:
    print(doc[0], end='  ')
