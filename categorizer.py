import os
from nltk import word_tokenize
from nltk import pos_tag
from nltk import PorterStemmer
import string
import re
import random as r
import numpy as np
import math
import pandas as pd
import json
import shutil
import time

FILES_FROM_EACH_CATEGORY = 20

#Input: file from collection
#Output: list of words contained within the file (tokenization)
def parse(file):
    words_array = []
    with open(file,"r") as f:
        for i in f.readlines():
            s = re.sub('[' + string.punctuation + ']', '',i)
            for j in word_tokenize(s):
                words_array.append(j)
    return words_array

#Input: the output of parse function (The list of words contained within the file)
#Output: The list of open class words
def stopword_removal_stemming(data):
    ps = PorterStemmer()
    closed_class = ['CD', 'CC', 'DT', 'EX', 'IN', 'LS', 'MD', 'PDT', 'POS', 'PRP', 'PRP$', 'RP', 'TO', 'UH', 'WDT', 'WP', 'WP$', 'WRB']
    data = pos_tag(data)
    open_class_words = []
    for i in data:
    
        if i[1] not in closed_class:
            open_class_words.append(ps.stem(i[0]))
        
    return open_class_words

#Input: The list of open class words, and an optional list of unique words to append it with the output of this function
#Output: Unique words of the list of open class words for a document
def find_unique_words(data,current_array_of_unique_words=None):
    unique_words = []

    if current_array_of_unique_words != None:
        unique_words += current_array_of_unique_words
    for i in data:
        if i not in unique_words:
            unique_words.append(i)
    return unique_words

#Input: The list of open class words of a document, The output of find_unique_words function (the unique stem list), and a list containing all lists of open class words 
#(all processed documents)
#Output: A vector with tf-idf data within its elements.
def tf_idf(data,unique_stems,processed_data):
    tf_idf_data = []
    unique_stemmed_data = unique_stems

    for i in range(len(unique_stemmed_data)):
        tf = data.count(unique_stemmed_data[i])
        df = get_df(unique_stemmed_data[i],processed_data)
        idf = math.log10(get_number_of_docs(processed_data)/df)
        tf_idf_data.append(tf*idf)
    return tf_idf_data

#Input: 1 stem from the list containing all unique stems, and a list containing all lists of open class words
#Output: Document frequency for the stem given as input
def get_df(word, processed_data):
    df = 0
    for i in range(len(processed_data)):
        for j in processed_data[i]:
            if word in j:
                df += 1
                continue
    return df
#Input: A list containing all lists of open class words
#Output: Number of documents within collection
def get_number_of_docs(processed_data):
    N = 0
    for i in range(len(processed_data)):
        N += len(processed_data[i])
    return N

""" def jaccard_set(list1, list2):
    Define Jaccard Similarity function for two sets
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union """
#Input: 2 lists for similarity calculation
#Output: jaccard similarity
def jaccard_set(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))

#Input: 2 vectors, the one contains the tf-idf data of a document from collection E and the other contains a document from collection A
#Output: Cosine similarity between each document
def cosine_similarity(numpy_array1,numpy_array2):
    return np.dot(numpy_array1,numpy_array2)/(np.linalg.norm(numpy_array1)*np.linalg.norm(numpy_array2))


start = time.time() #begin timer

dir_path = os.path.abspath(os.curdir + '/20_newsgroups') #calculate the path that leads to category folders


collection_categories = os.listdir(dir_path)
index = collection_categories.index('UNCATEGORIZED')
collection_A = collection_categories.pop(index)
collection_E = []

#This for picks FILES_FROM_EACH_CATEGORY documents per category
for i in range(len(collection_categories)):
    collection_E.append([])
    doc_number = 0
    pool = []

    while doc_number < FILES_FROM_EACH_CATEGORY:
        j = r.randint(0,len(os.listdir(dir_path + '/' + collection_categories[i])) - 1)
        if j not in pool:
            pool.append(j)
            collection_E[i].append(dir_path + '/' + collection_categories[i] + '/' + os.listdir(dir_path + '/' + collection_categories[i])[j])
            doc_number += 1


"End of task1b"


"""Task2b"""

all_unique_words_E = []
processed_data = []
#if file "saved_processed_data.json" does not exist then create it. Explanation of this json file is given within the report
#if file does exist then simply load it (after else statement)
if not os.path.exists(os.path.abspath(os.curdir + '/saved_processed_data.json')):
    for i in range(len(collection_E)):
        processed_data.append([])

        for j in collection_E[i]:
            a = parse(file=j)
            b = stopword_removal_stemming(a)
            processed_data[i].append(b)
    
            all_unique_words_E = find_unique_words(b,all_unique_words_E)
    body = {
        'processed_data' : processed_data,
        'all_unique_words_E' : all_unique_words_E
    }
    with open('saved_processed_data.json','w+') as json_file:
            json.dump(body, json_file, indent = 4)
            json_file.truncate()
else: 
    with open('saved_processed_data.json','r') as json_file:
        data_json = json.load(json_file)
        processed_data = data_json.get('processed_data')
        all_unique_words_E = data_json.get('all_unique_words_E')
        
all_tf_idf_E_array = np.zeros([1,len(all_unique_words_E)])

#If file "saved_data.csv" does not exist create it. Explanation about this file is given within report
#if file does exist simply load it (after else statement)
if not os.path.exists(os.path.abspath(os.curdir + '/saved_data.csv')):
    for i in range(len(processed_data)):
        for j in range(len(processed_data[i])):
            tf_idf_array = tf_idf(processed_data[i][j],all_unique_words_E,processed_data)
            all_tf_idf_E_array = np.append(all_tf_idf_E_array,[tf_idf_array],axis=0)
        print(i)
    all_tf_idf_E_array = np.delete(all_tf_idf_E_array,0,axis=0)
    ds = pd.DataFrame(all_tf_idf_E_array)
    ds.to_csv(os.path.abspath(os.getcwd()) + '\\saved_data.csv', index=False)
else:
    all_tf_idf_E_array = pd.read_csv("saved_data.csv")
    all_tf_idf_E_array = all_tf_idf_E_array.iloc[:,:].values

"""Start of final task"""


parser = []
tf_idf_data_A = np.zeros([1,len(all_tf_idf_E_array[0,:])])
#This for loads and preprocesses data from collection A

for i in range(len(os.listdir(dir_path + '/' + collection_A))):
    data = parse(dir_path + '/' + collection_A + '/' + os.listdir(dir_path + '/' + collection_A)[i])
    data = stopword_removal_stemming(data)
    parser.append(data)
    tf_idf_data_A = np.append(tf_idf_data_A, [tf_idf(parser[i],all_unique_words_E,processed_data)], axis=0)
tf_idf_data_A = np.delete(tf_idf_data_A,0,axis=0)

"""End of final task"""

all_bucket = []

#This for computes cosine similarity
for i in range(len(parser)):
    bucket = []
    
    for j in range(len(all_tf_idf_E_array[:,0])):
        bucket.append(cosine_similarity(tf_idf_data_A[i,:],all_tf_idf_E_array[j,:]))
    
    all_bucket.append(bucket)


avg_bucket = []
count = 0
categorized_indexes = []

#Compute the average cosine similarity per category
for bucket in all_bucket:
    avg_bucket.append([])
    for i in range(int(len(bucket)/FILES_FROM_EACH_CATEGORY)):
        temp = 0
        for j in range(i*FILES_FROM_EACH_CATEGORY,(i+1)*FILES_FROM_EACH_CATEGORY):
            temp += bucket[j]
        avg_bucket[count].append(temp/FILES_FROM_EACH_CATEGORY)
    categorized_indexes.append(avg_bucket[count].index(max(avg_bucket[count])))
    count += 1

#This for moves files from collection A to their predicted categories in collection E
for i in range(1,len(categorized_indexes)+1):
    print("Moving document {0} from category UNCATEGORIZED to category {1}".format(os.listdir(dir_path + '/' + collection_A)[-1],
                                                                        os.listdir(dir_path)[categorized_indexes[-i]]))

    shutil.move(os.path.abspath(dir_path + '/' + collection_A + '/' + os.listdir(dir_path + '/' + collection_A)[-1]), 
                os.path.abspath(dir_path + '/' + os.listdir(dir_path)[categorized_indexes[-i]] + '/' + os.listdir(dir_path + '/' + collection_A)[-1]))

end = time.time()
print("Time elapsed at", end-start , "seconds")
print("All files have been moved from file UNCATEGORIZED to their predicted files")