# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 21:01:30 2017

@author: xzf0724
"""


from email import header
from gensim.models import word2vec
import logging
import csv
import re


vec_dim = 200

def split_except_alphabetDigitChinese(unicode_str):
    """
    @return:['matched str', '']
    """
    if type(unicode_str) != type(u''):
        try:
            unicode_str = unicode_str.decode('utf8')
        except:
            pass
        
    result = re.findall(u'[\u4e00-\u9fa5]+|[A-Z][a-z0-9]{2,}|[A-Z]{2,}(?![a-z])|(?![A-Z])[a-z][a-z0-9]{2,}',unicode_str)
    result = [t.strip().lower() for t in result if t]
    return result

def getCorpusTxt(csvPath, corpusPath):
    corpusList = []
    with open(csvPath, 'r') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            methodName = row[2]
            srcClassName = row[3].split("/")[-1].split(".")[0]
            tagClassName = row[4].split("/")[-1].split(".")[0]
            stringLine = split_except_alphabetDigitChinese(methodName)+split_except_alphabetDigitChinese(srcClassName)+split_except_alphabetDigitChinese(tagClassName)
            corpusList.append(' '.join(stringLine))
    with open(corpusPath, 'w') as f:
        f.writelines('\n'.join(corpusList))


def getDisCorpusTxt(liuDataPath, liuDataPath2, corpusPathDis):
    disList = []
    with open(liuDataPath, 'r') as f1:
        dataItems = f1.readlines()
        for item in dataItems:
            info = item.split()
            #print(info[4])
            #print(info[5])
            disList.append(" ".join([str(float(format(float(info[4]), '.2f'))), str(float(format(float(info[5]), '.2f')))])+'\n')
            
    with open(liuDataPath2, 'r') as f2:
        dataItems = f2.readlines()
        for item in dataItems:
            info = item.split()
            disList.append(" ".join([str(float(format(float(info[4]), '.2f'))), str(float(format(float(info[5]), '.2f')))])+'\n')
    #print("disList",disList)
    with open(corpusPathDis, 'w') as f3:
        f3.writelines(disList)

def dis2vec(corpusPath):
    print("start train word2vec model")
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus(corpusPath)  
    
    model = word2vec.Word2Vec(min_count=0, window=10, size=vec_dim, workers = 3, iter = 10)
    model.build_vocab(sentences)
    model.train(sentences, total_examples = model.corpus_count, epochs = 10)
    model.save("src/test/envyModel/layers/dis_model.bin")
    print(model['1.0'])
    #y1 = model.similarity("create", "apply")
    #print(y1)

def text8Train(corpusPath):
    print("start train word2vec model")
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus(corpusPath)  
    
    model = word2vec.Word2Vec(min_count=0, window=10, size=vec_dim, workers = 3, iter = 10)
    model.build_vocab(sentences)
    model.train(sentences, total_examples = model.corpus_count, epochs = 10)
    model.save("src/test/envyModel/layers/model.bin")
    y1 = model.similarity("create", "apply")
    print(y1)
    
def moreTrain(moreCorpusPath):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    new_model = word2vec.Word2Vec.load('src/test/envyModel/layers/model.bin')
    more_sentences = word2vec.Text8Corpus(moreCorpusPath)

    new_model.build_vocab(more_sentences, update=True)
    new_model.train(more_sentences, total_examples=new_model.corpus_count, epochs=5)
    print(new_model['apply'])
    new_model.save('src/test/envyModel/layers/new_model.bin')
    
def getVecByWord(word):
    model = word2vec.Word2Vec.load('src/test/envyModel/layers/model.bin')
    return model[word]

if __name__ == "__main__":
    csvPath = "/home/yqx/Documents/my-FeatureEnvy-dataset/groundTruth.csv"
    corpusPath = "/home/yqx/Documents/my-FeatureEnvy-dataset/datasetLiuCorpus.txt"
    moreCorpusPath = "/home/yqx/Documents/my-FeatureEnvy-dataset/datasetLiuNewCorpus.txt"

    liuDataPath = "/home/yqx/Documents/my-FeatureEnvy-dataset/Dataset_Class_Class/dataset_liu_format.txt"
    liuDataPath2 = "/home/yqx/Documents/my-FeatureEnvy-dataset/Dataset_Method_Class/dataset_liu_format.txt"
    corpusPathDis = "/home/yqx/Documents/my-FeatureEnvy-dataset/datasetLiuCorpusDis.txt"

    # word2vec----------------------------
    #getCorpusTxt(csvPath, corpusPath)
    #text8Train(corpusPath)
    #moreTrain(moreCorpusPath)
    #print(getVecByWord("apply"))

    # dis2vec-----------------------------
    getDisCorpusTxt(liuDataPath, liuDataPath2, corpusPathDis)
    dis2vec(corpusPathDis)