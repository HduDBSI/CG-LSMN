import os
import json
from data_preprocess.astTools import *
from tqdm import tqdm
#from func_timeout import func_set_timeout, FunctionTimedOut

def getCodeGraphDataByPath(codePath):
    pureAST = code2AST(codePath) #得到AST需要的数据，递归各节点遍历出一棵树 tree
    newtree, nodelist = getNodeList(pureAST)
    ifcount,whilecount,forcount,blockcount,docount,switchcount,alltokens,vocabdict = astStaticCollection(pureAST)
    h,x,vocabdict,edge_index,edge_attr = getFA_AST(newtree, vocabdict)
    return h,x,vocabdict,edge_index,edge_attr

def getOptFAAST(h, edge_index, keyEntities):
        new_h = []
        delete_index = []
        for i,w in enumerate(h):
            if w in keyEntities:
                new_h.append(w)
            else:
                delete_index.append(i)

        new_src = []
        new_tag = []
        index = []
        for i in range(len(edge_index[0])):
            if edge_index[0][i] not in delete_index and edge_index[1][i] not in delete_index:
                new_src.append(edge_index[0][i])
                new_tag.append(edge_index[1][i])
        index = new_src + new_tag
        number_list = list(set(index))
        number_list = sorted(number_list, key=None, reverse=False)

        indexDict = {}
        for i,num in enumerate(number_list):
            indexDict[num] = i

        src = []
        tag = []
        for i in range(len(new_src)):
            src.append(indexDict[new_src[i]])
            tag.append(indexDict[new_tag[i]])
        new_edge_index = [src, tag]
        return new_h, new_edge_index

def getCodeGraphPairByDataItem(codaDataPath, dataItem, vocabList, vocabDict, vocabIndex):
    itemdata = dataItem.split()
    codePathX = codaDataPath + itemdata[0] + '/' + itemdata[1] + '.java'
    codePathY = codaDataPath + itemdata[0] + '/' + itemdata[2] + '.java'
    itemlabel = itemdata[3]
    #print(dataItem)
    #print(codePathX)
    #print(codePathY)
    #print(itemlabel)

    h1,_,_,edge_index1,_ = getCodeGraphDataByPath(codePathX)
    h2,_,_,edge_index2,_ = getCodeGraphDataByPath(codePathY)

    #print(vocabdict1)
    #print(vocabdict2)
    # 创建词列表的同时，构造词典，用词序构造 h1_index、h2_index
    h1_index = []
    h2_index = []
    for v in h1:
        if v not in vocabDict: 
            vocabList.append(v)
            vocabDict[v] = vocabIndex
            h1_index.append(vocabIndex)
            vocabIndex += 1
        else:
            h1_index.append(vocabDict[v])
    for v in h2:
        if v not in vocabDict:
            vocabList.append(v)
            vocabDict[v] = vocabIndex
            h2_index.append(vocabIndex)
            vocabIndex += 1
        else:
            h2_index.append(vocabDict[v])
    return vocabList, vocabDict, vocabIndex, h1_index, edge_index1, h2_index, edge_index2, itemlabel

def getCodeGraphPairByDataItemOpt(codaDataPath, dataItem, vocabList, vocabDict, vocabIndex):
    itemdata = dataItem.split()
    codePathX = codaDataPath + itemdata[0] + '/' + itemdata[1] + '.java'
    codePathY = codaDataPath + itemdata[0] + '/' + itemdata[2] + '.java'
    keyEntitiesXPath = codePathX.replace(".java", "_keyEntities.txt")
    keyEntitiesYPath = codePathY.replace(".java", "_keyEntities.txt")
    itemlabel = itemdata[3]
    #print(dataItem)
    #print(codePathX)
    #print(codePathY)
    #print(itemlabel)
    
    h1,_,_,edge_index1,_ = getCodeGraphDataByPath(codePathX)
    h2,_,_,edge_index2,_ = getCodeGraphDataByPath(codePathY)
    with open(keyEntitiesXPath, 'r') as f:
        keyEntitiesX = f.read().split()
    h1, edge_index1 = getOptFAAST(h1, edge_index1, keyEntitiesX)
    
    with open(keyEntitiesYPath, 'r') as f:
        keyEntitiesY = f.read().split()
    h2, edge_index2 = getOptFAAST(h2, edge_index2, keyEntitiesY)
    #print(vocabdict1)
    #print(vocabdict2)
    # 创建词列表的同时，构造词典，用词序构造 h1_index、h2_index
    h1_index = []
    h2_index = []
    for v in h1:
        if v not in vocabDict: 
            vocabList.append(v)
            vocabDict[v] = vocabIndex
            h1_index.append(vocabIndex)
            vocabIndex += 1
        else:
            h1_index.append(vocabDict[v])
    for v in h2:
        if v not in vocabDict:
            vocabList.append(v)
            vocabDict[v] = vocabIndex
            h2_index.append(vocabIndex)
            vocabIndex += 1
        else:
            h2_index.append(vocabDict[v])
    
    return vocabList, vocabDict, vocabIndex, h1_index, edge_index1, h2_index, edge_index2, itemlabel

def getDataItemListByPath(labelPath):
    labelFile = open(labelPath, 'r')
    dataItemList = labelFile.readlines()
    labelFile.close()
    print("dataItemList",len(dataItemList))
    return dataItemList


def getVocabListByAST(codaDataPath, dataItemList, saveVocab2TxtPath, vocabDictPath, newLabelPath):
    #不存在则获取词表，写入txt文件，方便下次直接使用
    vocabList = []
    vocabDict = {} # 创建词列表的同时，构造词典
    vocabIndex = 0  # 词为键，词序为值，以便于用词序构造 h1_index、h2_index
    th = 12000
    # dataDict 保存路径
    dataDictSavePath = codaDataPath + "allDataDict.json"
    #dataDictSavePath = codaDataPath + "allDataDict_arg_noArg.json"
    #dataDictSavePath = codaDataPath + "allDataDict-graphOpt.json"
    allDataDict = {}
    newDataItemList = []
    astparseexceptions = 0
    if not os.path.exists(saveVocab2TxtPath):
        print("词表文件不存在，获取中！")
        for dataItem in tqdm(dataItemList):
            try:
                vocabList, vocabDict, vocabIndex, h1_index, edge_index1, h2_index, edge_index2, itemlabel = getCodeGraphPairByDataItem(codaDataPath, dataItem, vocabList, vocabDict, vocabIndex)
                #vocabList, vocabDict, vocabIndex, h1_index, edge_index1, h2_index, edge_index2, itemlabel = getCodeGraphPairByDataItemOpt(codaDataPath, dataItem, vocabList, vocabDict, vocabIndex)
                
                # 执行到此说明没有异常，则进行下一步
                if len(h1_index) < th and len(h2_index) < th:
                    newDataItemList.append(dataItem)
                    dataItemDict = {}
                    dataItemDict["h1_index"] = h1_index
                    dataItemDict["edge_index1"] = edge_index1
                    dataItemDict["h2_index"] = h2_index
                    dataItemDict["edge_index2"] = edge_index2
                    dataItemDict["itemlabel"] = int(itemlabel)
                    # 套娃，数据全部保存在dict中以便加载
                    allDataDict['_'.join([dataItem.split()[0],dataItem.split()[1],dataItem.split()[2]])] = dataItemDict
            except:
                astparseexceptions+=1
                print(astparseexceptions)
        vocabTxt = open(saveVocab2TxtPath, 'w')
        for v in vocabList:
            vocabTxt.write(v + '\n')
        vocabTxt.close()

        # 保存 vocabDict 文件 到本地
        vocabDictFile = open(vocabDictPath, "w")
        json.dump(vocabDict,vocabDictFile)
        vocabDictFile.close()

        # 保存到 dataDict json 文件
        dataDictFile = open(dataDictSavePath, "w")
        json.dump(allDataDict,dataDictFile)
        dataDictFile.close()

        # 保存 newDataItemList 到本地
        newDataItemListFile = open(newLabelPath, "w")
        for item in newDataItemList:
            newDataItemListFile.write(item)
        newDataItemListFile.close()

        # AST异常数目
        print("astparseexceptions", astparseexceptions)
    # 词表存在则直接读取词表
    else:
        print("词表文件已存在，读取中！")
        vocabTxt = open(saveVocab2TxtPath, 'r')
        vocabList = vocabTxt.readlines()
        vocabTxt.close()
        print("vocabList", len(vocabList))

        print("\n词典本地加载中！")
        # read file
        with open(vocabDictPath, 'r') as vocabDictFile:
            vocabDictData = vocabDictFile.read()
        # parse file
        vocabDict = json.loads(vocabDictData)
        print("vocabDict",len(vocabDict))

        print("\n读取 new_labels 中！")
        with open(newLabelPath, 'r') as newDataItemListFile:
            newDataItemList = newDataItemListFile.readlines()
        print("newLabels",len(newDataItemList))

        print("\n加载所有数据样本到内存中！")
        with open(dataDictSavePath, 'r') as dataDictFile:
            allData = dataDictFile.read()
        allDataDict = json.loads(allData)
        pop_list = []
        for item in allDataDict:
            if len(allDataDict[item]['h1_index']) > th or len(allDataDict[item]['h2_index']) > th:
                pop_list.append(item)
        for item in pop_list:
            allDataDict.pop(item)
    print("allDataDict",len(allDataDict))
    return vocabList, vocabDict, newDataItemList, allDataDict

def getAllProjectNames(dataItemList):
    projectList = []
    for data in dataItemList:
        projectName = data.split()[0].split('/')[1].split('_')[0]
        if projectName not in projectList:
            #print(projectName)
            projectList.append(projectName)
    print('projectList',len(projectList))
    return projectList

def getTrainAndTestSetBySeedFold(project_list, fold_num, fold_idx): # e.g. 分为5折（1，2，3，4，5）
    fold_size = len(project_list)//fold_num
    #print("data_list",len(project_list))
    print("fold_size",fold_size)
    train_project_list = []
    test_project_list = []
    for index, value in enumerate(project_list):
        if index >=  (fold_idx - 1) * fold_size and index < fold_idx * fold_size:
            test_project_list.append(value)
        else:
            train_project_list.append(value)
    return train_project_list, test_project_list

def getDataListByProjectList(dataItemList, task, train_project_list, test_project_list):
    train_list = []
    test_list = []
    for data in dataItemList:
        if data.split()[0].split('/')[1].split('_')[0] in train_project_list:
            train_list.append(data)
        elif data.split()[0].split('/')[1].split('_')[0] in test_project_list:  
            test_list.append(data)
            # if task == 1:
            #     if data.split()[0].split('/')[1].split('_')[-1] == '0': #测试集只采用原始样本,忽略数据增强的样本
            #         test_list.append(data)
            # elif task == 2:
            #     if '_'.join(data.split()[0].split('/')[1].split('_')[-2:]) == 'pos_0' or data.split()[0].split('/')[1].split('_')[-2] == 'neg': #测试集只采用原始样本,忽略数据增强的样本
            #         test_list.append(data)
    print("train_list",len(train_list))
    print("test_lsit",len(test_list))
    return train_list, test_list



