import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
from operator import concat
import csv
import numpy as np
from tqdm import tqdm, trange
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pandas as pd
import datetime
from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score,accuracy_score

padV = [0 for i in range(128)]

def readTxt2List(txtPath):
    with open(txtPath, 'r') as f:
        dataItems = f.readlines()
    return dataItems
    
def readJson2Dict(jsonPath):
    with open(jsonPath, 'r') as f:
        jsonData = f.read()
    dataDict = json.loads(jsonData)
    return dataDict


def getMethodVecDict(c2vEmbDict):
    return c2vEmbDict

def getInfoByLabelItem(labelItem):
    infos = labelItem.rstrip('\n').split()
    itemName = infos[0]
    itemInfoPath = datasetDir + infos[0] + "/" + "dataItemInfo.json"
    itemInfo = readJson2Dict(itemInfoPath)
    a_b_flag = ""
    groundTrue = 0

    projectName = itemInfo["projectName"]
    srcClassName = itemInfo["srcClassName"]
    tagClassName = itemInfo["tagClassName"]
    MethodNamesInSrc = itemInfo["MethodNamesInSrc"]
    MethodNamesInTag = itemInfo["MethodNamesInTag"]

    label = itemInfo["label"]
    if label==1:
        a_b_flag = "a"
        groundTrue = True
    elif label == 0:
        a_b_flag = "b"
        groundTrue = False

    methodListOfCode1 = []
    methodListOfCode2 = []
    #print(itemInfo.keys())
    if "sampleMethodName" in itemInfo.keys():
        #print("method & class")
        sampleMethodName = itemInfo["sampleMethodName"]
        methodListOfCode1.append('__'.join([projectName, a_b_flag, srcClassName, sampleMethodName]))
        for methodName in MethodNamesInTag:
            methodListOfCode2.append('__'.join([projectName, a_b_flag, tagClassName, methodName]))
    elif "methodName" in itemInfo.keys():
        #print("class & class")
        sampleMethodName = itemInfo["methodName"]
        removedMethodNamesInTag = itemInfo["removedMethodNamesInTag"]
        methodListOfCode1.append('__'.join([projectName, a_b_flag, srcClassName, sampleMethodName]))
        for methodName in MethodNamesInTag:
            if methodName not in removedMethodNamesInTag:
                methodListOfCode2.append('__'.join([projectName, a_b_flag, tagClassName, methodName]))
    
    return itemName, methodListOfCode1, methodListOfCode2, groundTrue

def getCodeVecBymethodList(methodListOfCode):
    vecList = []
    for m in methodListOfCode:
        try:
            vecList.append(methodVecDict[m])
        except:
            vecList.append(padV)
    arrays = [np.array(x) for x in vecList]
    vector = [np.mean(k) for k in zip(*arrays)]
    return vector

def getAllData2Dict(labelList):
    AllDataDict = {}
    notFound = 0
    print("loading all data to dict")
    for labelItem in tqdm(labelList):
        itemName, methodListOfCode1, methodListOfCode2, groundTrue = getInfoByLabelItem(labelItem)
        vector1 = getCodeVecBymethodList(methodListOfCode1)
        vector2 = getCodeVecBymethodList(methodListOfCode2)
        AllDataDict[itemName] = vector1 + vector2 + [groundTrue]
    print("notFound",notFound)
    return AllDataDict

def save_list_to_csv(datalist, savePath):
    with open(savePath, mode='w', newline='') as predict_file:
        csv_writer = csv.writer(predict_file)
        for row in range(len(datalist)):
            csv_writer.writerow(datalist[row])

def getTrainOrTestDataByLabelList(labelList, savePath):
    dataList = []
    random.shuffle(labelList)
    for labelItem in tqdm(labelList):
        _, methodListOfCode1, methodListOfCode2, groundTrue = getInfoByLabelItem(labelItem)
        vector1 = getCodeVecBymethodList(methodListOfCode1)
        vector2 = getCodeVecBymethodList(methodListOfCode2)
        if len(vector1) < 128 or len(vector2) < 128:
            continue
        if vector1 + vector2 + [groundTrue] not in dataList:
            dataList.append(vector1 + vector2 + [groundTrue])
        #if datasetDir == "/home/yqx/Documents/my-FeatureEnvy-dataset/Dataset_Class_Class/":
        #    dataList.append(vector2 + vector1 + [groundTrue])
    save_list_to_csv(dataList, savePath)
    df = pd.read_csv(savePath,header=None)
    for u in df.columns:
        if df[u].dtype==bool:
            df[u]=df[u].astype('int')
    return df

def train_test(df1,df2):
    Xtrain = df1.iloc[:,:-1]
    Xtest = df2.iloc[:,:-1]

    Ytrain = df1.iloc[:,-1]
    Ytest = df2.iloc[:,-1]
    
    return Xtrain,Xtest,Ytrain,Ytest

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
            
    print("train_list",len(train_list))
    print("test_lsit",len(test_list))
    return train_list, test_list 

def train(X_train, X_test, y_train, y_test, test_csv, test_result):
    """
    运用网格搜索寻找最优参数，再对最优模型进行测试
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    # 网格搜索参数列表
    tuned_parameters = [{'kernel': ['rbf', 'poly', 'sigmoid'], 'gamma': [1/128, 1/256, 1e-2, 1e-3, 1e-4],
                         'C': [0.1, 0.4, 0.7, 1, 10, 100, 500, 1000], 'max_iter': [1000000]}]
    
    grid = GridSearchCV(SVC(probability=True), param_grid=tuned_parameters, cv=5, scoring="roc_auc", verbose=2, n_jobs=-1)
    grid.fit(X_train, y_train)
    cls = grid.best_estimator_
    info = "The best parameters are %s with a score of %0.4f" % (grid.best_params_, grid.best_score_)

    # 把数据交给模型训练
    cls.fit(X_train, y_train)
    test(cls, test_csv, test_result, info)

def test(estimator, test_csv, test_result, info):
    f = open(test_csv)  # 打开文件
    JMove_data = pd.read_csv(f, header=None).values.tolist()
    JMove_vec = []
    JMove_label = []
    for el in JMove_data:
        if el[-1]:
            JMove_label.append(1)
        else:
            JMove_label.append(0)
        el.pop()
        vec = el
        JMove_vec.append(vec)
    f.close()

    # 测试数据
    predict_res = estimator.predict(JMove_vec)
    y_prob = estimator.predict_log_proba(JMove_vec)
    y_probs = []
    for line in y_prob:
        y_probs.append(line[1])

    p = float(format(precision_score(JMove_label, predict_res), '.4f'))
    r = float(format(recall_score(JMove_label, predict_res), '.4f'))
    f1 = float(format(f1_score(JMove_label, predict_res), '.4f'))
    acc = float(format(accuracy_score(JMove_label, predict_res), '.4f'))
    auc = float(format(roc_auc_score(Ytest, y_probs), '.4f'))
    
    print("\n p, r, f1, acc, auc:", p, r, f1, acc, auc)

    test_p_r_f1 = open(test_result, 'a')
    test_p_r_f1.write(info + "\n")
    test_p_r_f1.write(str(p) +" "+ str(r) +" "+ str(f1) +" "+ str(acc) +" "+ str(auc)+"\n")
    test_p_r_f1.close()

    

if __name__ == '__main__':
    
    # 
    for task in range(1,3):
        if task == 1:
            datasetDir = "/home/yqx/Documents/my-FeatureEnvy-dataset/Dataset_Class_Class/"
        elif task == 2:
            datasetDir = "/home/yqx/Documents/my-FeatureEnvy-dataset/Dataset_Method_Class/"

        labelPath = datasetDir + "new_labels.txt"
        c2vEmbedding = "/home/yqx/Documents/my-FeatureEnvy-dataset/MdgC2vData/C2V_Embedding.json"
        saveModelPath = datasetDir + 'saveModels'
        test_result = saveModelPath + '/Path_based_result.txt'
        train_csv = saveModelPath + "train_data.csv"
        test_csv = saveModelPath + "test_data.csv"

        labelList = readTxt2List(labelPath)
        c2vEmbDict = readJson2Dict(c2vEmbedding)
        methodVecDict = getMethodVecDict(c2vEmbDict)
        
        seed = 100
        fold_num = 5 #分为5折（1，2，3，4，5）
        print("\n -----------------------DataInfo------------------------")
        print(datasetDir)
        print(test_result)
        print("seed =",seed)
        print("labelList",len(labelList))
        print("c2vEmbDict",len(c2vEmbDict))

        projectList = getAllProjectNames(labelList)

        #首先要获取项目名称列表、然后将其打乱（设定种子）、然后5等/3等分数据集划分出训练集和测试集中包含的项目
        random.seed(seed)
        random.shuffle(projectList)
        
        for fold_idx in range(1,6):

            print(' fold_idx:',fold_idx)
            #数据集划分，按项目名称划分，进行5折/3折交叉验证
            train_project_list, test_project_list = getTrainAndTestSetBySeedFold(projectList, fold_num, fold_idx)
            print("train_project_list",len(train_project_list))
            print("test_project_list",len(test_project_list))
            #获取训练集和测试集
            train_list, test_list = getDataListByProjectList(labelList, task, train_project_list, test_project_list)

            if(os.path.exists(train_csv)):   # 判断文件是否存在
                os.remove(train_csv)
            if(os.path.exists(test_csv)):   # 判断文件是否存在
                os.remove(test_csv)

            print("loading train data...")
            train_data = getTrainOrTestDataByLabelList(train_list, train_csv)
            print("loading test data...")
            test_data = getTrainOrTestDataByLabelList(test_list, test_csv)

            df1,df2 = train_data, test_data
            print('df1',len(df1))
            print('df2',len(df2))
            Xtrain,Xtest,Ytrain,Ytest = train_test(df1,df2)

            train(Xtrain, Xtest, Ytrain, Ytest, test_csv, test_result)