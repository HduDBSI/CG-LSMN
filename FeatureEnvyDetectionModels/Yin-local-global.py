from cmath import cos
from turtle import distance
import torch
import json
import torch.nn as nn
from torch.nn import Conv1d, Linear, Tanh, ReLU, Sigmoid
import torch.nn.functional as F
import re
from gensim.models import word2vec
import os
from matplotlib.pyplot import flag
import random
import argparse
import torch.optim as optim
from tqdm import tqdm, trange
from data_preprocess.dataPreprocess import *
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_auc_score
from layers.bi_lstm import BiLSTM
from layers.simple_attention_layer import SimpleAttention
from layers.self_attention_layer import SelfAttentionLayer

print("-----------------------DataInfo------------------------\n")

# ======================模型配置=======================
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--lr_decay_step_size', type=int, default=50)
args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

device = "cuda" if torch.cuda.is_available() else "cpu"

datasetPath = "/home/yqx/Documents/my-FeatureEnvy-dataset/Dataset_Class_Class/dataset_liu_format.txt"

paddingVec = [0 for i in range(200)]

def getVecByWord(word):
    try:
        model = word2vec.Word2Vec.load('src/test/envyModel/layers/new_model.bin')
        vec = model[word]
    except:
        vec = paddingVec
        print("vec not found")
    return vec

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

def getNameVec(name_list):
    name_vec_list = []
    if len(name_list) < 5:
        for i in range(len(name_list)):
            name_vec_list.append(getVecByWord(name_list[i]))
        for j in range(5-len(name_list)):
            name_vec_list.append(paddingVec)
    else:
        for i in range(5):
            name_vec_list.append(getVecByWord(name_list[i]))
    return name_vec_list 

def getNameVec2(name_list):
    name_vec_list = []
    for i in range(len(name_list)):
        name_vec_list.append(getVecByWord(name_list[i]))
    return name_vec_list 

def getAllDataDict(datasetPath):
    with open(datasetPath, 'r') as f:
        items = f.readlines()
        datasetDict = {}
        for row in tqdm(items):
            data = row.split(" ")
            dataItemName = str(data[0]) #键
            dataItemDict = {} #值
            name_m = data[1]
            name_ec = data[2]
            name_et = data[3]
            distance = [float(data[4]), float(data[5])]
            label = [[0,1]] if int(data[6]) == 1 else [[1,0]]
            # dataItemDict["names"] = name_m + name_ec + name_et
            dataItemDict["names"] = [name_m, name_ec, name_et]
            dataItemDict["distance"] = distance
            dataItemDict["label"] = label
            datasetDict[dataItemName] = dataItemDict
        return datasetDict

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def getDataItemListByPath(labelPath):
    labelFile = open(labelPath, 'r')
    dataItemList = labelFile.readlines()
    labelFile.close()
    print("dataItemList",len(dataItemList))
    return dataItemList

def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list

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

def Undersampling(trainlist):
    print('trainlist',len(trainlist))
    pos = 0
    neg = 0
    posSamples = []
    negSamples = []
    selectSamples = []
    random.shuffle(trainlist)
    for sample in trainlist:
        if sample.split()[3] == '0':
            neg+=1
            negSamples.append(sample)
        else:
            pos+=1
            posSamples.append(sample)
    print('sample ratio(pos:neg): ',pos,':',neg)
    if pos>=neg:
        sampleNum = int(neg)
        selectSamples = negSamples[:sampleNum] + posSamples[:sampleNum]
    elif neg>pos:
        sampleNum = int(pos)
        selectSamples = negSamples[:sampleNum] + posSamples[:sampleNum]
    random.shuffle(selectSamples)
    pos = 0
    neg = 0
    for item in selectSamples:
        if item.split()[3] == '0':
            neg+=1
        else:
            pos+=1
    print('after sampling(pos:neg): ',pos,':',neg)
    return selectSamples

def getBatchData(train_data, allDataDict, batch_size, batch_index, device):
    start_index = batch_index * batch_size
    end_index = start_index + batch_size
    batch_items = train_data[start_index:end_index]
    batch_data = []
    #print(allDataDict)
    for item in batch_items:
        data_item = []
        data_name = str(item.split("  ")[0].split('/')[1])
        # print("item",item)
        # print("data_name",data_name)
        if data_name in allDataDict:
            #print("data_name",data_name)
            name_m = getNameVec(split_except_alphabetDigitChinese(allDataDict[data_name]["names"][0]))
            name_ec = getNameVec(split_except_alphabetDigitChinese(allDataDict[data_name]["names"][1]))
            name_tc = getNameVec(split_except_alphabetDigitChinese(allDataDict[data_name]["names"][2]))
            names_global = torch.as_tensor(name_m + name_ec + name_tc).unsqueeze(0).permute(1,0,2).to(device)
            names_local = torch.as_tensor([name_ec, name_m, name_tc]).to(device)
            distance = torch.as_tensor(allDataDict[data_name]["distance"]).reshape(1,-1,1).to(device)
            label = torch.Tensor(allDataDict[data_name]["label"]).to(device)
            data_item = [names_global, distance, names_local, label]
            batch_data.append(data_item)
    return batch_data

class LSTM_Block(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super(LSTM_Block, self).__init__()
        self.LSTM = BiLSTM(in_dim, out_dim, 1, False, dropout)
        self.Dense = Linear(out_dim*15, out_dim)

    def forward(self, input):
        input = torch.tanh(self.LSTM(input))
        output = torch.tanh(self.Dense(input.reshape(1,-1)))
        return output

class Conv_Block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Conv_Block, self).__init__()
        self.conv1d = Conv1d(in_dim, out_dim, kernel_size=1, padding=0)
        self.Dense = Linear(out_dim, out_dim)

    def forward(self, input):
        input = torch.tanh(self.conv1d(input))
        output = torch.tanh(self.Dense(input.squeeze(2)))
        return output.unsqueeze(2)

class LocalAndGlobal(nn.Module):
    def __init__(self, vec_dim, dis_dim):
        super(LocalAndGlobal, self).__init__()

        self.conv = Conv_Block(2, 200)
        self.lstm = LSTM_Block(200, 200)
        self.self_attention = SimpleAttention(200, 200)
        self.csd = Linear(2, 200)
        # merge model
        self.full = Linear(600, 128)
        self.out = Linear(128, 2)

    @staticmethod
    def div_with_small_value(n, d, eps=1e-8):
        d = d * (d > eps) + eps * (d <= eps)
        return n / d
    
    def cosine_attention(self, v1, v2):
        a = torch.mm(v1, v2.permute(1, 0))
        v1_norm = v1.norm(p=2, dim=1, keepdim=True)  # (batch, len1, 1)
        v2_norm = v2.norm(p=2, dim=1, keepdim=True).permute(1, 0)  # (batch, len2, 1)
        d = v1_norm * v2_norm
        return self.div_with_small_value(a, d)

    def forward(self, names_global, distance, names_local):
        
        name_ec = names_local[0].squeeze(0)
        name_m = names_local[1].squeeze(0)
        name_tc = names_local[2].squeeze(0)

        # for distance
        distance = self.conv(distance.reshape(1,-1,1)).reshape(1,-1)
        #print("distance",distance.shape)
        # for global semantics
        global_semantics = self.lstm(names_global)
        #print("names_global",names_global.shape)
        # for local semantics -Siamese
        name_ec = self.self_attention(name_ec)
        name_m = self.self_attention(name_m)
        name_tc = self.self_attention(name_tc)
        local_semantics = torch.tanh(self.csd(torch.cat([self.cosine_attention(name_ec,name_m), self.cosine_attention(name_m,name_tc)]).reshape(1,-1)))
        #print("local_semantics",local_semantics.shape)
        fusion_feature = torch.cat([distance, global_semantics, local_semantics]).reshape(1,-1)
        output = torch.tanh(self.full(fusion_feature))
        output = torch.sigmoid(self.out(output))
        return output

def test(testlist, model_index, allDataDict):
    with torch.no_grad():
        #model.load_state_dict(torch.load('./model/epoch'+str(model_index)+'.pkl'))
        model.eval()

        notFound = 0
        testCount = 0
        y_preds = []
        y_trues = []
        y_lables = []
        y_scores = []
        batches = split_batch(testlist, 1)
        Test_data_batches = trange(len(batches), leave=True, desc = "Test")
        for i in Test_data_batches:
            #lable
            #data
            try:
                line_info = batches[i][0].split()
                itemName = line_info[0].split('/')[1]
                label = int(line_info[3])
                data = allDataDict[itemName]
                name_m = getNameVec(split_except_alphabetDigitChinese(data["names"][0]))
                name_ec = getNameVec(split_except_alphabetDigitChinese(data["names"][1]))
                name_tc = getNameVec(split_except_alphabetDigitChinese(data["names"][2]))
                names_global = torch.as_tensor(name_m + name_ec + name_tc).unsqueeze(0).permute(1,0,2).to(device)
                names_local = torch.as_tensor([name_ec, name_m, name_tc]).to(device)
                distance = torch.as_tensor(data["distance"]).reshape(1,-1).to(device)
                #distance = torch.as_tensor(data["distance"]).reshape(1,-1,1).to(device)
                testCount += 1
            except:
                notFound += 1
            #predict
            output = model(names_global, distance, names_local)
            _, predicted = torch.max(output.data, 1)
            
            y_trues += [label]
            y_preds += predicted.tolist()

            y_label = [0,1] if label == 1 else [1,0]
            y_lables += y_label
            y_scores += output.tolist()[0]

            r=recall_score(y_trues, y_preds)
            p=precision_score(y_trues, y_preds)
            f1=f1_score(y_trues, y_preds)
            acc = accuracy_score(y_trues, y_preds)
            auc = roc_auc_score(y_lables, y_scores)
            #matrix = confusion_matrix(y_trues, y_preds)

            Test_data_batches.set_description("Test (p=%.4g,r=%.4g,f1=%.4g, acc=%.4g, auc=%.4g)" % (p, r, f1, acc, auc))
        #print("testCount",testCount)
        print("notFound",notFound)
        #print("acc",acc)
        #print("matrix",type(matrix),matrix)
        #print("tp,tn,fp,fn:",matrix[1][1],matrix[0][0],matrix[0][1],matrix[1][0])
        p = float(format(p, '.4f'))
        r = float(format(r, '.4f'))
        f1 = float(format(f1, '.4f'))
        acc = float(format(acc, '.4f'))
        auc = float(format(auc, '.4f'))
        print("\n p, r, f1, acc, auc:", p, r, f1, acc, auc)
        return p, r, f1, acc, auc

def train(train_list, test_list, fold_idx):
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    print("模型总参：", get_parameter_number(model))

    fold_index_record = open(test_result, 'a')
    fold_index_record.write("\n-------fold_index_record:" + str(fold_idx) + "--------\n")
    fold_index_record.close()

    iterations = 0
    epochs = trange(args.epochs, leave=True, desc = "Epoch")
    for epoch in epochs:
        totalloss=0.0
        main_index=0.0
        count = 0
        right = 0
        acc = 0
        #exit()
        train_data = Undersampling(train_list)
        #random.shuffle(train_data)
        print("train_data",len(train_data))
        model.train()
        for batch_index in tqdm(range(int(len(train_data)/args.batch_size))):
            batch_data = getBatchData(train_data, allDataDict, args.batch_size, batch_index, device)
            #print("batch_data",batch_data)
            batchloss= 0
            for data in batch_data:
                #print("data",data)
                names_global, distance, names_local, label = data
                output = model(names_global, distance, names_local)
                batchloss = batchloss + F.binary_cross_entropy(output, label)
                count += 1
                right += torch.sum(torch.eq(torch.argmax(output, dim=1), torch.argmax(label, dim=1)))
            acc = right*1.0/count
            optimizer.zero_grad()
            #batchloss.backward(retain_graph = True)
            batchloss.backward()
            optimizer.step()
            totalloss += batchloss.item()
            main_index = main_index + len(batch_data)
            loss = totalloss/main_index
            epochs.set_description("Epoch (Loss=%g) (Acc = %g)" % (round(loss,5) , acc))
            iterations += 1

        p, r, f1, acc, auc = test(test_list, epoch, allDataDict)
        print('f1:',f1)
        test_p_r_f1 = open(test_result, 'a')
        test_p_r_f1.write('epoch'+str(epoch) +" "+ str(p) +" "+ str(r) +" "+ str(f1) +" "+ str(acc) +" "+ str(auc)+"\n")
        test_p_r_f1.close()




if __name__ == '__main__':
    #-----------------------------数据集设置--------------------------
    datasetPath = "/home/yqx/Documents/my-FeatureEnvy-dataset/"
    for task in range(1,3):
        if task == 1:
            codaDataPath = datasetPath + "Dataset_Class_Class/"
        elif task == 2:
            codaDataPath = datasetPath + "Dataset_Method_Class/"
        print('__task__:',task, codaDataPath)

        datasetLiuPath = codaDataPath + "dataset_liu_format.txt"
        newLabelPath = codaDataPath + "new_labels.txt"
        saveModelPath = codaDataPath + 'saveModels'
        test_result = saveModelPath + '/yin_local_global.txt'

        seed = 100
        fold_num = 5 #分为5折（1，2，3，4，5）
        print("\n -----------------------DataInfo------------------------")
        print(codaDataPath)
        print(test_result)
        print("seed =",seed)

        dataItemList = getDataItemListByPath(newLabelPath)
        projectList = getAllProjectNames(dataItemList)
        print("Generating dataDict... ")
        allDataDict = getAllDataDict(datasetLiuPath)
        
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
            train_list, test_list = getDataListByProjectList(dataItemList, task, train_project_list, test_project_list)
            model = LocalAndGlobal(200, 200).to(device)
            random.shuffle(test_list)
            train(train_list, test_list, fold_idx)

