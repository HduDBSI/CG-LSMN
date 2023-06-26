
import os
from matplotlib.pyplot import flag
import torch
import time
import random
import argparse
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
import torch.nn.functional as F
from torch import LongTensor, as_tensor
from models.cross_graph_local_match import crossGraphLocalMatch
from layers.focalloss import FocalLoss
from data_preprocess.dataPreprocess import *
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score



print("-----------------------DataInfo------------------------\n")

# ======================模型配置=======================
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--in_features', type=int, default=128)
parser.add_argument('--out_features', type=int, default=128)
parser.add_argument('--dropout', type=int, default=0.1)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--alpha', type=int, default=0.2)
parser.add_argument("--threshold", default=0)
args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
criterion = FocalLoss().to(device)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def getBatchData(train_data, allDataDict, batch_size, batch_index, device):
    start_index = batch_index * batch_size
    end_index = start_index + batch_size
    batch_items = train_data[start_index:end_index]
    batch_data = []
    for item in batch_items:
        data_item = []
        data_name = '_'.join([item.split()[0],item.split()[1],item.split()[2]])
        
        if data_name in allDataDict and len(allDataDict[data_name]["edge_index1"][0])>0 and len(allDataDict[data_name]["edge_index2"][0])>0:
            h1_index = torch.LongTensor(torch.as_tensor(allDataDict[data_name]["h1_index"])).to(device)
            edge_index1 = torch.LongTensor(torch.as_tensor(allDataDict[data_name]["edge_index1"])).to(device)
            h2_index = torch.LongTensor(torch.as_tensor(allDataDict[data_name]["h2_index"])).to(device)
            edge_index2 = torch.LongTensor(torch.as_tensor(allDataDict[data_name]["edge_index2"])).to(device)
            itemlabel = int(allDataDict[data_name]["itemlabel"])
            
            data_item = [h1_index, edge_index1, h2_index, edge_index2, itemlabel]

            batch_data.append(data_item)
        else:
            continue
    return batch_data
    
def Undersampling(trainlist):
    print('trainlist',len(trainlist))
    random.seed(int(time.time()))
    random.shuffle(trainlist)
    pos = 0
    neg = 0
    posSamples = []
    negSamples = []
    selectSamples = []
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
    random.seed(int(time.time()))
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

def countDataSize(allDataDict):
    minNodeSize = 1000000
    maxNodeSize = 0
    avgNodeSize = 0
    sumNodeSize = 0
    th = 20000
    countBth = 0
    num = 0
    for item in allDataDict:
        if len(allDataDict[item]["h1_index"]) > maxNodeSize:
            maxNodeSize = len(allDataDict[item]["h1_index"])
        if len(allDataDict[item]["h2_index"]) > maxNodeSize:
            maxNodeSize = len(allDataDict[item]["h2_index"])

        if len(allDataDict[item]["h1_index"]) < minNodeSize:
            minNodeSize = len(allDataDict[item]["h1_index"])
        if len(allDataDict[item]["h2_index"]) < minNodeSize:
            minNodeSize = len(allDataDict[item]["h2_index"])

        if len(allDataDict[item]["h1_index"]) < th and len(allDataDict[item]["h2_index"]) < th:
            countBth += 1
        #print(allDataDict[item]["h1_index"][:20])
        #print(allDataDict[item]["h2_index"][:20])
        #print(allDataDict[item]["itemlabel"])

        sumNodeSize += len(allDataDict[item]["h1_index"])
        sumNodeSize += len(allDataDict[item]["h2_index"])
        num += 1

    avgNodeSize = sumNodeSize / num
    print('minNodeSize',minNodeSize)
    print('maxNodeSize',maxNodeSize)
    print('avgNodeSize',avgNodeSize)
    print('num',num)
    print('countBth',countBth)

def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list

def test(testlist, model_index, allDataDict):
    with torch.no_grad():
        #model.load_state_dict(torch.load('./model/epoch'+str(model_index)+'.pkl'))
        model.eval()

        notFound = 0
        testCount = 0
        y_preds = []
        y_trues = []
        y_labels = []
        y_scores = []
        batches = split_batch(testlist, 1)
        Test_data_batches = trange(len(batches), leave=True, desc = "Test")
        for i in Test_data_batches:
            #lable
            line_info = batches[i][0].split()
            itemName = '_'.join([line_info[0],line_info[1],line_info[2]])
            label = int(line_info[3])
            #data
            try:
                data = allDataDict[itemName]
                h1_index = LongTensor(as_tensor(data["h1_index"])).to(device)
                edge_index1 = LongTensor(as_tensor(data["edge_index1"])).to(device)
                h2_index = LongTensor(as_tensor(data["h2_index"])).to(device)
                edge_index2 = LongTensor(as_tensor(data["edge_index2"])).to(device)
                label = int(data["itemlabel"])
                
                testCount += 1
            except:
                notFound += 1
            #predict
            output = model(h1_index, edge_index1, h2_index, edge_index2)
            _, predicted = torch.max(output.data, 1)
            
            y_label = [0,1] if label == 1 else [1,0]
            y_trues += [label]
            y_preds += predicted.tolist()
            y_labels += y_label
            y_scores += output.tolist()[0]
            # print("y_trues",y_trues)
            # print("y_preds",y_preds)
            # print("y_scores",y_scores)
            # exit()
            r = recall_score(y_trues, y_preds)
            p = precision_score(y_trues, y_preds)
            f1 = f1_score(y_trues, y_preds)
            acc = accuracy_score(y_trues, y_preds)
            auc = roc_auc_score(y_labels, y_scores)
            #matrix = confusion_matrix(y_trues, y_preds)

            Test_data_batches.set_description("Test (p=%.4g,r=%.4g,f1=%.4g, acc=%.4g, auc=%.4g)" % (p, r, f1, acc, auc))
        #print("testCount",testCount)
        #exit()
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

def train(train_list, test_list, fold_idx, task):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    #print("loaded ", './saveModel/epoch'+str(start_train_model_index)+'.pkl')
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    print("模型总参：", get_parameter_number(model))

    fold_index_record = open(test_result, 'a')
    fold_index_record.write("-------fold_index_record:" + str(fold_idx) + "--------")
    fold_index_record.close()

    f1_max = 0
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
        #train_data = train_list
        #random.shuffle(train_data)
        print("train_data",len(train_data))
        model.train()
        for batch_index in tqdm(range(int(len(train_data)/args.batch_size))):
            batch_data = getBatchData(train_data, allDataDict, args.batch_size, batch_index, device)
            
            batchloss= 0
            for data in batch_data:
                h1_index, edge_index1, h2_index, edge_index2, itemlabel = data
                #print("itemlabel",itemlabel)
                itemlabel = torch.Tensor([[0,1]]).to(device) if itemlabel == 1 else torch.Tensor([[1,0]]).to(device)
                output = model(h1_index, edge_index1, h2_index, edge_index2)
                #print('h1_index',h1_index[:10])
                #print('itemlabel',itemlabel)
                #print('output',output)
                #exit()
                batchloss = batchloss + criterion(output, itemlabel)
                count += 1
                right += torch.sum(torch.eq(torch.argmax(output, dim=1), torch.argmax(itemlabel, dim=1)))
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

        if f1 > f1_max:
            f1_max = f1
            torch.save(model.state_dict(), saveModelPath+'/'+ str(task) +'__epoch'+str(epoch)+'.pkl')




if __name__ == '__main__':
    #-----------------------------数据集设置--------------------------
    datasetPath = "/home/yqx/Documents/my-FeatureEnvy-dataset/"
    
    for task in range(1,2):
        if task == 1:
            codaDataPath = datasetPath + "Dataset_Class_Class/"
            vocabDictPath = codaDataPath + "vocabDict.json"
            saveVocab2TxtPath = codaDataPath + "vocabDictList.txt"
            labelPath = codaDataPath + "labels.txt"
            newLabelPath = codaDataPath + "new_labels.txt"
            saveModelPath = codaDataPath + 'saveModels'
            test_result = saveModelPath + '/result_ours.txt'
        elif task == 2:
            codaDataPath = datasetPath + "Dataset_Method_Class/"
            vocabDictPath = codaDataPath + "vocabDict.json"
            saveVocab2TxtPath = codaDataPath + "vocabDictList.txt"
            labelPath = codaDataPath + "labels.txt"
            newLabelPath = codaDataPath + "new_labels.txt"
            saveModelPath = codaDataPath + 'saveModels'
            test_result = saveModelPath + '/result_ours.txt'
        print('__task__:',task, codaDataPath)


        seed = 100
        fold_num = 5 #分为5折（1，2，3，4，5）
        print("\n -----------------------DataInfo------------------------")
        print(codaDataPath)
        print("seed =",seed)
        print("fold_num =",fold_num)

        dataItemList = getDataItemListByPath(labelPath)
        vocabList, vocabDict, newDataItemList, allDataDict = getVocabListByAST(codaDataPath, dataItemList, saveVocab2TxtPath, vocabDictPath, newLabelPath)
        vocabSize = len(vocabDict)
        projectList = getAllProjectNames(newDataItemList)
        #首先要获取项目名称列表、然后将其打乱（设定种子）、然后5等/3等分数据集划分出训练集和测试集中包含的项目
        random.seed(seed)
        random.shuffle(projectList)
        
        countDataSize(allDataDict)
        for fold_idx in range(3,6):

            print(' fold_idx:',fold_idx)
            #数据集划分，按项目名称划分，进行5折/3折交叉验证
            train_project_list, test_project_list = getTrainAndTestSetBySeedFold(projectList, fold_num, fold_idx)
            print("train_project_list",len(train_project_list))
            print("test_project_list",len(test_project_list))
            #获取训练集和测试集
            train_list, test_list = getDataListByProjectList(newDataItemList, task, train_project_list, test_project_list)
            
            model = crossGraphLocalMatch(vocabSize, args.hidden, args.in_features, args.out_features, args.num_classes, args.dropout, args.alpha).to(device)
            train(train_list, test_list, fold_idx, task)