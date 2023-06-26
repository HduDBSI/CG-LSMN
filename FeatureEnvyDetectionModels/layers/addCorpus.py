import re
labelPath = "/home/yqx/Documents/my-FeatureEnvy-dataset/Dataset_Method_Class/dataset_liu_format.txt"
newCorpusPath = "/home/yqx/Documents/my-FeatureEnvy-dataset/datasetLiuNewCorpus.txt"

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

def getCorpusTxt(labelPath, newCorpusPath):
    corpusList = []
    with open(labelPath, 'r') as f:
        lines = f.readlines()
        #print("lines",lines)
        for row in lines:
            info = row.split(" ")
            #print(info)
            methodName = info[1]
            srcClassName = info[2]
            tagClassName = info[3]
            stringLine = split_except_alphabetDigitChinese(methodName)+split_except_alphabetDigitChinese(srcClassName)+split_except_alphabetDigitChinese(tagClassName)
            corpusList.append(' '.join(stringLine))
    with open(newCorpusPath, 'w') as f:
        f.writelines('\n'.join(corpusList))

getCorpusTxt(labelPath, newCorpusPath)