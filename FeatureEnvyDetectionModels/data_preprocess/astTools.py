import javalang
from javalang.ast import Node
import csv
from anytree import AnyNode
#import re
import regex as re
import os
import json

def get_token(node):
    token = ''
    #print(isinstance(node, Node))
    #print(str(node))
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__
    #print(node.__class__.__name__,str(node))
    #print(node.__class__.__name__, node)
    return token
def get_child(root):
    #print(root)
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []
 
    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    #print(sub_item)
                    yield sub_item
            elif item:
                #print(item)
                yield item
    return list(expand(children))
def createtree(root,node,nodelist,parent=None):
    id = len(nodelist)
    #print(id)
    token, children = get_token(node), get_child(node)
    if id==0:
        root.token=token
        root.data=node
    else:
        newnode=AnyNode(id=id,token=token,data=node,parent=parent)
    nodelist.append(node)
    for child in children:
        if id==0:
            createtree(root,child, nodelist, parent=root)
        else:
            createtree(root,child, nodelist, parent=newnode)


def get_sequence(node, sequence):
    token, children = get_token(node), get_child(node)
    sequence.append(token)
    #print(len(sequence), token)
    for child in children:
        get_sequence(child, sequence)

def getnodeandedge(node,nodeindexlist,vocabdict,src,tgt,edgetype):
    token=node.token
    nodeindexlist.append(vocabdict[token])
    for child in node.children:
        src.append(node.id)
        tgt.append(child.id)
        edgetype.append(0)
        src.append(child.id)
        tgt.append(node.id)
        edgetype.append(0)
        getnodeandedge(child,nodeindexlist,vocabdict,src,tgt,edgetype)

#Tools
edges={'childFather':0,'Nextsib':1,'Nexttoken':2,'Prevtoken':3,'Nextuse':4,'Prevuse':5,'If':6,'Ifelse':7,'While':8,'For':9,'Nextstmt':10,'Prevstmt':11,'Prevsib':12}

def getedge_nextsib(node,vocabdict,src,tgt,edgetype):
    token=node.token
    for i in range(len(node.children)-1):
        src.append(node.children[i].id)
        tgt.append(node.children[i+1].id)
        edgetype.append(1)
        src.append(node.children[i+1].id)
        tgt.append(node.children[i].id)
        edgetype.append(edges['Prevsib'])
    for child in node.children:
        getedge_nextsib(child,vocabdict,src,tgt,edgetype)
def getedge_flow(node,vocabdict,src,tgt,edgetype,ifedge=False,whileedge=False,foredge=False):
    token=node.token
    if whileedge==True:
        if token=='WhileStatement':
            src.append(node.children[0].id)
            tgt.append(node.children[1].id)
            edgetype.append(edges['While'])
            src.append(node.children[1].id)
            tgt.append(node.children[0].id)
            edgetype.append(edges['While'])
    if foredge==True:
        if token=='ForStatement':
            src.append(node.children[0].id)
            tgt.append(node.children[1].id)
            edgetype.append(edges['For'])
            src.append(node.children[1].id)
            tgt.append(node.children[0].id)
            edgetype.append(edges['For'])
            '''if len(node.children[1].children)!=0:
                src.append(node.children[0].id)
                tgt.append(node.children[1].children[0].id)
                edgetype.append(edges['For_loopstart'])
                src.append(node.children[1].children[0].id)
                tgt.append(node.children[0].id)
                edgetype.append(edges['For_loopstart'])
                src.append(node.children[1].children[-1].id)
                tgt.append(node.children[0].id)
                edgetype.append(edges['For_loopend'])
                src.append(node.children[0].id)
                tgt.append(node.children[1].children[-1].id)
                edgetype.append(edges['For_loopend'])'''
    #if token=='ForControl':
        #print(token,len(node.children))
    if ifedge==True:
        if token=='IfStatement':
            src.append(node.children[0].id)
            tgt.append(node.children[1].id)
            edgetype.append(edges['If'])
            src.append(node.children[1].id)
            tgt.append(node.children[0].id)
            edgetype.append(edges['If'])
            if len(node.children)==3:
                src.append(node.children[0].id)
                tgt.append(node.children[2].id)
                edgetype.append(edges['Ifelse'])
                src.append(node.children[2].id)
                tgt.append(node.children[0].id)
                edgetype.append(edges['Ifelse'])
    for child in node.children:
        getedge_flow(child,vocabdict,src,tgt,edgetype,ifedge,whileedge,foredge)
def getedge_nextstmt(node,vocabdict,src,tgt,edgetype):
    token=node.token
    if token=='BlockStatement':
        for i in range(len(node.children)-1):
            src.append(node.children[i].id)
            tgt.append(node.children[i+1].id)
            edgetype.append(edges['Nextstmt'])
            src.append(node.children[i+1].id)
            tgt.append(node.children[i].id)
            edgetype.append(edges['Prevstmt'])
    for child in node.children:
        getedge_nextstmt(child,vocabdict,src,tgt,edgetype)
def getedge_nexttoken(node,vocabdict,src,tgt,edgetype,tokenlist):
    def gettokenlist(node,vocabdict,edgetype,tokenlist):
        token=node.token
        if len(node.children)==0:
            tokenlist.append(node.id)
        for child in node.children:
            gettokenlist(child,vocabdict,edgetype,tokenlist)
    gettokenlist(node,vocabdict,edgetype,tokenlist)
    for i in range(len(tokenlist)-1):
            src.append(tokenlist[i])
            tgt.append(tokenlist[i+1])
            edgetype.append(edges['Nexttoken'])
            src.append(tokenlist[i+1])
            tgt.append(tokenlist[i])
            edgetype.append(edges['Prevtoken'])
def getedge_nextuse(node,vocabdict,src,tgt,edgetype,variabledict):
    def getvariables(node,vocabdict,edgetype,variabledict):
        token=node.token
        if token=='MemberReference':
            for child in node.children:
                if child.token==node.data.member:
                    variable=child.token
                    variablenode=child
            if not variabledict.__contains__(variable):
                variabledict[variable]=[variablenode.id]
            else:
                variabledict[variable].append(variablenode.id)      
        for child in node.children:
            getvariables(child,vocabdict,edgetype,variabledict)
    getvariables(node,vocabdict,edgetype,variabledict)
    #print(variabledict)
    for v in variabledict.keys():
        for i in range(len(variabledict[v])-1):
                src.append(variabledict[v][i])
                tgt.append(variabledict[v][i+1])
                edgetype.append(edges['Nextuse'])
                src.append(variabledict[v][i+1])
                tgt.append(variabledict[v][i])
                edgetype.append(edges['Prevuse'])  

def getNodeList(tree):
    nodelist = []
    newtree=AnyNode(id=0,token=None,data=None)
    #def createtree(root,node,nodelist,parent=None):
    createtree(newtree, tree, nodelist)
    return newtree, nodelist

def code2AST(codepath):
    programfile=open(codepath,encoding='utf-8')
    programtext=programfile.read()
    #print("programtext",programtext)
    #删除代码中的所有注释
    programtext = re.sub(r'((\/[*]([*].+|[\\n]|\\w|\\d|\\s|[^\\x00-\\xff])+[*]\/))', "", programtext)
    #print("programtext",programtext)
    programtokens=javalang.tokenizer.tokenize(programtext)
    parser=javalang.parse.Parser(programtokens)
    programast=parser.parse_member_declaration()
    programfile.close()
    tree = programast
    return tree


def astStaticCollection(pureAST):
    alltokens=[]
    get_sequence(pureAST,alltokens)
    ifcount=0
    whilecount=0
    forcount=0
    blockcount=0
    docount = 0
    switchcount = 0
    for token in alltokens:
        if token=='IfStatement':
            ifcount+=1
        if token=='WhileStatement':
            whilecount+=1
        if token=='ForStatement':
            forcount+=1
        if token=='BlockStatement':
            blockcount+=1
        if token=='DoStatement':
            docount+=1
        if token=='SwitchStatement':
            switchcount+=1
    alltokens=list(set(alltokens))
    vocabsize = len(alltokens)
    tokenids = range(vocabsize)
    vocabdict = dict(zip(alltokens, tokenids))
    return ifcount,whilecount,forcount,blockcount,docount,switchcount,alltokens,vocabdict



def getFA_AST(newtree, vocabdict):
    x = []
    edgesrc = []
    edgetgt = []
    edge_attr = []
    nextsib=True
    ifedge=True
    whileedge=True
    foredge=True
    blockedge=True
    nexttoken=True
    nextuse=True
    #遍历出纯AST的结点和边
    getnodeandedge(newtree, x, vocabdict, edgesrc, edgetgt, edge_attr)
    
    #添加兄弟边
    if nextsib==True:
        getedge_nextsib(newtree,vocabdict,edgesrc,edgetgt,edge_attr)
    #添加if、while、for的控制流边
    getedge_flow(newtree,vocabdict,edgesrc,edgetgt,edge_attr,ifedge,whileedge,foredge)
    #添加代码块相关的边
    if blockedge==True:
        getedge_nextstmt(newtree,vocabdict,edgesrc,edgetgt,edge_attr)
    #添加到下一个token的边
    tokenlist=[]
    if nexttoken==True:
        getedge_nexttoken(newtree,vocabdict,edgesrc,edgetgt,edge_attr,tokenlist)
    #添加nextuse和preuse边
    variabledict={}
    if nextuse==True:
        getedge_nextuse(newtree,vocabdict,edgesrc,edgetgt,edge_attr,variabledict)
    
    edge_index=[edgesrc, edgetgt]

    # dict key value 互换 
    vocabdict = dict(zip(vocabdict.values(), vocabdict.keys()))

    h = [vocabdict[v] for v in x]

    return h,x,vocabdict,edge_index,edge_attr



def getTypeMetricxDict(typeMetricsfile):
    with open(typeMetricsfile, 'r') as csvfile:
        typeMetricsDict = {}
        spamreader = csv.reader(csvfile)
        line = 0
        for row in spamreader:
            if line>0:
                key = '__'.join(row[:3])
                value = list(map(float, row[3:]))
                typeMetricsDict[key] = value
            line+=1
    #print(len(typeMetricsDict))
    return typeMetricsDict

def getSADT_Dict(codeSatdsfile):
    with open(codeSatdsfile, 'r') as csvfile:
        codesmelldict = {}
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            classposition = row[2].split('/')
            #print(classposition)
            classname = row[1].split('.')[0]
            packagename = 'org'+'.'.join(classposition[1:]).split('org')[1]
            key = '__'.join([classposition[0], packagename, classname])
            codesmelldict[key] = 1
            
    return codesmelldict
    
def getMethodMetricxDict(methodMetricsfile):
    with open(methodMetricsfile, 'r') as csvfile:
        typeMetricsDict = {}
        spamreader = csv.reader(csvfile)
        line = 0
        for row in spamreader:
            if line>0:
                key = '__'.join(row[:4])
                value = list(map(float, row[4:]))
                typeMetricsDict[key] = value
            line+=1
    #print(len(typeMetricsDict))
    return typeMetricsDict


def getCodeSmellDict(codeSmellsfile):
    with open(codeSmellsfile, 'r') as csvfile:
        codesmelldict = {}
        spamreader = csv.reader(csvfile)
        line = 0
        for row in spamreader:
            if line>0:
                key = '__'.join(row[:3])
                value = str(row[3])
                codesmelldict[key] = value
            line+=1
    return codesmelldict

def getCodeSmellDictMethods(codeSmellsfile):
    with open(codeSmellsfile, 'r') as csvfile:
        codesmelldict = {}
        spamreader = csv.reader(csvfile)
        line = 0
        for row in spamreader:
            if line>0:
                key = '__'.join(row[:4])
                value = str(row[4])
                codesmelldict[key] = value
            line+=1
    return codesmelldict

def addItem2labelfile(typelabelfile, codeName, codesmelldict):
    with open(typelabelfile, 'a') as typefile:
        if codesmelldict.__contains__(codeName):
            #print(codesmelldict[codeName])
            typefile.write(codeName+'    '+'_'.join(codesmelldict[codeName].split())+'\n')
        else:
            typefile.write(codeName+'    '+ '0' +'\n')

def getJsonFile(nodelist,x,vocabdict,edge_index,edge_attr,edgelabels,edgelabels_new,typeMetrics,jsonfilepath,codeName):
    #vocabdict键值互换
    mydict_new=dict(zip(vocabdict.values(),vocabdict.keys()))

    jsonfile = {}

    nodes = {}
    for i,v in enumerate(x):
        nodes[i] = mydict_new[v]

    edges = {}
    for i in range(len(edge_attr)):
        key = str(edge_index[0][i])+"->"+str(edge_index[1][i])
        value = edgelabels[edgelabels_new[edge_attr[i]]]
        edges[key] = value


    jsonfile["nodelist"] = list(map(str, nodelist))
    jsonfile["nodes"] = nodes
    jsonfile["edges"] = edges
    jsonfile["metrics"] = typeMetrics[codeName]
    # 保存文件
    file = open(os.path.join(jsonfilepath,codeName+'.json'), "w")
    json.dump(jsonfile,file)
    file.close()
