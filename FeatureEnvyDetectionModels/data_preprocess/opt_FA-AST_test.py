import numbers
import os
import json
from types import new_class
from astTools import code2AST, getNodeList, astStaticCollection, getFA_AST
from tqdm import tqdm
#from func_timeout import func_set_timeout, FunctionTimedOut

def getCodeGraphDataByPath(codePath):
    pureAST = code2AST(codePath) #得到AST需要的数据，递归各节点遍历出一棵树 tree
    newtree, nodelist = getNodeList(pureAST)
    ifcount,whilecount,forcount,blockcount,docount,switchcount,alltokens,vocabdict = astStaticCollection(pureAST)
    h,x,vocabdict,edge_index,edge_attr = getFA_AST(newtree, vocabdict)
    return h,x,vocabdict,edge_index,edge_attr

if __name__ == '__main__':
    aClassPath = "/home/yqx/Downloads/ExtractNameFromJavaProject-main/src/test/envyModel/data_preprocess/A.java"
    keyWordPath = "src/test/envyModel/data_preprocess/keywords.txt"

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

    # 原始 FA-AST：
    h,_,_,edge_index,_ = getCodeGraphDataByPath(aClassPath)

    print('h ', len(h),h)
    print("edge_index ", len(edge_index[0]))
    with open(keyWordPath, 'r') as f:
        keyEntities = f.read().split()
        
    print("keyLines ",keyEntities)

    # get opt FA-AST：
    new_h, new_edge_index = getOptFAAST(h, edge_index, keyEntities)

    print()
    print("new_h ",len(new_h),new_h)
    print("new_edge_index ",len(new_edge_index[0]),new_edge_index)
    


    """
    ['ClassDeclaration', 'Modifier', 'public', 'Customer', 'FieldDeclaration', 'Modifier', 'private', 'ReferenceType', 'String', 'VariableDeclarator', 'name', 'FieldDeclaration', 'Modifier', 'private', 'ReferenceType', 'List', 'TypeArgument', 'ReferenceType', 'Rental', 'VariableDeclarator', 'rentals', 'ClassCreator', 'ReferenceType', 'ArrayList', 'TypeArgument', 'ReferenceType', 'Rental', 'ConstructorDeclaration', 'Modifier', 'public', 'Customer', 'FormalParameter', 'ReferenceType', 'String', 'name', 'StatementExpression', 'This', 'MethodInvocation', 'MemberReference', 'name', 'setName', 'MethodDeclaration', 'Modifier', 'public', 'ReferenceType', 'String', 'getName', 'ReturnStatement', 'MemberReference', 'name', 'MethodDeclaration', 'Modifier', 'public', 'setName', 'FormalParameter', 'ReferenceType', 'String', 'name', 'StatementExpression', 'Assignment', 'This', 'MemberReference', 'name', 'MemberReference', 'name', '=', 'MethodDeclaration', 'Modifier', 'public', 'ReferenceType', 'List', 'TypeArgument', 'ReferenceType', 'Rental', 'getRentals', 'ReturnStatement', 'MemberReference', 'rentals', 'MethodDeclaration', 'Modifier', 'public', 'setRentals', 'FormalParameter', 'ReferenceType', 'List', 'TypeArgument', 'ReferenceType', 'Rental', 'rentals', 'StatementExpression', 'Assignment', 'This', 'MemberReference', 'rentals', 'MemberReference', 'rentals', '=', 'MethodDeclaration', 'Modifier', 'public', 'addRental', 'FormalParameter', 'ReferenceType', 'Rental', 'rental', 'StatementExpression', 'MethodInvocation', 'rentals', 'MemberReference', 'rental', 'add', 'MethodDeclaration', 'Modifier', 'public', 'ReferenceType', 'String', 'getReport', 'LocalVariableDeclaration', 'ReferenceType', 'String', 'VariableDeclarator', 'result', 'BinaryOperation', '+', 'BinaryOperation', '+', 'Literal', '"Customer Report for "', 'MethodInvocation', 'getName', 'Literal', '"\\n"', 'LocalVariableDeclaration', 'ReferenceType', 'List', 'TypeArgument', 'ReferenceType', 'Rental', 'VariableDeclarator', 'rentals', 'MethodInvocation', 'getRentals', 'LocalVariableDeclaration', 'BasicType', 'double', 'VariableDeclarator', 'totalCharge', 'Literal', '0', 'LocalVariableDeclaration', 'BasicType', 'int', 'VariableDeclarator', 'totalPoint', 'Literal', '0', 'ForStatement', 'EnhancedForControl', 'VariableDeclaration', 'ReferenceType', 'Rental', 'VariableDeclarator', 'each', 'MemberReference', 'rentals', 'BlockStatement', 'LocalVariableDeclaration', 'BasicType', 'double', 'VariableDeclarator', 'eachCharge', 'Literal', '0', 'LocalVariableDeclaration', 'BasicType', 'int', 'VariableDeclarator', 'eachPoint', 'Literal', '0', 'LocalVariableDeclaration', 'BasicType', 'int', 'VariableDeclarator', 'daysRented', 'Literal', '0', 'IfStatement', 'BinaryOperation', '==', 'MethodInvocation', 'each', 'getStatus', 'Literal', '1', 'BlockStatement', 'LocalVariableDeclaration', 'BasicType', 'long', 'VariableDeclarator', 'diff', 'BinaryOperation', '-', 'MethodInvocation', 'each', 'MethodInvocation', 'getTime', 'getReturnDate', 'MethodInvocation', 'each', 'MethodInvocation', 'getTime', 'getRentDate', 'StatementExpression', 'Assignment', 'MemberReference', 'daysRented', 'BinaryOperation', '+', 'Cast', 'BasicType', 'int', 'BinaryOperation', '/', 'MemberReference', 'diff', 'BinaryOperation', '*', 'BinaryOperation', '*', 'BinaryOperation', '*', 'Literal', '1000', 'Literal', '60', 'Literal', '60', 'Literal', '24', 'Literal', '1', '=', 'BlockStatement', 'LocalVariableDeclaration', 'BasicType', 'long', 'VariableDeclarator', 'diff', 'BinaryOperation', '-', 'ClassCreator', 'MethodInvocation', 'getTime', 'ReferenceType', 'Date', 'MethodInvocation', 'each', 'MethodInvocation', 'getTime', 'getRentDate', 'StatementExpression', 'Assignment', 'MemberReference', 'daysRented', 'BinaryOperation', '+', 'Cast', 'BasicType', 'int', 'BinaryOperation', '/', 'MemberReference', 'diff', 'BinaryOperation', '*', 'BinaryOperation', '*', 'BinaryOperation', '*', 'Literal', '1000', 'Literal', '60', 'Literal', '60', 'Literal', '24', 'Literal', '1', '=', 'SwitchStatement', 'MethodInvocation', 'each', 'MethodInvocation', 'getPriceCode', 'getVideo', 'SwitchStatementCase', 'MemberReference', 'Video', 'REGULAR', 'StatementExpression', 'Assignment', 'MemberReference', 'eachCharge', 'Literal', '2', '+=', 'IfStatement', 'BinaryOperation', '>', 'MemberReference', 'daysRented', 'Literal', '2', 'StatementExpression', 'Assignment', 'MemberReference', 'eachCharge', 'BinaryOperation', '*', 'BinaryOperation', '-', 'MemberReference', 'daysRented', 'Literal', '2', 'Literal', '1.5', '+=', 'BreakStatement', 'SwitchStatementCase', 'MemberReference', 'Video', 'NEW_RELEASE', 'StatementExpression', 'Assignment', 'MemberReference', 'eachCharge', 'BinaryOperation', '*', 'MemberReference', 'daysRented', 'Literal', '3', '=', 'BreakStatement', 'StatementExpression', 'MemberReference', '++', 'eachPoint', 'IfStatement', 'BinaryOperation', '==', 'MethodInvocation', 'each', 'MethodInvocation', 'getPriceCode', 'getVideo', 'MemberReference', 'Video', 'NEW_RELEASE', 'StatementExpression', 'MemberReference', '++', 'eachPoint', 'IfStatement', 'BinaryOperation', '>', 'MemberReference', 'daysRented', 'MethodInvocation', 'each', 'getDaysRentedLimit', 'StatementExpression', 'Assignment', 'MemberReference', 'eachPoint', 'MethodInvocation', 'Math', 'MemberReference', 'eachPoint', 'MethodInvocation', 'each', 'MethodInvocation', 'getLateReturnPointPenalty', 'getVideo', 'min', '-=', 'StatementExpression', 'Assignment', 'MemberReference', 'result', 'BinaryOperation', '+', 'BinaryOperation', '+', 'BinaryOperation', '+', 'BinaryOperation', '+', 'BinaryOperation', '+', 'BinaryOperation', '+', 'BinaryOperation', '+', 'BinaryOperation', '+', 'Literal', '"\\t"', 'MethodInvocation', 'each', 'MethodInvocation', 'getTitle', 'getVideo', 'Literal', '"\\tDays rented: "', 'MemberReference', 'daysRented', 'Literal', '"\\tCharge: "', 'MemberReference', 'eachCharge', 'Literal', '"\\tPoint: "', 'MemberReference', 'eachPoint', 'Literal', '"\\n"', '+=', 'StatementExpression', 'Assignment', 'MemberReference', 'totalCharge', 'MemberReference', 'eachCharge', '+=', 'StatementExpression', 'Assignment', 'MemberReference', 'totalPoint', 'MemberReference', 'eachPoint', '+=', 'StatementExpression', 'Assignment', 'MemberReference', 'result', 'BinaryOperation', '+', 'BinaryOperation', '+', 'BinaryOperation', '+', 'BinaryOperation', '+', 'Literal', '"Total charge: "', 'MemberReference', 'totalCharge', 'Literal', '"\\tTotal Point:"', 'MemberReference', 'totalPoint', 'Literal', '"\\n"', '+=', 'IfStatement', 'BinaryOperation', '>=', 'MemberReference', 'totalPoint', 'Literal', '10', 'BlockStatement', 'StatementExpression', 'MethodInvocation', 'System.out', 'Literal', '"Congrat! You earned one free coupon"', 'println', 'IfStatement', 'BinaryOperation', '>=', 'MemberReference', 'totalPoint', 'Literal', '30', 'BlockStatement', 'StatementExpression', 'MethodInvocation', 'System.out', 'Literal', '"Congrat! You earned two free coupon"', 'println', 'ReturnStatement', 'MemberReference', 'result', 'MethodDeclaration', 'updateVideo', 'FormalParameter', 'ReferenceType', 'String', 'videoTitle', 'LocalVariableDeclaration', 'ReferenceType', 'List', 'TypeArgument', 'ReferenceType', 'Rental', 'VariableDeclarator', 'customerRentals', 'MethodInvocation', 'getRentals', 'ForStatement', 'EnhancedForControl', 'VariableDeclaration', 'ReferenceType', 'Rental', 'VariableDeclarator', 'rental', 'MemberReference', 'customerRentals', 'BlockStatement', 'IfStatement', 'BinaryOperation', '&&', 'MethodInvocation', 'rental', 'MethodInvocation', 'getTitle', 'MethodInvocation', 'MemberReference', 'videoTitle', 'equals', 'getVideo', 'MethodInvocation', 'rental', 'MethodInvocation', 'isRented', 'getVideo', 'BlockStatement', 'StatementExpression', 'MethodInvocation', 'rental', 'returnVideo', 'StatementExpression', 'MethodInvocation', 'rental', 'MethodInvocation', 'Literal', 'false', 'setRented', 'getVideo', 'BreakStatement']
    """

    """
    ['Customer', 'String', 'name', 'List', 'Rental', 'rentals', 'Rental', 'Customer', 'String', 'name', 'name', 'setName', 'String', 'getName', 'name', 'setName', 'String', 'name', 'name', 'name', 'List', 'Rental', 'getRentals', 'rentals', 'setRentals', 'List', 'Rental', 'rentals', 'rentals', 'rentals', 'addRental', 'Rental', 'rental', 'rentals', 'rental', 'add', 'String', 'getReport', 'String', 'getName', 'List', 'Rental', 'rentals', 'getRentals', 'Rental', 'each', 'rentals', 'each', 'getStatus', 'each', 'getTime', 'getReturnDate', 'each', 'getTime', 'getRentDate', 'getTime', 'Date', 'each', 'getTime', 'getRentDate', 'each', 'getPriceCode', 'getVideo', 'Video', 'Video', 'each', 'getPriceCode', 'getVideo', 'Video', 'each', 'getDaysRentedLimit', 'Math', 'each', 'getLateReturnPointPenalty', 'getVideo', 'min', 'each', 'getTitle', 'getVideo', 'println', 'println', 'updateVideo', 'String', 'List', 'Rental', 'getRentals', 'Rental', 'rental', 'rental', 'getTitle', 'equals', 'getVideo', 'rental', 'isRented', 'getVideo', 'rental', 'returnVideo', 'rental', 'setRented', 'getVideo']
    """