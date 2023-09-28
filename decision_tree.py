import copy
import pandas as pd
import math

import sys
sys.setrecursionlimit(5000)

class Node:
    """
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    """
    def __init__(self):
        self.left = None
        self.right = None
        self.attr = None
        self.vote = None


############################################################
## Training Tree

def logHelper(x):
    # if x is 0 return 0 (dont split on the feature)
    if x == 0 :return 0
    result = math.log(x, 2)
    res : float = result * (-x)
    return res

def entropyCalculate(f0, f1):
    result = logHelper(f0) + logHelper(f1)
    return result

def mutualInfoCalculate(data, s, labelE):
# find label
    label = data.columns[-1]
# find 0's 
     # where this feature is 0s
    data0 = data[data[s] == 0]
    # find num
    n0 = len(data0)
# find 1's 
    # where this feature is 0s
    data1 = data[data[s] == 1]
    # find num
    n1 = len(data1)
# find fraction 
    f0 = n0 /(n0 + n1)
    f1 = 1 - f0
# find the number of 1's in label when s = 0
    s0l1 = data0[label].sum()
    s0l0 = n0 - s0l1
# calculate the entropy for s = 0
    if n0 == 0: 
        e0 = 0 
    else: 
        e0 = entropyCalculate(s0l0 / n0, s0l1 /n0)
# find the number of 1's in label when s = 1
    s1l1 = data1[label].sum()
    s1l0 = n1 - s1l1
# calculate the entropy for s = 0
    if n1 == 0:
        e1 = 0
    else:
        e1 = entropyCalculate(s1l0 / n1, s1l1 / n1)

# calculate mutual information 
    mutualInfo = labelE - f0 * e0 - f1 * e1
    return mutualInfo, s0l0, s0l1, s1l0, s1l1, data0, data1


    
# calculate and compare all mutual information 
# return the name of the feature 
def findBestFeature(data, lableE):
    n = len(data)
    max = 0.0
    n00, n01, n10, n11 = 0,0,0,0
    featureName = ''
    data0, data1 = [],[]
    # loop that will calculate mutual information for all features and keeps tracks of the biggest mutual info
    for s in data.columns[: -1]:
        (curMI, s0l0, s0l1, s1l0, s1l1, data00, data11) = mutualInfoCalculate(data, s, lableE)
        if curMI>max: 
            max = curMI
            featureName = s
            n00, n01, n10, n11 = s0l0, s0l1, s1l0, s1l1
            data0, data1 = data00, data11
    return featureName, n00, n01, n10, n11, data0, data1, max

def generateTree(data, prevNode, splitFeature, curDepth, maxDepth, labelE, featureNum, minSplit, minMI):
    labelEntropy = labelE
    dataSplit0 = []
    dataSplit1 = []
    curN00, curN01, curN10, curN11 = 0, 0, 0, 0
    curMI = 0
    # find Node 0
    if curDepth == 0:
        num1s = data[splitFeature].sum()
        num0s = data.shape[0] - num1s
        vote = (num0s, num1s)
        prevNode.vote = vote
        labelEntropy = entropyCalculate(num0s / (num0s + num1s), num1s / (num0s + num1s))
        # stop when max depth is also 0 
        if maxDepth == 0: return
        # try to generate the later on tree
        generateTree(data, prevNode, splitFeature, curDepth+1, maxDepth, labelEntropy, featureNum, minSplit, minMI)
    else:
        num1s = data[splitFeature].sum()
        num0s = data.shape[0] - num1s
        labelEntropy = entropyCalculate(num0s / (num0s + num1s), num1s / (num0s + num1s))
    # call findBestFeture
        # return the first splitting condition and related info
        (featureName, n00, n01, n10, n11, data0, data1, bestMI) = findBestFeature(data, labelEntropy)
        # return if there is nothing to split on 
        if featureName == '': return
        if bestMI < minMI: return

        # generateNode
        l = Node()
        l.attr = featureName
        l.vote = (n00, n01)
        prevNode.left = l
        r = Node()
        r.attr = featureName
        r.vote = (n10, n11)
        prevNode.right = r

        # remember the datas
        dataSplit0 = data0
        dataSplit1 = data1
        curN00, curN01, curN10, curN11, curMI = n00, n01, n10, n11, bestMI

    #check whether should continue generate
    if curDepth < maxDepth and curDepth != featureNum and (curN00 + curN01) >= minSplit-2 and (curN10 + curN11) >= minSplit-2:
        if curN00 != len(dataSplit0) and curN01 != len(dataSplit0):
            generateTree(dataSplit0, l, splitFeature, curDepth+1, maxDepth, labelEntropy, featureNum, minSplit, minMI)
        if curN10 != len(dataSplit1) and curN11 != len(dataSplit1):
            generateTree(dataSplit1, r, splitFeature, curDepth+1, maxDepth, labelEntropy, featureNum, minSplit, minMI)
    return 

def constructVotePrint(num0, num1):
    s = '['
    m = ' 0/'
    e = ' 1]'
    n0 = str(num0)
    n1 = str(num1)
    return s + n0 + m + n1 + e 

def printTree(node, depth):
    # print the base Node
    if depth == 0:
        if node.vote == None: return
        n0, n1 = node.vote
        print(constructVotePrint(n0, n1))
    # print the children of current node if exist any 
    # then print its children 
    if node.left != None:
        l = node.left
        n0,n1 = l.vote
        print ('| ' * (depth+1) + l.attr + ' = 0: ' + constructVotePrint(n0, n1))
        printTree(l, depth + 1)
    if node.right != None:
        r = node.right
        n0,n1 = r.vote
        print ('| ' * (depth+1) + r.attr + ' = 1: ' + constructVotePrint(n0, n1))
        printTree(r, depth +1)
    return 

def trainTree(root, data, maxDepth, n, minMI):
    label = data.columns[-1]
    featureNum = data.shape[1]
    generateTree(data, root, label, 0, maxDepth, 0, featureNum, n, minMI)
    return 
############################################################
## Validation error

def findFeature(root):
    if root.left == None:
        if root.right == None:
            return ''
        return root.right.attr
    return root.left.attr

def majorityVote(vote):
    if vote == None: return 0
    n0, n1 = vote
    if n0 > n1: return 0
    return 1

def validateData(root0, root, data, curPredict, i):
    result = curPredict
    # base case return 
    if data.empty:
        return result
    feature = findFeature(root)
    data1 = data.iloc[0]
    # base case move on
    if root.left == None:
        prediction1 = majorityVote(root.vote)
        curPredict.append(prediction1)
        newRoot = copy.deepcopy(root0)
        dataRest = data.drop(data.index[0])
        return validateData(root0, newRoot, dataRest, curPredict, i+1)
    if data1[feature] == 0:
        return validateData(root0, root.left,data,curPredict, i)
    if data1[feature] == 1:
        return validateData(root0, root.right,data, curPredict, i)

def validateError(root, data):
    if root == None: return 0
    label = data.columns[-1]
    labelList = (data[label]).values.tolist()
    total = len(labelList)
    curPredict = []
    root0 = copy.deepcopy(root)
    prediction = validateData(root0, root, data, curPredict, 0)
    wrong = 0
    for i in range(total):
        if labelList[i] != prediction[i]: 
            wrong = wrong + 1
    return 1 - (wrong / total)

############################################################
## prunning 
def prune(node, validation_data):
    if node is None:
        return None
    node.left = prune(node.left, validation_data)
    node.right = prune(node.right, validation_data)

    accuracy_before_prune = validateError(node, validation_data)
    if node.left is None and node.right is None:
        return node

    l, r = node.left, node.right
    node.left, node.right = None, None
    accuracy_after_prune = validateError(node, validation_data)
    # revert or prune
    if accuracy_after_prune < accuracy_before_prune:
        node.left = l
        node.right = r
    return node


############################################################
## splitting criteria

# take in the number of min numb to split
# return the Maximun Validation Accuracy and the list of n's
def splittingNumber(dataTrain, dataVal):
    vaMax = 0.0
    nMax = 0
    
    for i in range(0, 200):
        root = Node()
        trainTree(root, dataTrain, 100, i, 0)
        va = validateError(root, dataVal)

        # Update vaMax and nMax accordingly
        if va > vaMax:
            vaMax = va
            nMax = i

    return vaMax, nMax

############################################################
## mutual Information criteria

# take in min mutual information to split
# return the Maximun Validation Accuracy and the mutual information
def splittingMI(dataTrain, dataVal):
    vaMax = 0.0
    miMax = 0.01
    mi = 0.01
    while mi <= 1:
        root = Node()
        trainTree(root, dataTrain, 100, 0, mi)
        va = validateError(root, dataVal)

        # Update vaMax and nMax accordingly
        if va > vaMax:
            vaMax = va
            miMax = mi
        mi = round((mi + 0.01), 2)

    return vaMax, miMax

############################################################

if __name__ == '__main__':
    #read data
    if len(sys.argv) != 4:
        print("Usage: python3 decision_tree.py <train_file> <valid_file> <parameter1>")
        sys.exit(1)
    dataTPath = sys.argv[1]
    dataVPath = sys.argv[2]
    maxDepth= int(sys.argv[3])   
    # initialize the tree 
    dataTrain = pd.read_csv(dataTPath, sep='\t')
    dataVal = pd.read_csv(dataVPath, sep='\t')
    # maxDepth = 1
    root = Node()
    # train the tree
    trainTree(root, dataTrain, maxDepth, 0, 0)
    # # calculate validation error
    va =validateError(root, dataVal)
    print('VA:')
    print(va)

    ta =validateError(root, dataTrain)
    print('TA:')
    print(ta)

    # print tree
    printTree(root, 0)

    # prun tree
    root0 = Node()
    trainTree(root0, dataTrain, 3, 0, 0)
    prune(root0, dataVal)
    print('prunned')
    printTree(root0, 0)

    #? split tree based on node size
    vMax, nMax = splittingNumber(dataTrain, dataVal)
    print('splitting base on node size:')
    print(vMax)
    print(nMax)

    #? split tree based on MI
    vMax, nMax = splittingMI(dataTrain, dataVal)
    print('splitting base on mutual info:')
    print(vMax)
    print(nMax)
    pass