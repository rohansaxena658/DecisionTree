##############
# Name: Rohan Saxena

import numpy as np
import pandas as pd
import sys
import os
import queue
import copy
import matplotlib.pyplot as plt

#Graph Q1
def graphQ1(trainFile, testFile):
    trainingPercentages = [40,60,80,100]
    trainingAccuracies = []
    testAccuracies = []
    lenTreeList = []
    for percentage in trainingPercentages:
        vanillaTree = Tree(trainFile, testFile)
        validationPercentage = 0
        dataFrame, validationFrame = vanillaTree.buildDataFrame(vanillaTree.trainFile, percentage, validationPercentage)
        testFrame = vanillaTree.buildTestFrame(vanillaTree.testFile)
        rootNode = vanillaTree.buildTree(dataFrame, -1, -1)
        numNodes = rootNode.levelOrder()
        lenTreeList.append(numNodes)
        trainAccuracy = vanillaTree.accuracy(dataFrame, rootNode)
        # if(validationFrame.empty == True):
        #     trainAccuracy = vanillaTree.accuracy(dataFrame, rootNode)
        # else:
        #     trainAccuracy = vanillaTree.accuracy(validationFrame, rootNode)
        testAccuracy = vanillaTree.accuracy(testFrame, rootNode)
        trainingAccuracies.append(trainAccuracy)
        testAccuracies.append(testAccuracy)
        # print("at percentage: %d && trainAccuracy: %.6f && testAccuracy: %.6f" % (percentage, trainAccuracy, testAccuracy))

    #Plot accuracy against trainingPercentages
    fig = plt.figure(1)
    plt.plot(trainingPercentages, trainingAccuracies, 'rs--', label='Training Accuracy')
    plt.plot(trainingPercentages, testAccuracies, 'bs--', label='Test Accuracy')
    plt.legend(loc='lower right')
    fig.suptitle('Accuracy vs Training Percentages (Vanilla Tree)', fontsize=16)
    plt.xlabel('Training Percentages', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.axis([40, 100, 0.6, 1.0])
    fig.savefig('VanillaAccuracy.png')

    #Plot number of nodes against trainingPercentages
    fig2 = plt.figure(2)
    plt.plot(trainingPercentages, lenTreeList, 'rs--', label='Number of Nodes')
    plt.legend(loc='lower right')
    plt.xlabel('Training Percentages', fontsize=16)
    plt.ylabel('Number of Tree Nodes', fontsize=16)
    fig2.suptitle('Number of Nodes vs Training Percentages (Vanilla Tree)', fontsize=16)
    plt.axis([40,100,0,100])
    fig2.savefig('VanillaNumNodes.png')
    plt.show()

#Graph Q2
def graphQ2(trainFile, testFile):
    trainingPercentages = [40,50,60,70,80]
    validationPercentage = 20
    maxList = [-1, -1, -1] #list[0]: percentage + list[1]: maxAccuracy + list[2]: depth where max accuracy was found
    maximumDepth = [5,10,15,20]
    maxDepth = []
    for percentage in trainingPercentages:
        depthTree = Tree(trainFile, testFile)
        dataFrame, validationFrame = depthTree.buildDataFrame(depthTree.trainFile, percentage, validationPercentage)
        testFrame = depthTree.buildTestFrame(depthTree.testFile)
        maxAcc = -1.0
        maxDep = -1.0
        for depth in maximumDepth:
            rootNode = depthTree.buildTree(dataFrame, depth, -1)
            trainAccuracy = depthTree.accuracy(dataFrame, rootNode)
            validationAccuracy = depthTree.accuracy(validationFrame, rootNode)
            testAccuracy = depthTree.accuracy(testFrame, rootNode)
            # print("trainAccuracy at depth: %d & percentage: %d is %.6f" % (depth, percentage, trainAccuracy))
            # print("validationAccuracy at depth: %d & percentage: %d is %.6f" % (depth, percentage, validationAccuracy))
            # print("testAccuracy at depth %d & percentage: %d is %.6f" % (depth, percentage, testAccuracy))
            if(validationAccuracy > maxAcc):
                maxAcc = validationAccuracy
                maxDep = depth
            if(validationAccuracy > maxList[1]):
                maxList[0] = percentage
                maxList[1] = validationAccuracy
                maxList[2] = depth
        maxDepth.append(maxDep)

    #Make plots

    trainingAccuracies = []
    testAccuracies = []
    lenTreeList = []
    for percentage in trainingPercentages:
        depthTree = Tree(trainFile, testFile)
        dataFrame, ValidationFrame = depthTree.buildDataFrame(depthTree.trainFile, percentage, validationPercentage)
        testFrame = depthTree.buildTestFrame(depthTree.testFile)
        rootNode = depthTree.buildTree(dataFrame,maxList[2], -1)
        numNodes = rootNode.levelOrder()
        lenTreeList.append(numNodes)
        trainAccuracy = depthTree.accuracy(dataFrame, rootNode)
        testAccuracy = depthTree.accuracy(testFrame, rootNode)
        trainingAccuracies.append(trainAccuracy)
        testAccuracies.append(testAccuracy)
        # print("@percentage: %d\n\ttrainAccuracy: %.6f\n\ttestAccuracy: %.6f" % (percentage,trainAccuracy, testAccuracy))

    #Plots training accuracies vs training percentages
    fig = plt.figure(1)
    plt.plot(trainingPercentages, trainingAccuracies, 'rs--', label='Training Accuracy')
    plt.plot(trainingPercentages, testAccuracies, 'bs--', label='Test Accuracy')
    plt.legend(loc='lower right')
    fig.suptitle('Accuracy vs Training Percentages (Depth Tree)', fontsize=16)
    plt.xlabel('Training Percentages', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.axis([40, 100, 0.6, 1.0])
    fig.savefig('DepthAccuracy.png')

    #Plots to plot optimal depth for each percentage
    fig2 = plt.figure(2)
    plt.plot(trainingPercentages, maxDepth, 'rs--', label='Optimal Depth')
    plt.legend(loc='lower right')
    plt.xlabel('Training Percentages', fontsize=16)
    plt.ylabel('Depth of Tree', fontsize=16)
    fig2.suptitle('Optimal Depth vs Training Percentages', fontsize=16)
    plt.axis([40,100,0,20])
    fig2.savefig('OptimalDepth.png')

    #Plot number of nodes against trainingPercentages
    fig3 = plt.figure(3)
    plt.plot(trainingPercentages, lenTreeList, 'rs--', label='Number of Nodes')
    plt.legend(loc='lower right')
    plt.xlabel('Training Percentages', fontsize=16)
    plt.ylabel('Number of Tree Nodes', fontsize=16)
    fig3.suptitle('Number of Nodes vs Training Percentages (Depth Tree)', fontsize=16)
    plt.axis([40,100,0,100])
    fig3.savefig('DepthNumNodes.png')

    plt.show()

def graphQ3(trainFile, testFile):
    validationPercentage = 20
    trainingPercentages = [40,50,60,70,80]
    trainingAccuracies = []
    testAccuracies = []
    lenTreeList = []
    for percentage in trainingPercentages:
        pruneTree = Tree(trainFile, testFile)
        dataFrame, validationFrame = pruneTree.buildDataFrame(pruneTree.trainFile, percentage, validationPercentage)
        testFrame = pruneTree.buildTestFrame(pruneTree.testFile)
        rootNode = pruneTree.buildTree(dataFrame, -1,-1)
        trainAccuracy = pruneTree.accuracy(dataFrame, rootNode)
        validationAccuracy = pruneTree.accuracy(validationFrame, rootNode)
        rootNode = pruneTree.prune(rootNode, rootNode, validationAccuracy, validationFrame)
        testAccuracy = pruneTree.accuracy(testFrame, rootNode)
        trainingAccuracies.append(trainAccuracy)
        testAccuracies.append(testAccuracy)
        # print("trainAccuracy: %.6f && validationAccuracy %.6f && testAccuracy: %.6f at percentage: %d" % (trainAccuracy, validationAccuracy, testAccuracy,percentage))
        numNodes = rootNode.levelOrder()
        lenTreeList.append(numNodes)

    #Accuracy Training Percentages
    fig = plt.figure(1)
    plt.plot(trainingPercentages, trainingAccuracies, 'rs--', label='Training Accuracy')
    plt.plot(trainingPercentages, testAccuracies, 'bs--', label='Test Accuracy')
    plt.legend(loc='lower right')
    fig.suptitle('Accuracy vs Training Percentages (Prune Tree)', fontsize=16)
    plt.xlabel('Training Percentages', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.axis([40, 100, 0.6, 1.0])
    fig.savefig('pruneAccuracy.png')

    #Plot number of nodes against trainingPercentages
    fig2 = plt.figure(2)
    plt.plot(trainingPercentages, lenTreeList, 'rs--', label='Number of Nodes')
    plt.legend(loc='lower right')
    plt.xlabel('Training Percentages', fontsize=16)
    plt.ylabel('Number of Tree Nodes', fontsize=16)
    fig2.suptitle('Number of Nodes vs Training Percentages (Prune Tree)', fontsize=16)
    plt.axis([40,100,0,100])
    fig2.savefig('pruneNodes.png')

    plt.show()


def entropy(freqs):
    """
    entropy(p) = -SUM (Pi * log(Pi))
    >>> entropy([10.,10.])
    1.0
    >>> entropy([10.,0.])
    0
    >>> entropy([9.,3.])
    0.811278
    """
    all_freq = sum(freqs)
    entropy = 0
    for fq in freqs:
        prob = fq * (1.0 / all_freq)
        if abs(prob) > 1e-8:
            entropy += -prob * np.log2(prob)
    return entropy

#Custom entropy function
def myEntropy(subset):
    numLiving = len(subset[subset['survived'] == 1])
    numDead = len(subset[subset['survived'] == 0])
    lengthOfSubset = numLiving + numDead
    entropy = 0.0
    firstComponent = 0.0
    secondComponent =  0.0
    numDead
    if(numDead != 0):
        firstComponent = (-numDead / lengthOfSubset) * np.log2((numDead / lengthOfSubset))
    if(numLiving != 0):
        secondComponent = ((numLiving/lengthOfSubset) * np.log2(numLiving/lengthOfSubset))
    entropy = firstComponent - secondComponent
    return round(entropy,4);



def infor_gain(before_split_freqs, after_split_freqs):
    """
    gain(D, A) = entropy(D) - SUM ( |Di| / |D| * entropy(Di) )
    >>> infor_gain([9,5], [[2,2],[4,2],[3,1]])
    0.02922
    """
    gain = entropy(before_split_freqs)
    overall_size = sum(____)
    for freq in after_split_freqs:
        ratio = sum(____) * 1.0 / ____
        gain -= ratio * entropy(___)
    return round(gain,4)

def myInfoGain(entropyList):
    infoGain = 0.0
    for tup in entropyList:
        entropy = tup[0]
        subsetSize = tup[1]
        totalSize = tup[2]
        infoGain += ((subsetSize / totalSize) * entropy)

    infoGain = 1 - infoGain
    return round(infoGain,4)

def checkBaseCase(dataFrame):
    #First Base Case: check if all survived column are the same
    survivedSet = set(dataFrame['survived'].values)
    if(len(survivedSet) == 1):
        return True
    #Second Base Case: When all entries in each column are the same
    for column in dataFrame:
        if(column == 'survived'):
            continue
        columnSet = set(dataFrame[column].values)
        if(len(columnSet) != 1):
            return False;
    return True


class Node(object):
    def __init__(self, dataFrame, columnName, splitVal, infoGain):
        self.dataFrame = dataFrame
        self.columnName = columnName
        self.splitVal = splitVal
        self.rightChild = None
        self.leftChild = None
        self.infoGain = infoGain

    def makeTerminalNode(self):
        self.columnName = "survived"
        numLiving = len(self.dataFrame[self.dataFrame['survived'] == 1])
        numDead = len(self.dataFrame[self.dataFrame['survived'] == 0])
        if(numLiving >= numDead):
            self.splitVal = 1
        else:
            self.splitVal = 0

    #Level order traversal to display Tree
    def levelOrder(self):
        q = queue.Queue()
        numNodes = 0
        q.put(self)
        while(True):
            if(q.empty() == True):
                break
            currentSize = q.qsize()
            numNodes += currentSize
            while(currentSize != 0):
                currentNode = q.get()
                # print("currentNode: %s %d" % (currentNode.columnName, currentNode.splitVal))
                if(currentNode.leftChild != None):
                    q.put(currentNode.leftChild)
                if(currentNode.rightChild != None):
                    q.put(currentNode.rightChild)
                currentSize -= 1
        return numNodes

class Tree(object):
    #Constructor
    def __init__(self, trainFile, testFile):
        self.trainFile = trainFile
        self.testFile = testFile

    #Build testDataFrame
    def buildTestFrame(self, testFile):
        tempFile = testFile
        trainFile = tempFile + ".data"
        labelFile = tempFile + ".label"
        dataFrame = pd.read_csv(trainFile, index_col=None, engine='python')
        labelFrame = pd.read_csv(labelFile, index_col=None, engine='python')
        dataFrame = pd.concat([dataFrame,labelFrame], axis = 1)

        #Turn continuous variables "Fare" to discrete variable
        fareMedian = dataFrame.loc[:,"Fare"].median()
        dataFrame['Fare'] = dataFrame['Fare'].mask(dataFrame['Fare'] < fareMedian,0)
        dataFrame['Fare'] = dataFrame['Fare'].mask(dataFrame['Fare'] >= fareMedian,1)
        ageMedian = dataFrame.loc[:,"Age"].median()
        dataFrame['Age'] = dataFrame['Age'].mask(dataFrame['Age'] < ageMedian,0)
        dataFrame['Age'] = dataFrame['Age'].mask(dataFrame['Age'] >= ageMedian,1)
        relativeMedian = dataFrame.loc[:,"relatives"].median()
        dataFrame['relatives'] = dataFrame['relatives'].mask(dataFrame['relatives'] < relativeMedian,0)
        dataFrame['relatives'] = dataFrame['relatives'].mask(dataFrame['relatives'] >= relativeMedian,1)
        return dataFrame

    #Build dataFrame
    def buildDataFrame(self, trainFile, trainPercentage, validationPercentage):
        # print("Validation Percentage: %d" % (validationPercentage))
        tempFile = trainFile
        trainFile = tempFile + ".data"
        labelFile = tempFile + ".label"
        dataFrame = pd.read_csv(trainFile, index_col=None, engine='python')
        labelFrame = pd.read_csv(labelFile, index_col=None, engine='python')
        dataFrame = pd.concat([dataFrame, labelFrame], axis = 1)
        # print("before modifying dataFrame: \n", dataFrame)
        #get percentage of the data
        tempDataFrame = copy.copy(dataFrame)
        validationFrame = None
        if(trainPercentage != 100):
            trainIndex = int((((trainPercentage / 100)) * len(tempDataFrame))) - 1
            # print("trainIndex: %d" % (trainIndex))
            dataFrame = copy.copy(tempDataFrame.iloc[0 : trainIndex])
        # print("len(tempDataFrame): %d" % len(tempDataFrame))
        if(validationPercentage != 0):
            validationIndex = int((validationPercentage / 100) * len(tempDataFrame) - 1)
            # print("len(tempDataFrame) - validationIndex: %d" % (len(tempDataFrame) - validationIndex))
            validationFrame = copy.copy(tempDataFrame.iloc[len(tempDataFrame) - validationIndex : len(tempDataFrame)])
        elif(validationPercentage == 0):
            validationFrame = pd.DataFrame()

        #Turn continuous variables "Fare" to discrete variable
        fareMedian = dataFrame.loc[:,"Fare"].median()
        dataFrame['Fare'] = dataFrame['Fare'].mask(dataFrame['Fare'] < fareMedian,0)
        dataFrame['Fare'] = dataFrame['Fare'].mask(dataFrame['Fare'] >= fareMedian,1)
        ageMedian = dataFrame.loc[:,"Age"].median()
        dataFrame['Age'] = dataFrame['Age'].mask(dataFrame['Age'] < ageMedian,0)
        dataFrame['Age'] = dataFrame['Age'].mask(dataFrame['Age'] >= ageMedian,1)
        relativeMedian = dataFrame.loc[:,"relatives"].median()
        dataFrame['relatives'] = dataFrame['relatives'].mask(dataFrame['relatives'] < relativeMedian,0)
        dataFrame['relatives'] = dataFrame['relatives'].mask(dataFrame['relatives'] >= relativeMedian,1)
        if(validationFrame.empty == False):
            fareMedian = validationFrame.loc[:,"Fare"].median()
            validationFrame['Fare'] = validationFrame['Fare'].mask(validationFrame['Fare'] < fareMedian,0)
            validationFrame['Fare'] = validationFrame['Fare'].mask(validationFrame['Fare'] >= fareMedian,1)
            ageMedian = validationFrame.loc[:,"Age"].median()
            validationFrame['Age'] = validationFrame['Age'].mask(validationFrame['Age'] < ageMedian,0)
            validationFrame['Age'] = validationFrame['Age'].mask(validationFrame['Age'] >= ageMedian,1)
            relativeMedian = validationFrame.loc[:,"relatives"].median()
            validationFrame['relatives'] = validationFrame['relatives'].mask(validationFrame['relatives'] < relativeMedian,0)
            validationFrame['relatives'] = validationFrame['relatives'].mask(validationFrame['relatives'] >= relativeMedian,1)

        return dataFrame, validationFrame


    #Calculate information gain
    def calculateMaxInfoGain(self, dataFrame):
        infoGainList = []
        maxInfoGain = -1.0
        maxEntropy = -1.0
        maxTuple = (-1.0, -1.0, "") #first component of the tuple is maxInfoGain, second component is maxEntropy, third component is the columnName
        maxColumnName = None
        maxEntropy = 0.0
        splitVal = None
        for column in dataFrame:
            if(column == 'survived'):
                continue

            # print(column)
            columnSet = set(dataFrame[column].values)
            # print(columnSet)
            entropyList = []
            for x in columnSet:
                subset = dataFrame[dataFrame[column] == x]
                # print(subset)
                numLiving = len(subset[subset['survived'] == 1])
                numDead = len(subset[subset['survived'] == 0])
                entropy = myEntropy(subset)
                entropyAndSize = (entropy, (numLiving + numDead), len(dataFrame[column]), x)
                entropyList.append(entropyAndSize)
            informationGain = myInfoGain(entropyList)
            if(informationGain > maxInfoGain):
                maxInfoGain = informationGain
                maxColumnName = column
                maxEntropy = -1.0
                for x in entropyList:
                    if(x[0] > maxEntropy):
                        maxEntropy = x[0]
                        splitVal = x[len(x) - 1]

        #print("RETURNING FROM CALCMAXINFOGAIN: %s %d %.4f" % (maxColumnName, splitVal, maxInfoGain))
        return Node(dataFrame, maxColumnName, splitVal, maxInfoGain)

    #Build Tree
    def buildTree(self, dataFrame, maxDepth, minSplit):
        root = self.calculateMaxInfoGain(dataFrame)
        self.splitTree(root, maxDepth,1, minSplit)
        return root


    #Begin splitting nodes
    def splitTree(self, rootNode, maxDepth, currentDepth, minSplit):
        if(rootNode == None):
            # print("returning none")
            return
        #minSplit Tree
        if(len(rootNode.dataFrame) < minSplit):
            # print("minsplit reached")
            rootNode.makeTerminalNode()
            return
        #Depth base case
        if(currentDepth == maxDepth):
            rootNode.makeTerminalNode()
            return
        #RECURSION BASE CASES
        #Call function to see if base case criteria is met
        if(checkBaseCase(rootNode.dataFrame) == True):
            # print("base case criteria met")
            rootNode.makeTerminalNode()
            return

        centralDataFrame = rootNode.dataFrame
        #Splitting by splitVal
        leftGroupDF = centralDataFrame[centralDataFrame[rootNode.columnName] != rootNode.splitVal]
        rightGroupDF = centralDataFrame[centralDataFrame[rootNode.columnName] == rootNode.splitVal]
        #Dropping the column after we have split
        leftGroupDF = leftGroupDF.drop([rootNode.columnName], axis = 1)
        rightGroupDF = rightGroupDF.drop([rootNode.columnName], axis = 1)
        #Check for emptry frame and make node terminal if empty
        leftNode = self.calculateMaxInfoGain(leftGroupDF)
        rightNode = self.calculateMaxInfoGain(rightGroupDF)
        rootNode.leftChild = leftNode
        rootNode.rightChild = rightNode
        self.splitTree(leftNode, maxDepth, currentDepth + 1, minSplit)
        self.splitTree(rightNode, maxDepth, currentDepth + 1, minSplit)

    #Predict function
    def predict(self, rootNode, row):
        if(rootNode == None):
            return 0
        # print("rootNode.columnName: %s && rootNode.splitVal: %d" % (rootNode.columnName, rootNode.splitVal))
        # print("row[rootNode.columnName]: %d" % (row[rootNode.columnName]))
        #Base case: We are at a terminal node
        if(rootNode.columnName == 'survived'):
            if(rootNode.splitVal == row.loc['survived']):
                # print("predicted correctly")
                return 1
            else:
                return 0
        #See if row with columnName is not equal to splitVal and follow leftChild
        if(row[rootNode.columnName] != rootNode.splitVal):
            # print("following left")
            return self.predict(rootNode.leftChild, row)
        #Else follow rightChild
        else:
            # print("following right")
            return self.predict(rootNode.rightChild, row)

    #Accuracy function
    def accuracy(self, dataFrame, rootNode):
        correctPredictions = 0
        for index, row in dataFrame.iterrows():
            correctPredictions += self.predict(rootNode, row)
        # print("correctPredictions: %d && len(dataFrame): %d" % (correctPredictions, len(dataFrame)))
        return (correctPredictions / len(dataFrame))

    #Prune function
    def prune(self, rootNode, currentNode, validationAccuracy, validationFrame):
        #Base case: When both leftChild and rightChild are terminal
        if(currentNode.leftChild.columnName == 'survived' and currentNode.rightChild.columnName == 'survived'):
            leftTemp = currentNode.leftChild
            rightTemp = currentNode.rightChild
            currentNode.leftChild = None
            currentNode.rightChild = None
            currentNode.makeTerminalNode()
            currValidationAccuracy = self.accuracy(validationFrame, rootNode)
            if(currValidationAccuracy > validationAccuracy):
                validationAccuracy = currValidationAccuracy
                return self.prune(rootNode, rootNode, validationAccuracy, validationFrame)
            else:
                currentNode.leftChild = leftTemp
                currentNode.rightChild = rightTemp
                return rootNode
        #Prune left
        if(currentNode.leftChild.columnName != 'survived'):
            return self.prune(rootNode, currentNode.leftChild, validationAccuracy, validationFrame)
        #Prune right
        if(currentNode.rightChild.columnName != 'survived'):
             return self.prune(rootNode, currentNode.rightChild, validationAccuracy, validationFrame)




def ID3(____):
	pass


if __name__ == "__main__":
    # parse arguments
    for x in sys.argv:
        pass
        # print('arg: ', x)
    """
    if(len(sys.argv) < 5):
        print("USAGE: not enough arguments passed")
        exit(0)
    """
    model = sys.argv[3]



    #Graph plots
    if(model == "graph"):
        # graphQ1(sys.argv[1], sys.argv[2])
        # graphQ2(sys.argv[1], sys.argv[2])
        graphQ3(sys.argv[1], sys.argv[2])
        exit(0)
	# build decision tree
    trainPercentage = int(sys.argv[4])
    #vanilla tree implementation
    if(model == "vanilla"):
        noDepth = -1
        minSplit = -1
        tree = Tree(sys.argv[1], sys.argv[2])
        validationPercentage = 0
        dataFrame, validationFrame = tree.buildDataFrame(tree.trainFile, trainPercentage, validationPercentage)
        testFrame = tree.buildTestFrame(tree.testFile)
        rootNode = tree.buildTree(dataFrame, noDepth, minSplit)
        rootNode.levelOrder()
    	#predict on testing set & evaluate the testing accuracy
        trainAccuracy = tree.accuracy(dataFrame, rootNode)
        testAccuracy = tree.accuracy(testFrame, rootNode)
        print("Accuracy on training set: %.6f" % (trainAccuracy))
        print("Accuracy on test set: %.6f" % (testAccuracy))
    #maxDepth tree implementation
    if(model == "depth"):
        validationPercentage = int(sys.argv[5])
        depth = int(sys.argv[6])
        minSplit = -1
        depthTree = Tree(sys.argv[1], sys.argv[2])
        dataFrame, validationFrame = depthTree.buildDataFrame(depthTree.trainFile, trainPercentage, validationPercentage)
        testFrame = depthTree.buildTestFrame(depthTree.testFile)
        rootNode = depthTree.buildTree(dataFrame, depth, minSplit)

        #predict on testing set & evaluate the testing accuracy
        trainAccuracy = depthTree.accuracy(dataFrame, rootNode)
        print("Accuracy on training set: %.6f" % (trainAccuracy))
        validationAccuracy = depthTree.accuracy(validationFrame, rootNode)
        print("Accuracy on validation set: %.6f" % (validationAccuracy))
        testAccuracy = depthTree.accuracy(testFrame, rootNode)
        print("Accuracy on test set: %.6f" % (testAccuracy))
    if(model == 'min_split'):
        validationPercentage = int(sys.argv[5])
        minSplit = int(sys.argv[6])
        depth = -1
        minSplitTree = Tree(sys.argv[1], sys.argv[2])
        dataFrame, validationFrame = minSplitTree.buildDataFrame(minSplitTree.trainFile, trainPercentage, validationPercentage)
        testFrame = minSplitTree.buildTestFrame(minSplitTree.testFile)
        rootNode = minSplitTree.buildTree(dataFrame,depth,minSplit)

        #predict on testing set & evaluate the testing accuracy
        trainAccuracy = minSplitTree.accuracy(dataFrame, rootNode)
        print("Accuracy on training set: %.6f" % (trainAccuracy))
        validationAccuracy = minSplitTree.accuracy(validationFrame, rootNode)
        print("Accuracy on validation set: %.6f" % (validationAccuracy))
        testAccuracy = minSplitTree.accuracy(testFrame, rootNode)
        print("Accuracy on test set: %.6f" % (testAccuracy))
    if(model == "prune"):
        noDepth = -1
        noMinSplit = -1
        validationPercentage = int(sys.argv[5])
        pruneTree = Tree(sys.argv[1], sys.argv[2])
        dataFrame, validationFrame = pruneTree.buildDataFrame(pruneTree.trainFile, trainPercentage, validationPercentage)
        testFrame = pruneTree.buildTestFrame(pruneTree.testFile)
        rootNode = pruneTree.buildTree(dataFrame, noMinSplit, noDepth)

        #Predict and get accuracies
        trainAccuracy = pruneTree.accuracy(dataFrame, rootNode)
        print("Accuracy on training set: %.6f" % (trainAccuracy))
        validationAccuracy = pruneTree.accuracy(validationFrame, rootNode)
        #prune tree
        rootNode = pruneTree.prune(rootNode, rootNode, validationAccuracy, validationFrame)
        testAccuracy = pruneTree.accuracy(testFrame, rootNode)
        print("Accuracy on testing set: %.6f" % (testAccuracy))
