# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 19:49:29 2018

@author: Ranakrc
"""
import numpy as np
import itertools
import re
import collections
import scipy.stats
import random


def testing(test, mu, var, trans, prior, n, coeff, varNoise):
    
    #generate x from I
    tempTest = (n - 1) * '0' + test
    x = []
    for i in range(n - 1, len(tempTest)):
        total = 0
        for j in range(n):
            element = int(tempTest[i - j]) * coeff[j]
            total += element
        total += np.random.normal(0, varNoise, 1)[0]
        x.append(total)
    
    globalArrProb = []
    globalArrPar = []
    arrProb = []
    arrPar = []
    
    #initialize the first row of globalArrProb & globalArrPar
    for i in range(len(mu)):
        val = prior[i] * scipy.stats.norm(mu[i], np.sqrt(var[i])).pdf(x[0])
        arrProb.append(val)
        arrPar.append(-1)
    globalArrProb.append(arrProb)
    globalArrPar.append(arrPar)
    
    #Markov Model
    for i in range(1, len(x)):
        arrProb = []
        arrPar = []
        for j in range(len(mu)):
            #initialize the first and second half pair of nodes
            if(j == 0 or j == int(len(mu)/2)):
                index1 = 0
                index2 = 1
            #when to use the first and the second half pair of nodes
            if(j >= 0 and j < int(len(mu)/2)):
                col = 0
            else:
                col = 1
            
            val1 = globalArrProb[i - 1][index1] * trans[index1][col] * scipy.stats.norm(mu[j], np.sqrt(var[j])).pdf(x[i])
            val2 = globalArrProb[i - 1][index2] * trans[index2][col] * scipy.stats.norm(mu[j], np.sqrt(var[j])).pdf(x[i])
            if(val1 >= val2):
                arrProb.append(val1)
                arrPar.append(index1)
            else:
                arrProb.append(val2)
                arrPar.append(index2)
            
            index1 += 2
            index2 += 2
        
        globalArrProb.append(arrProb)
        globalArrPar.append(arrPar)


    classList = []
    classVal = globalArrProb[len(x) - 1].index(max(globalArrProb[len(x) - 1]))
    classList.append(classVal)
    cnt = 0
    
    #backtracking
    for i in range(len(x) - 1, 0, -1):
        idx = classList[cnt]
        classVal = globalArrPar[i][idx]
        classList.append(classVal)
        cnt += 1
    classList.reverse()
    
    #building the predicted string
    result = ''
    for i in range(len(classList)):
        if(classList[i] < int(len(mu)/2)):
            result += '0'
        else:
            result += '1'
    
    #computing accuracy
    right = 0;
    for i in range(len(classList)):
        if(result[i] == test[i]):
            right += 1
    
    accuracy = (right/len(test)) * 100;
    return result, accuracy
    

def findDistribution(classes, count, n, coeff, varNoise):

    mu = []
    var = []
    x_k = []
    
    #find the sum which is the same for each class
    for i in range(len(count)):
        total = 0
        for j in range(n):
            element = int(classes[i][j]) * coeff[j]
            total += element
        x_k.append(total)
            
    #include the noise for each sample of each class
    for i in range(len(count)):
        x = x_k[i] + np.random.normal(0, varNoise, count[i])
        variance = np.var(x)
        var.append(variance)
        average = np.mean(x)
        mu.append(average)
        
    return mu, var

    
def main():
    
    config = open("config.txt", "r")
    trainFile = open("train.txt", "r")
    train = trainFile.read()
    testFile = open("test.txt", "r")
    test = testFile.read()
    
    n, l = [int(x) for x in next(config).split()] 
    coeff = []
    for x in next(config).split():
        coeff.append(float(x))
    varNoise = [float(x) for x in next(config).split()]
    
    #generate classes
    classes = ["".join(seq) for seq in itertools.product("01", repeat = n)] 
    
    #find the number of occurences of each substring
    count = []
    for i in range(len(classes)):
        s = '(' + '?' + '=' + classes[i][::-1] + ')' 
        count.append(len(re.findall(s, train)))
    
    #find the prior probabilities
    total = sum(count)
    priorProb = [x/total for x in count]
    print(priorProb)
    
    #find the conditional probabilities
    indexList = []
    for i in range(len(classes)):
        s = '(' + '?' + '=' + classes[i][::-1] + ')'
        arr = [m.start() for m in re.finditer(s, train)]
        indexList.append(arr)
    
    val1 = 0
    transProb = []
    for i in range(len(classes)):
        print(i)
        new_list = [x+1 for x in indexList[i]]
        count1 = len(list(set(indexList[val1]).intersection(new_list)))
        count2 = len(new_list) - count1
        print(count1)
        print(count2)
        tempList = []
        total = count1 + count2
        tempList.append(count1 / total)
        tempList.append(count2 / total)
        transProb.append(tempList)
        if (i%2 == 1):
            val1 += 1
    print(transProb)
    
    #find the training parameters mu and sigma
    mu, var = findDistribution(classes, count, n, coeff, varNoise)
    print(mu)
    
    #testing
    result, accuracy = testing(test, mu, var, transProb, priorProb, n, coeff, varNoise) 
    
    #evaluation
    print(test)
    print(result)
    print(accuracy)
    
    #closing files
    config.close()
    trainFile.close()
    testFile.close()
    
if __name__== "__main__":
    np.random.seed(3)
    main()