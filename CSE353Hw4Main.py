import math
import random

import numpy as  np
import matplotlib.pyplot as plt

##Assume that X as a list of lists where x = [[1, x, y], [1, x, y], ...] unless otherwise stated
##simple dot product between the w vector and another vector x
def dotProduct(w, x):
    total = 0
    for i in range(len(w)):
        total += w[i] * x[i]
    return total
##calculates the norm squared of a gradient vector
def normSquared(N, vec):
    total = 0
    for i in range(len(vec)):
        total += vec[i]/N
    return total
##The linear Regression method
def linearReg(x, y):
    transposeX = np.transpose(x);
    return np.linalg.inv(transposeX * x) * transposeX * y
##The logistic regression method
def logisticReg(w, x, y, iterations ,N ,a, e):
    returnedW = [0,0,0]
    for i in range(iterations):
        for j in range(len(w)):
            sigma = logisticRegSigma(w, x, y, N)
            if(normSquared(N, sigma) <= e):
                break
            w[j] += (a * 1/N * sigma[j])
    return w
##helper method to find sigma
def logisticRegSigma(w, x, y, N):
    returned = [0,0,0]
    if(N == len(x)):
        for i in range(N):
            point = x[i]
            sigmaValue = (1/(1 + np.exp(-y[i] * dotProduct(point, w))))
            for j in range(len(point)):
                returned[j] = sigmaValue * (point[j] * y[i])
        return returned
    for i in range(N):
        randomIndex = random.randint(0, len(x)-1)
        point = x[randomIndex]
        sigmaValue = (1 / (1 + np.exp(-y[randomIndex] * dotProduct(point, w))))
        for j in range(len(point)):
            returned[j] = sigmaValue * (point[j] * y[randomIndex])
    return returned

##creates the the list of lists x from the input list of strings for PLA
def getListOfXPLA(input, stride, pointSize):
    tempX = []
    for i in range(len(input)):
        tempX.extend(input[i].split(','))
        tempX = [x.replace('\n', '') for x in tempX]
    for i in range(len(tempX)):
        tempX[i] = float(tempX[i])
    return groupXPLA(tempX, stride, pointSize)

##helper method for getListOfX which groups the sub lists together forming each individual [1, x, y, ...]
def groupXPLA(list, stride, pointSize):
    returnedX = []
    for i in range(stride):
        tempXPoint = []
        for j in range(pointSize):
            tempXPoint.append(list[i + j*stride])
        returnedX.append(tempXPoint)
    return returnedX

##gets the list of y boolean truth values for PLA
def getListOfYPLA(input):
    tempY = []
    for i in range(len(input)):
        tempY.extend(input[i].split(','))
        tempY = [x.replace('\n', '') for x in tempY]
    for i in range(len(tempY)):
        tempY[i] = float(tempY[i])
    return tempY

##creates the the list of lists x from the input list of strings
def getListOfX(input, numOfPoints, pointSize):
    tempX = []
    returnedX = []
    for i in range(len(input)):
        tempX.extend(input[i].split(','))
        tempX = [x.replace('\n', '') for x in tempX]
    for i in range(len(tempX)):
        tempX[i] = float(tempX[i])
    for i in range(numOfPoints):
        tempXPoint = []
        for j in range(pointSize):
            tempXPoint.append(tempX[j + 3 * i ])
        returnedX.append(tempXPoint)
    return returnedX
##gets the list of y boolean truth values
def getListOfY(input):
    tempY = []
    input = [x.replace('\n', '') for x in input]
    for i in range(len(input)):
        tempY.append(float(input[i]))
    return tempY

##counts the number of mistakes
def numMistakes(w, x, y):
    total = 0
    for i in range(len(x)):
        point = x[i]
        dot =  point * w
        if (dot < 0 and y[i] > 0) or (dot > 0 and y[i] < 0):
            total += 1
    return total

##counts the number of mistakes
def numMistakesLists(w, x, y):
    total = 0
    for i in range(len(x)):
        point = x[i]
        dot = dotProduct(w, point)
        if (dot < 0 and y[i] > 0) or (dot > 0 and y[i] < 0):
            total += 1
    return total
##The Pla alg
def PLA(w, x, y):
    while (hasMistake(w, x, y)):
        for i in range(len(x)):
            point = x[i]
            dot = dotProduct(w, point)
            if (dot < 0 and y[i] > 0) or (dot > 0 and y[i] < 0):
                w = newW(w, x, y, i)
    return w
## counts the number of iterations of w
def PLAIterationCounter(w, x, y):
    total = 0
    while (hasMistake(w, x, y)):
        for i in range(len(x)):
            point = x[i]
            dot = dotProduct(w, point)
            if (dot < 0 and y[i] > 0) or (dot > 0 and y[i] < 0):
                w = newW(w, x, y, i)
                total += 1
    return total
##checks for a mistake
def hasMistake(w, x, y):
    for i in range(len(x)):
        point = x[i]
        dot = dotProduct(w, point)
        if (dot < 0 and y[i] > 0) or (dot > 0 and y[i] < 0):
            return True
    return False
##generates the new w for pla
def newW(w, x, y, index):
    returnedW = w
    point = x[index]
    for i in range(len(w)):
        returnedW[i] = w[i] + y[index] * point[i]
    return returnedW

##gets wither the largest x value or smallest value
def getExtremeX(x, largest):
    largestX = 0
    smallestX = 0
    for i in range(len(x)):
        point = x[i]
        if point[1] > largestX:
            largestX = point[1]
        if point[1] < smallestX:
            smallestX = point[1]
    if largest:
        return largestX
    return smallestX

##gets the list of x values with some boolean binary num which is either -1 or 1
def extractListOfX(x, y, num):
    listOfXValues = []
    for i in range(len(x)):
        if y[i] == num:
            point = x[i]
            listOfXValues.append(point[1])
    return listOfXValues

##gets the list of y values with some boolean binary num which is either -1 or 1
def extractListOfY(x, y, num):
    listOfYValues = []
    for i in range(len(x)):
        if y[i] == num:
            point = x[i]
            listOfYValues.append(point[2])
    return listOfYValues
##gets the y values from the w linear line from pla
def getYFromW(w, x):
    return -1 * ((w[0] + (w[1] * x))/w[2])

##question 1
##initialize the lists
print("Question 1: Linear Regression")
xDir = 'X.txt'
yDir = 'Y.txt'
with open(xDir) as f:
    hw4x = f.readlines()
    f.close()
with open(yDir) as f:
    hw4y = f.readlines()
    f.close()
hw4x = getListOfX(hw4x, 40, 3)
hw4y = getListOfY(hw4y)
mhw4x = np.matrix(np.array(hw4x))
mhw4y = np.transpose(np.matrix(np.array(hw4y)))
wlin = linearReg(mhw4x, mhw4y)#the w obtained from linear regression
originalWlin = wlin
tempwlin = originalWlin

numiterPLA = PLAIterationCounter([1,1,1], hw4x, hw4y)#pla without regression
numiterLin = PLAIterationCounter(tempwlin, hw4x, hw4y)#pla with regression

linRegError = numMistakes(wlin, mhw4x, mhw4y)
print("Error of Linear Regression is " + str(linRegError/40))
wPLAfromLin = PLA(wlin, hw4x, hw4y)
wPla = PLA([1,1,1], hw4x, hw4y)

print("Number of iterations without Linear Regression: " + str(numiterPLA))
print("Number of iterations with Linear Regression: " + str(numiterLin))
##graphing of the points and line
listX1 = extractListOfX(hw4x, hw4y, 1)
listXn1 = extractListOfX(hw4x, hw4y, -1)
listY1 = extractListOfY(hw4x, hw4y, 1)
listYn1 = extractListOfY(hw4x, hw4y, -1)
smallestX = getExtremeX(hw4x, True)
largestX = getExtremeX(hw4x, False)

plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Non Linear Regression initialization')
plt.plot(listX1, listY1, 'bo')
plt.plot(listXn1, listYn1, 'rx')
plt.plot([smallestX, largestX], [getYFromW(wPla, smallestX), getYFromW(wPla, largestX)])
temp = np.transpose(wPLAfromLin)

plt.subplot(1, 2, 2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression initialization')
plt.plot(listX1, listY1, 'bo')
plt.plot(listXn1, listYn1, 'rx')
plt.plot([smallestX, largestX], [getYFromW((temp[0].tolist())[0], smallestX), getYFromW((temp[0].tolist())[0], largestX)])
plt.show()

### this is the start of question 2
print("\n")
print("Question 2: Logistic Regression")

wlin = linearReg(mhw4x, mhw4y)
wtemp = np.transpose(wlin).tolist()[0]


originalLogW = logisticReg([1,1,1], hw4x, hw4y, 500 ,40 ,0.005, .1)#logistic regression with 0 vector
linLogW = logisticReg(wtemp, hw4x, hw4y, 500 ,40 ,0.005, .1)#logistic regression with linear regression

errorOriginalLogW = numMistakesLists(originalLogW, hw4x, hw4y)
print("The error of Logistic regression using a vector [1, 1, 1] is: " + str(errorOriginalLogW/100))

errorLogW = numMistakesLists(linLogW, hw4x, hw4y)
print("The error of Logistic regression with using w obtained from linear regression is: " + str(errorLogW/100))
print("Using w obtained by linear regression will give a more accurate input w for linear regression")
#changing the learing step size
wtemp = np.transpose(wlin).tolist()[0]
linLogW2 = logisticReg(wtemp, hw4x, hw4y, 500 ,40 ,.02, .1)
wtemp = np.transpose(wlin).tolist()[0]
linLogW3 = logisticReg(wtemp, hw4x, hw4y, 500 ,40 ,3, .1)
wtemp = np.transpose(wlin).tolist()[0]
linLogW4 = logisticReg(wtemp, hw4x, hw4y, 500 ,40 ,15, .1)
print("w from learning of 0.02 " + str(linLogW2) + " with an error of " + str(numMistakesLists(linLogW2, hw4x, hw4y)/ 100))
print("w from learning of 3 " + str(linLogW3) + " with an error of " + str(numMistakesLists(linLogW3, hw4x, hw4y)/ 100))
print("w from learning of 15 " + str(linLogW4) + " with an error of " + str(numMistakesLists(linLogW4, hw4x, hw4y)/ 100))
print("Once the learning rate goes below 1, the error decreases and as it get bigger so does the values of w")


## starting of question 3
print("\n")
print("Question 3: SGD Logistic Regression")

wtemp = np.transpose(wlin).tolist()[0]
SGDlinLogWOriginal = logisticReg(wtemp, hw4x, hw4y, 500 , 20,0.005, .1)##SGD logistic regression
print("The w obtained from SGD is " + str(SGDlinLogWOriginal) + "compared to the one obtained from the full list " + str(linLogW))
print("The error is " + str(numMistakesLists(SGDlinLogWOriginal, hw4x, hw4y)/ 100))
print("SGD is less computationally expensive compared to regular logistic regression because of the smaller amount of times sigma must loop")
##SGD regression with differing k size
wtemp = np.transpose(wlin).tolist()[0]
SGDlinLogW2 = logisticReg(wtemp, hw4x, hw4y, 500 , 5,0.005, .1)
wtemp = np.transpose(wlin).tolist()[0]
SGDlinLogW3 = logisticReg(wtemp, hw4x, hw4y, 500 , 10,0.005, .1)
wtemp = np.transpose(wlin).tolist()[0]
SGDlinLogW4 = logisticReg(wtemp, hw4x, hw4y, 500 , 30,0.005, .1)
print("w from k size of 5 " + str(SGDlinLogW2) + " with an error of " + str(numMistakesLists(SGDlinLogW2, hw4x, hw4y)/ 100))
print("w from k size of 10 " + str(SGDlinLogW3) + " with an error of " + str(numMistakesLists(SGDlinLogW3, hw4x, hw4y)/ 100))
print("w from k size of 30 " + str(SGDlinLogW4) + " with an error of " + str(numMistakesLists(SGDlinLogW4, hw4x, hw4y)/ 100))
print("As k gets larger each value of w gets smaller in magnitude (gets closer to zero) while the error rate stays relatively the same but does get smaller as k get larger")
