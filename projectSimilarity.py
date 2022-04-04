
"""
    AUTHOR : Argyrios Theodoridis 2978
                                        """



import pandas as pd
import copy
import math
import random
import itertools
import csv
#import Gnuplot

from randomPermutation import create_random_permutation
from universalHashFunctions import create_random_hash_function

#filename = input("Give the filename:\n")
#data = pd.read_csv(filename)
data = pd.read_csv("ratings_100users.csv")
df = pd.DataFrame(data, columns = ['userId','movieId','rating','timestamp'])

def userListMaker ():

    userList = {}
    temp = {}
    userMoviesId = []
    previousUser = 1
    for ind in df.index:
        if (df['userId'][ind] == previousUser):
            userMoviesId.append(df['movieId'][ind])
        else:
            temp = {previousUser : userMoviesId}
            userList.update(temp)
            previousUser = previousUser + 1
            userMoviesId = []
            userMoviesId.append(df['movieId'][ind])

    return userList

def movieMapMaker():

    counter = 1
    movieMap = {}
    temp = {}
    for ind in df.index:
        if (df['userId'][ind] == 1):
            temp = {df['movieId'][ind] : counter}
            movieMap.update(temp)
            counter = counter + 1

        else:
            if((movieMap.get(df['movieId'][ind])) == None ):
                temp = {df['movieId'][ind] : counter}
                movieMap.update(temp)
                counter = counter + 1

    return movieMap

def movieListMaker():

    movieList = copy.deepcopy(movieMap)
    for i in movieMap:
        movieList.pop(i)

    for i in userList:
        for value in userList[i] :
            try:
                movieList[value].add(i)
            except KeyError:
                movieList[value] = {i}

    return movieList

""" Dhmioyrgw ta leksika """
userList = userListMaker()
movieMap = movieMapMaker()
movieList = movieListMaker()
"""   ...............     """

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list

def jaccardSimilarity(movieId1, movieId2):
    s1 = set(movieList[movieId1])
    s2 = set(movieList[movieId2])
    Js = ((len(intersection(s1, s2))) / (len(union(s1, s2))))
    #print("Jaccard similarity of s1,s2 = %.2f" %Js

    return Js

def minHash(n):

    K = len(userList)
    N = len(movieMap) - 6
    inf = math.inf
    #signatures = [[inf]*N]*n     lathos tis python 22 wres panw sto pc xamenes :(
    signatures =  [[inf for j in range(N)] for i in range(n)]
    permutations = []
    for i in range(n):
        random_permutation = create_random_permutation(K)
        permutations.append(random_permutation)

    row = 0
    for k in userList:
        for j in  userList[k]:
            col = (movieMap.get(j) -1)
            for i in range(0,n):
                if((permutations[i][row]) < (signatures[i][col])):
                    signatures[i][col] = permutations[i][row]

        row += 1

    return signatures

signatures = minHash(40)

def signatureSimilarity(movieId1, movieId2, signatures, n):

    counter = 0
    for i in range(n):
        if((signatures[i][movieMap[movieId1]]) == (signatures[i][movieMap[movieId2]])):
            counter += 1

    #print("Similarity between movie(%d and %d) = %d%% ." %(movieId1, movieId2, ((counter/n))*100))
    return (counter/n)

#signatureSimilarity(1,3,signatures,40)

def LSH(signatures, s, n, b, r):

    myHashFunction = create_random_hash_function()
    appopriateSize = len(str(len(userList)))         #The number of digits that all numbers must have
    N = 20 #len(signatures[0])                       #Poses tainies na sygrinei
    LshList = []

    for k in range (b):
        buckets = {}
        for j in range(N):
            partOfSignature = []
            for i in range (r):
                distFromGoal = appopriateSize - (len(str(signatures[k * r + i][j])))
                partOfSignature.append( '0'*distFromGoal + str(signatures[k * r + i][j]))

            myVectorInString = ''.join(map(str,partOfSignature))
            vector = int(myVectorInString)
            hashNumber = myHashFunction(vector)
            if(hashNumber not in buckets):
                temp = {hashNumber : j}
                buckets.update(temp)
            else:
                tempList = []
                listOfPairs =  []
                while (buckets.get(hashNumber) != None):
                    num = buckets.pop(hashNumber)            # epistrefei 'h lista 'h arithmo
                    if type(num) is list :
                        for i in range(len(num)):
                            tempList.append(num[i])
                            myPair = [num[i],j]
                            if(myPair not in LshList):
                                LshList.append(myPair)

                    else:
                        if (num == None):
                            break;
                        tempList.append(num)
                        myPair = [num,j]
                        if(myPair not in LshList):
                            LshList.append(myPair)

                tempList.append(j)
                temp = {hashNumber : tempList}
                buckets.update(temp)

    more = n - b*r
    if (more > 0):
        buckets = {}
        for j in range(N):
            partOfSignature = []
            for i in range (b*r, n):
                distFromGoal = appopriateSize - (len(str(signatures[i][j])))
                partOfSignature.append( '0'*distFromGoal + str(signatures[i][j]))

            myVectorInString = ''.join(map(str,partOfSignature))
            vector = int(myVectorInString)
            hashNumber = myHashFunction(vector)
            if(hashNumber not in buckets):
                temp = {hashNumber : j}
                buckets.update(temp)
            else:                               # tha karataw kai mia lista me ta kleidia poy mphkan mesa !
                tempList = []
                listOfPairs =  []
                while (buckets.get(hashNumber) != None):
                    num = buckets.pop(hashNumber)            # epistrefei h lista h arithmo
                    if type(num) is list :
                        for i in range(len(num)):
                            tempList.append(num[i])
                            myPair = [num[i],j]
                            if(myPair not in LshList):
                                LshList.append(myPair)

                    else:
                        if (num == None):
                            break;
                        tempList.append(num)
                        myPair = [num,j]
                        if(myPair not in LshList):
                            LshList.append(myPair)

                tempList.append(j)
                temp = {hashNumber : tempList}
                buckets.update(temp)

    """for i in range (len(LshList)):
        print(LshList[i])
    print("%d different candidate pair " %len(LshList))"""
    return LshList

#LSH(signatures, 0.25, 40, 8, 5)

def experiment1():
    s = 0.25
    counter = 0
    myList = []
    for i in range (len(movieMap)):
        if (movieMap.get(i) != None):
            myList.append(i)
            counter += 1
            if( counter == 20):
                break;

    combinations = []
    allCombinations = list(itertools.permutations(myList, 2))
    for i in range (0, len(allCombinations)):
        maxx = max(allCombinations[i][0],allCombinations[i][1])
        minn = min(allCombinations[i][0],allCombinations[i][1])
        tempList = [minn, maxx]
        if( tempList not in combinations ):
            combinations.append(tempList)

    jacSimList = []
    for i in range (0, len(combinations)):
        jsim = jaccardSimilarity(combinations[i][0], combinations[i][1])
        if (jsim >= s):
            jacSimList.append(combinations[i])

    signatures = minHash(40)
    """for i in range (len(signatures)):
        print(signatures[i])"""

    signSim = [[0 for j in range(len(combinations))] for i in range(8)]
    for n in range(5,45,5):
        for i in range (0, len(combinations)):
            signSim[(n//5) -5][i] = signatureSimilarity(combinations[i][0], combinations[i][1], signatures, n)

    falsePositives = [0 for i in range (8)]
    falseNegatives = [0 for i in range (8)]
    truePositives = [0 for i in range (8)]
    trueNegatives = [0 for i in range (8)]

    for n in range(8):
        for i in range (0, len(combinations)):
            if ((signSim[n][i] >= s) and (combinations[i] not in jacSimList)) :
                falsePositives[n] += 1
            elif((signSim[n][i] < s) and (combinations[i] in jacSimList)) :
                falseNegatives[n] += 1
            elif((signSim[n][i] >= s) and (combinations[i] in jacSimList)) :
                truePositives[n] += 1
            elif((signSim[n][i] < s) and (combinations[i] not in jacSimList)) :
                trueNegatives[n] += 1


    precision = [0 for i in range (8)]
    recall = [0 for i in range (8)]
    f1 = [0 for i in range (8)]
    for i in range(0, 8):
        precision[i] = truePositives[i] /(truePositives[i] + falsePositives[i])
        recall[i] = truePositives[i] /(truePositives[i] + falseNegatives[i])
        f1[i] = recall[i] * precision[i] / (recall[i] + precision[i])


    #to provlima den einai oti oi pinakes den einai nupmy einai ston ypologisth moy, an thelete dokimaste afairontas ta sxolia ston diko sas


    """.........................Gia thn grafikh parastash ...................."""
    """g = Gnuplot.Gnuplot(persist = 0)
    x = [1,2,3,4,5,6]
    pr = g.Data(x,precision,with_='lpts color rgb "blue"')
    g.plot(pr)
    rc = g.Data(x,recall,with_='lpts color rgb "red"')
    g.plot(rc)
    f1p = g.Data(x,f1,with_='lpts color rgb "green"')
    g.plot(f1p)"""
    """........................................................................."""

    print("precisions :", precision)
    print("recalls :", recall)
    print("f1s :", f1)

#experiment1()

def experiment2():
    signatures = minHash(40)
    LSHList = []
    s = 0.25
    counter = 0
    myList = []
    for i in range (len(movieMap)):
        if (movieMap.get(i) != None):
            myList.append(i)
            counter += 1
            if( counter == 20):
                break;

    combinations = []
    allCombinations = list(itertools.permutations(myList, 2))
    for i in range (0, len(allCombinations)):
        maxx = max(allCombinations[i][0],allCombinations[i][1])
        minn = min(allCombinations[i][0],allCombinations[i][1])
        tempList = [minn, maxx]
        if( tempList not in combinations ):
            combinations.append(tempList)

    jacSimList = []
    for i in range (0, len(combinations)):
        jsim = jaccardSimilarity(combinations[i][0], combinations[i][1])
        if (jsim >= s):
            jacSimList.append(combinations[i])

    LSHList.append(LSH(signatures, s, 20, 20, 2))
    LSHList.append(LSH(signatures, s, 20, 10, 4))
    LSHList.append(LSH(signatures, s, 20, 8, 5))
    LSHList.append(LSH(signatures, s, 20, 5, 8))
    LSHList.append(LSH(signatures, s, 20, 4, 10))
    LSHList.append(LSH(signatures, s, 20, 2, 20))

    relativeElements = [0 for i in range (6)]
    falseNegatives = [0 for i in range (6)]
    falsePositives = [0 for i in range (6)]
    for k in range (len(LSHList)):
        for i in range (len(LSHList[k])):
            if(LSHList[k][i] in jacSimList):
                relativeElements[k] += 1
            elif(LSHList[k][i] not in jacSimList):
                falsePositives[k] += 1

        falseNegatives[k] = len(jacSimList) - relativeElements[k]

    print("relativeElements")
    print(relativeElements)
    print("falseNegatives")
    print(falseNegatives)
    print("falsePositives")
    print(falsePositives)

#experiment2()
