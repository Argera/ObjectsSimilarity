import random
from operator import itemgetter, attrgetter
from universalHashFunctions import create_random_hash_function

# INPUT:  positive integer K
# OUTPUT: a random permutation of the list [1,2,3,...,K]
def create_random_permutation(K):

    myHashFunction = create_random_hash_function()

    hashList = []             # stores pairs (i , H(i) )...
    randomPermutation = []    # stores the permutation of [1,2,...,K]

    for i in range(0,K):
        j=int(myHashFunction(i))
        hashList.append( (i,j) )

    # sort the hashList by second argument of the pairs...
    sortedHashList = sorted( hashList, key=itemgetter(1) )

    for i in range(0,len(sortedHashList)):
        randomPermutation.append(1 + sortedHashList[i][0])

    return randomPermutation

# usage examples...
permutedList = create_random_permutation(100)
