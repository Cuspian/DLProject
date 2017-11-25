import math
import numpy as np
import csv

def RoundToTens(number):
    mod10 = number % 10
    if mod10 < 5: 
        return number - mod10
    return number + 10 - mod10

def RandomEntropy(list_):
    return math.log(len(set(list_)), 2)  

def containsSublist(list_, sublist):
    for i in xrange(0, len(list_)-len(sublist)+1):
        if list_[i: i+len(sublist)] == sublist:
            return True
    return False

def RealEntropy(list_):
    Shortest_Substring_Length = [1]
    for i in range(1, len(list_)):
        sequences = [list_[i]]  
        count = 1
        while containsSublist(list_[:i], sequences) and i + count <= len(list_) - 1:
            sequences.append(list_[i+count])
            count +=1
        Shortest_Substring_Length.append(len(sequences))
    RealEntropy = math.log(len(list_)) * len(list_) / sum(Shortest_Substring_Length)
    return RealEntropy

def UnCorrelatedEntropy(list_):
    """Computes the Shannon entropy of the elements of list_. 
    Assumes list_ is an array of nonnegative integers
    that have a max value of approximately the number of unique values.
    """
    if list_ is None or len(list_) < 2:
        return 0.
    list_ = np.asarray(list_)
    list_ = list_.flatten()
    counts = np.bincount(list_)
    counts = counts[counts > 0]
    if len(counts) == 1:
        return 0.
    probs = 1.0 * counts / list_.size
    return -np.sum(probs * np.log2(probs))

def getEntropy(filename):
    name_ = filename.split('_')[0]
    # Loading the time series
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
    #next(reader, None)
        timeseries = map(lambda features: map(lambda feature: int(feature), features), reader)

    # Calculating entropies
    entropies = map(lambda ts: 
            [ts] + [len(set(ts))] + [RandomEntropy(ts)] + [UnCorrelatedEntropy(ts)] + [RealEntropy(ts)] + [sum(ts)],
            timeseries)
    
    # Saving the entropies to a csv
    filename = name_+"_entropy.csv"
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, quoting = csv.QUOTE_ALL)
        map(lambda entropy: writer.writerow(entropy), entropies)

