import numpy as np
import csv

def Function(x, N, S):
    return 1.0*(-x*np.log(x)-(1-x)*np.log(1-x)+(1-x)*np.log(N-1)-S*np.log(2))

def FirstDerivative(x, N):
    return 1.0*(np.log(1-x)-np.log(x)-np.log(N-1))

def SecondDerivative(x):
    return 1.0/((x-1)*x)

def CalculateNewApproximation(x, N, S):
    function = Function(x, N, S)
    first_derivative = FirstDerivative(x, N)
    second_derivative = SecondDerivative(x)
    return 1.0*function/(first_derivative-function*second_derivative/(2*first_derivative))

def maximum_predictability(N, S):
    S = round(S, 9)
    if S > round(np.log2(N), 9):
        return "No solutions"
    else:
        if S <= 0.01:
            return 0.999
        else:
            x = 1.0000000001/N
            while abs(Function(x, N, S))>0.00000001:
                x = x - CalculateNewApproximation(x, N, S)
    return round(x, 10)

def getPredictability(filename):
    name_ = filename.split('_')[0]
    # Loading the N^(i) and S_i for CitiBike time series
    filename = name_+"_entropy.csv"
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        #next(reader, None)
        entropies = map(lambda features: [features[0], map(lambda feature: float(feature), features[1:])], reader)
    
    # Calculating predictabilities
    predictabilities = map(lambda entropy: [entropy[0]] + entropy[1] + map(maximum_predictability,
                                           [entropy[1][0]]*3, entropy[1][1:4]), entropies)
    
    # Saving the predictability to a csv
    filename = name_+"_entropy_predictability.csv"

    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, quoting = csv.QUOTE_ALL)
        map(lambda predictability: writer.writerow(predictability), predictabilities)


