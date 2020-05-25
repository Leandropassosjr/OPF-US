import opfython.math.general as g
import opfython.stream.loader as l
import opfython.stream.parser as p
import opfython.stream.splitter as s

from underSampling_OPF import US
import os

import numpy as np
import sys

import logging
logging.disable(sys.maxsize)
np.set_printoptions(threshold=sys.maxsize)


datasets = ['indian_liver','secom','seismic_bumps', 'spam','vertebral_column','wilt']
folds = np.arange(1,21)

usOPF = US('ResultsGeneral')


for dsds in range(len(datasets)):
    ds = datasets[dsds]
    for ff in range(len(folds)): 
        f = folds[ff]
        train = np.loadtxt('data/{}/{}/train.txt'.format(ds,f),delimiter=',', dtype=np.float32)
        valid = np.loadtxt('data/{}/{}/valid.txt'.format(ds,f),delimiter=',', dtype=np.float32)
        test = np.loadtxt('data/{}/{}/test.txt'.format(ds,f),delimiter=',', dtype=np.float32)
        
        concat = np.concatenate((train, valid))
        X = concat[:,:-1]
        Y = concat[:,-1].astype(np.int) 
        indices = np.arange(len(X))
        
        output = usOPF.run(X, Y, indices)
        
        X = X[:len(train),...]
        Y = Y[:len(train),...]
        output = output[:len(train),...]
        
        X_test = test[:,:-1]
        Y_test = test[:,-1].astype(np.int) 
        
        
        pathDataset = 'data/{}/{}'.format(ds,f)
        if not os.path.exists(pathDataset):
            os.makedirs(pathDataset)  
            
            
        
        #1st case: remove samples from majoritary class with negative scores      
        usOPF.major_negative( output, X, Y, X_test, Y_test,pathDataset, 1, ds,f,2)

        #2st case: remove samples from majoritary class with negative or zero scores
        usOPF.major_neutral( output, X, Y, X_test, Y_test,pathDataset, 1, ds,f,2)   
        
        #3st case: remove all samples with negative
        usOPF.negative( output, X, Y, X_test, Y_test,pathDataset, 1, ds,f,2)
        
        #4st case: remove samples from majoritary class with negative or zero scores 
            # and from minoritary class with negative scores
        usOPF.negatives_major_zero( output, X, Y,X_test, Y_test, pathDataset, 1, ds,f,2)
        
        #5st case: remove samples from majoritary class until balancing the dataset
        usOPF.balance( output, X, Y, X_test, Y_test, pathDataset, 1, ds,f,2)
        
        
