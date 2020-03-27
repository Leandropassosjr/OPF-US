import opfython.math.general as g
import opfython.stream.loader as l
import opfython.stream.parser as p
import opfython.stream.splitter as s
from opfython.models.supervised import SupervisedOPF
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score
import numpy as np
import sys
import os


class DS(object):

	def __init__(self):
		self.opfSup = SupervisedOPF(distance='log_squared_euclidean', pre_computed_distance=None)

	def __classify(self, x_train,y_train, x_valid, y_valid):
		# Training the OPF                
		indexes = np.arange(len(x_train))
		self.opfSup.fit(x_train, y_train,indexes)

		# Prediction of the validation samples
		y_pred,_ = self.opfSup.predict(x_valid)
		y_pred = np.array(y_pred)
		
		# Validation measures for this k nearest neighbors
		accuracy = accuracy_score(y_valid, y_pred)
		recall = recall_score(y_valid, y_pred, pos_label=2) # assuming that 2 is the minority class
		f1 = f1_score(y_valid, y_pred, pos_label=2)
		return accuracy, recall, f1


	def __saveResults(self, X_train,Y_train, X_test, Y_test,  ds,f, approach):

		path = 'Results/down_{}/{}/{}'.format(approach,ds,f)
		if not os.path.exists(path):
			os.makedirs(path)

		results_print=[]
		accuracy, recall, f1 = self.__classify(X_train,Y_train, X_test, Y_test)
		results_print.append([0,accuracy, recall, f1])

		np.savetxt('{}/results.txt'.format(path), results_print, fmt='%d,%.5f,%.5f,%.5f')

	def __saveDataset(self, X_train,Y_train, pathDataset,ds_name):
		DS = np.insert(X_train,len(X_train[0]),Y_train , axis=1)
		np.savetxt('{}/train_{}.txt'.format(pathDataset, ds_name),DS,  fmt='%.5f,'*(len(X_train[0]))+'%d')

	def __computeScore(self, labels, preds, conqs, score):
		
		for i in range(len(labels)):
		    if labels[i]==preds[i]:
		        score[conqs[i]]+=1
		    else:
		        score[conqs[i]]-=1

	def major_negative(self, output, X, Y, path, majority_class, ds, f):
		#1st case: remove samples from majoritary class with negative scores        
		output_majority = output[output[:,1]==majority_class]
		output_majority_negative = output_majority[output_majority[:,2]<0]

		X_train = np.delete(X, output_majority_negative[:,0],0)
		Y_train = np.delete(Y, output_majority_negative[:,0])
		self.__saveDataset(X_train,Y_train, path,'major_negative')
		self.__saveResults(X_train,Y_train, X_test, Y_test, ds,f, 'major_negative')


	def major_neutral(self, output, X, Y, X_test, Y_test, path, majority_class, ds, f):
		#2st case: remove samples from majoritary class with negative or zero scores
		output_majority = output[output[:,1]==majority_class]
		output_majority_neutal = output_majority[output_majority[:,2]<=0]

		X_train = np.delete(X, output_majority_neutal[:,0],0)
		Y_train = np.delete(Y, output_majority_neutal[:,0])
		self.__saveDataset(X_train,Y_train, path,'major_neutral')
		self.__saveResults(X_train,Y_train, X_test, Y_test, ds,f, 'major_neutral')

	def negative(self, output, X, Y, X_test, Y_test, path, majority_class, ds, f):
		#3st case: remove all samples with negative
		output_negatives = output[output[:,2]<0]

		X_train = np.delete(X, output_negatives[:,0],0)
		Y_train = np.delete(Y, output_negatives[:,0])
		self.__saveDataset(X_train,Y_train, path,'negative')
		self.__saveResults(X_train,Y_train, X_test, Y_test, ds,f, 'negative')

	def negatives_major_zero(self, output, X, Y, X_test, Y_test, path, majority_class, ds, f):
		#4st case: remove samples from majoritary class with negative or zero scores 
		# and from minoritary class with negative scores
		output_negatives = output[output[:,2]<0]

		output_negatives_major_zero = output_negatives[output_negatives[:,1]==majority_class]
		output_negatives_major_zero = output_negatives_major_zero[output_negatives_major_zero[:,2]<=0]

		X_train = np.delete(X, output_negatives_major_zero[:,0],0)
		Y_train = np.delete(Y, output_negatives_major_zero[:,0])
		self.__saveDataset(X_train,Y_train, path,'negatives_major_zero')
		self.__saveResults(X_train,Y_train, X_test, Y_test, ds,f, 'negatives_major_zero')

	def balance(self, output, X, Y, X_test, Y_test, path, majority_class, ds, f):
		#5st case: remove samples from majoritary class until balancing the dataset

		# find the number of samples to remove
		n_samples = len(output)
		n_samples_minority = len(output[output[:,1]==2])
		n_samples_to_remove = n_samples - (n_samples_minority*2)

		# sort samples from majority class by score
		output_majority= output[output[:,1]==majority_class]
		order = np.argsort(output_majority[:,2])
		output_majority_ordered = output_majority[order,:]

		# remove samples
		output_to_remove = output_majority_ordered[:n_samples_to_remove,:]
		X_train = np.delete(X, output_to_remove[:,0],0)
		Y_train = np.delete(Y, output_to_remove[:,0])

		# save new dataset and results
		self.__saveDataset(X_train,Y_train, path,'balance')
		self.__saveResults(X_train,Y_train, X_test, Y_test, ds,f, 'balance')

	def __runOPF(self, X_train,y_train,index_train,X_test,y_test,index_test, score):
		# Creates a SupervisedOPF instance
		opf = SupervisedOPF(distance='log_squared_euclidean',
		                    pre_computed_distance=None)

		# Fits training data into the classifier
		opf.fit(X_train, y_train, index_train)
		
		# Predicts new data
		preds, conqs = opf.predict(X_test)
		
		self.__computeScore(y_test, preds, conqs, score)


	
	def run(self, X, Y, indices):
		# Create stratified k-fold subsets
		kfold = 5 # no. of folds
		skf = StratifiedKFold(kfold, shuffle=True,random_state=1)
		skfind = [None] * kfold  # skfind[i][0] -> train indices, skfind[i][1] -> test indices
		cnt = 0
		for index in skf.split(X, Y):
			skfind[cnt] = index
			cnt += 1		
		
		score = np.zeros((5,len(X)))

		for i in range(kfold):
			train_indices = skfind[i][0]   
			test_indices = skfind[i][1]
			X_train = X[train_indices]
			y_train = Y[train_indices]
			index_train = indices[train_indices]
		
		
			X_test = X[test_indices]
			y_test = Y[test_indices]
			index_test = indices[test_indices]
			self.__runOPF(X_train,y_train,index_train,X_test,y_test,index_test, score[i])
		

		output=  np.zeros((len(indices),8))

		score_t = np.transpose(score)
		output[:,0] =indices
		output[:,1] =Y
		output[:,2] =np.sum(score_t,axis=1)
		output[:,3:] =score_t

		return output
