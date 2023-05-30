import numpy as np
from scipy.spatial import distance
from hungarian import linear_sum_assignment

## Created by Ethan. Ask him questions ##

import pandas as pd
data1 = pd.read_csv('/home/mrrobot/Documents/correspondence_tests/data/data_exp_0x3x0@0.csv')
data2 = pd.read_csv('/home/mrrobot/Documents/correspondence_tests/data/data_exp_4x3x1@45.csv')
data3 = pd.read_csv('/home/mrrobot/Documents/correspondence_tests/data/data_exp_-2x3x0@0.csv')


def convert(list1):   ## converting nasty str to a usable int format
	listFinal = []
	listTemp = []
	neg = [list1[1],list1[13],list1[25]]
	listTemp.extend([list1[2:12],list1[14:24],list1[26:36]])
	for x in range(3):
		if neg[x] == '-':
			listFinal.append(-float(listTemp[x]))
		else:
			listFinal.append(float(listTemp[x]))
	return listFinal

def subtract3(matrix1, matrix2): ## subtracting the values of the matrices
	finalMatrix = []
	tempMatrix = []
	for term1 in matrix1:
		for term2 in matrix2:
			tempMatrix.append(distance.euclidean(term1, term2)) ## This gives the distances between the two XYZ coordintes
		finalMatrix.append(tempMatrix.copy())
		tempMatrix.clear()
	return finalMatrix
	

data1Positions = np.array([convert(data1.Object_Position[x]) for x in range(len(data1.Object_Position))])
data2Positions = np.array([convert(data2.Object_Position[x]) for x in range(len(data2.Object_Position))])
data3Positions = np.array([convert(data3.Object_Position[x]) for x in range(len(data3.Object_Position))])


def getResults2(positions1, positions2):
	cost = np.array(subtract3(positions1, positions2))
	row_ind, col_ind = linear_sum_assignment(cost)
	print('First Dataset: \n' + str(positions1))
	print('\nSecond Dataset: \n' + str(positions2)) 
	print('\nThe Cost Matrix: \n' + str(cost))
	print('\nRelation: \n' + str(row_ind) + '\n' + str(col_ind))
	print('\nConfidence Number (The lower the number, the closer the world coordinates of the same object between the two camera angles): ' + str(cost[row_ind, col_ind].sum()))

def getResults3(positions1, positions2, positions3):
	cost12 = np.array(subtract3(positions1, positions2))
	cost23 = np.array(subtract3(positions2, positions3))
	cost31 = np.array(subtract3(positions3, positions1))
	row_ind12, col_ind12 = linear_sum_assignment(cost12)
	row_ind23, col_ind23 = linear_sum_assignment(cost23)
	row_ind31, col_ind31 = linear_sum_assignment(cost31)
	firstRows = [row_ind12, row_ind23, row_ind31]
	secondRows = [col_ind12, col_ind23, col_ind31]
	confidences = [cost12[row_ind12, col_ind12].sum(),cost23[row_ind23, col_ind23].sum(),cost31[row_ind31, col_ind31].sum()]
	lowest1 = confidences.index(min(confidences))
	print(lowest1)
	firstRelation = np.stack((firstRows[lowest1],secondRows[lowest1]))
	print(firstRelation)
	del confidences[lowest1]
	del firstRows[lowest1]
	del secondRows[lowest1]
	lowest2 = confidences.index(min(confidences))
	print(lowest2)
	secondRelation = np.stack((firstRows[lowest2],secondRows[lowest2]))
	print(secondRelation, '\n')
	del confidences[lowest2]
	del firstRows[lowest2]
	del secondRows[lowest2]
	finalRelation = np.stack((firstRelation[0], firstRelation[1], secondRelation[1]))
	print(finalRelation)
	
	
	
getResults3(data1Positions, data2Positions, data3Positions)
getResults2(data2Positions, data3Positions)


