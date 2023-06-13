import os
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt

from utils import *

class Node:
	"""Node class holds information regarding a node with edges on a graph

	An instance of this class represents a node on a graph, which has edges, represented
	as vectors (numpy lists). It has the ability to calculate the angle between two vectors, 
	using the calculateAngle method. Each instance also holds its position
	"""
	def __init__(self, position, vectors = [], angles = []):
		"""Constructor class which is used to create a Node instance
		
		Accepts a mandatory position param and two optional params, vectors and angles.
		
		Args:
			position (1x3 np array): This holds the Cartesian coordinate information for location of this node
			vectors (list of 1x3 np arrays): This holds all vectors (edges) which go from this node to another
			angles (list of ints): This holds all angles between vectors 
		
		"""
		self.position = position  ## np format
		self.vectors = []  ## list of vectors in np format
		self.angles = []  ## list of angles
	def calculateAngle(self, vector1, vector2): 
		"""Calculates the angle between two param vectors
		
		Accepts two mandatory params, vector1 and vector2, then uses the formula acos((v1*v2)/(|v1|*|v2|)) to 
		calculate the angle between the two given vectors
		
		Args:
			vector1 (1x3 np array): The vector information with the parent node as the origin
			vector2 (1x3 np array): The vector information with the parent node as the origin
		
		Returns:
			int: The angle in radians between the two given vectors
		"""	
		return math.acos(np.dot(vector1, vector2) / ( np.linalg.norm(vector1)*np.linalg.norm(vector2) ) ) 
	def calculateAllAngles(self):
		for key, vector1 in enumerate(self.vectors[:-1]): ## Parse through all of the vectors (exlcuding the last one, since by the time it gets to the second to last, all vectors have already been compared to the last one
			for vector2 in self.vectors[(key+1):]: ## Parse through all of the vectors excluding the one from the last line as well as those before it
				self.addAngle(self.calculateAngle(vector1, vector2)) ## add angle from two vectors, this way each pair of vectors have exactly one angle
	def addVector(self, newVector):
		self.vectors.append(newVector)
		
		def sortFn(v):
			return np.linalg.norm(v)
		
		self.vectors.sort(key=sortFn)
	def addAngle(self, newAngle):
		self.angles.append(newAngle)
	def getPosition(self):
		return self.position
	def getAngles(self):
		return self.angles
	def getVectors(self):
		return self.vectors
		
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

def createNodeOrder(listNodes, origin = np.array([0, 0, 0]), initialVector = np.array([1, 0, 0])):
	tempNodeOrder = []
	
	def sortFn(l): ## We are going to sort the list based on the angle, which is the first element in the list created below
		return l[0]
	
	if (origin == np.array([0,0,0])).all(): ## no need to find coordinate if position is already relative to center
		for node in listNodes:
			currentAngleNode = node.calculateAngle(node.position, initialVector)
			tempNodeOrder.append([currentAngleNode, node])
	else:
		for node in listNodes: ## need to calculate coordinate relative to new origin
			currentAngleNode = node.calculateAngle((node.position - origin), initialVector)
			tempNodeOrder.append([currentAngleNode, node])
	
	tempNodeOrder.sort(key=sortFn) ## Sorts the nodes based on the angle
	
	finalNodeOrder = [term[1] for term in tempNodeOrder] ## extract the nodes, now that they are in the correct order. The angle is no longer relevant
	
	return finalNodeOrder

def compareNodes(node1, node2, pw=-1, aw=-1, lw=-1):
	angleDiff = abs(node1.getAngles()[0] - node2.getAngles()[0]) / ((node1.getAngles()[0] + node2.getAngles()[0]) / 2) ## There should only be one angle, since there should only be two edges
	lengthDiff = 0
	positionDiff = 0
	
	node1Vectors = node1.getVectors()
	node2Vectors = node2.getVectors()
	
	for key in range(2):
		#v1Length = np.linalg.norm(node1Vectors[key])
		#v2Length = np.linalg.norm(node2Vectors[key])
		lengthDiff += np.linalg.norm(np.cross(node1Vectors[key], node2Vectors[key]))
		#lengthDiff += abs(v1Length - v2Length) / ((v1Length + v2Length) / 2) ## I thought of some formulas which would be good and I think this works best
		## Finds the difference between two angles, then divides it by the average length of each vector to get how many standard deviations away it is from the average of the two
	
	positionDiff = np.linalg.norm(node1.getPosition() - node2.getPosition())#/15
	
	positionDiffWeight = 1
	angleDiffWeight = 0
	lengthDiffWeight = 0
	
	if (pw != -1):
		positionDiffWeight = pw
	if (aw != -1):
		angleDiffWeight = aw
	if (lw != -1):
		lengthDiffWeight = lw
	
	return angleDiff*angleDiffWeight + lengthDiff*lengthDiffWeight + positionDiff*positionDiffWeight
	
	## At this point, angleDiff is the number of standard deviations away the difference of the angles of the vectors is from the average angle of the vectors
	## lengthDiff is the number of standard deviations away the difference of the lengths of the vectors is from the average length of the vectors

	
def network_relation(pw=-1, aw=-1, lw=-1):

	files = ['data_exp_pos_0-2.83-0+rot_0-0.79-0.csv', 'data_exp_pos_3-2.83--5+rot_0-1.57-0.csv']
	num_people = 5
	
	correspondence = pd.DataFrame(np.random.randint(0,5,size=(num_people, len(files))), columns = [file.split('.csv')[0] for file in files], index=['Person_'+str(x) for x in range(num_people)])
	correspondence[:] = np.nan
	
	for file in files:
		if not file.endswith('.csv'):
			print("Files must be csv files")
			quit()
			
	first_file = files[0]
	second_file = files[1]
	
	df_1 = get_data(first_file)
	df_2 = get_data(second_file)

	if df_1.shape[0] != df_2.shape[0]:
		print('Error')
		if df_1.shape[0] > df_2.shape[0]:
			df_2.loc[len(df_2.index)] = [None, None] 
		else:
			df_1.loc[len(df_1.index)] = [None, None]
			
	temp = pd.DataFrame(np.random.random_sample(size=(num_people, num_people)), columns = df_1['Object'].to_list(), index=df_2['Object'].to_list())

	df_1_nodelist = []
	df_2_nodelist = []
	
	
	for i in range(len(df_1)):
		df_1_nodelist.append(Node(np.array(convert(df_1['Object_Position'][i]))))
	
	for j in range(len(df_2)):
		df_2_nodelist.append(Node(np.array(convert(df_2['Object_Position'][j]))))
		
	nodeOrderList1 = createNodeOrder(df_1_nodelist)
	nodeOrderList2 = createNodeOrder(df_2_nodelist)
	
	list1Length = len(nodeOrderList1)
	list2Length = len(nodeOrderList2)
	
	for key, node in enumerate(nodeOrderList1):
		if key == (list1Length - 1): ## If node is the last in the list, create edge for one before it and the first in the list to come full circle
			node.addVector(nodeOrderList1[key-1].getPosition() - node.getPosition())
			node.addVector(nodeOrderList1[0].getPosition() - node.getPosition())
		elif key == 0: ## If node is the first in the list, create edge for the last in the list and the next node in the list to come full circle
			node.addVector(nodeOrderList1[list1Length - 1].getPosition() - node.getPosition())
			node.addVector(nodeOrderList1[key+1].getPosition() - node.getPosition())
		else: ## If node is somewhere in the middle of the list, create edge for one before it and one after it to come full circle
			node.addVector(nodeOrderList1[key-1].getPosition() - node.getPosition())
			node.addVector(nodeOrderList1[key+1].getPosition() - node.getPosition())
		node.calculateAllAngles()
	
	for key, node in enumerate(nodeOrderList2):
		if key == (list1Length - 1): ## If node is the last in the list, create edge for one before it and the first in the list to come full circle
			node.addVector(nodeOrderList1[key-1].getPosition() - node.getPosition())
			node.addVector(nodeOrderList1[0].getPosition() - node.getPosition())
		elif key == 0: ## If node is the first in the list, create edge for the last in the list and the next node in the list to come full circle
			node.addVector(nodeOrderList1[list1Length - 1].getPosition() - node.getPosition())
			node.addVector(nodeOrderList1[key+1].getPosition() - node.getPosition())
		else: ## If node is somewhere in the middle of the list, create edge for one before it and one after it to come full circle
			node.addVector(nodeOrderList1[key-1].getPosition() - node.getPosition())
			node.addVector(nodeOrderList1[key+1].getPosition() - node.getPosition())
		node.calculateAllAngles()
	
	
	for key1, original1 in enumerate(df_1_nodelist):
		for key2, original2 in enumerate(df_2_nodelist):
			for node1 in nodeOrderList1:
				for node2 in nodeOrderList2:
					if ((original1.getPosition() == node1.getPosition()).all() & (original2.getPosition() == node2.getPosition()).all()):
						difference = compareNodes(node1, node2, pw, aw, lw)

						temp.at[df_2['Object'][key2], df_1['Object'][key1]] = difference
		
	print(temp.to_numpy())
	cols, rows, _ = hungarian(temp.to_numpy())
	#print(first_file.split('.csv')[0], cols)
	#print(second_file.split('.csv')[0], rows)
	#print()
	

	#if n == 0:
	correspondence[first_file.split('.csv')[0]] = cols
	correspondence[second_file.split('.csv')[0]] = rows
	#else: 
	#	correspondence = update_correspondence(cols, rows, correspondence, first_file.split('.csv')[0], second_file.split('.csv')[0])
	#n+=1	

		
	files=files[1:]
	correspondence = correspondence.astype(int)
	print(correspondence)
	return [temp.to_numpy()[cols, rows].sum(), cols]

solutions = []
#for pw in range(0, 11):
#	for aw in range(0, 11):
#		for lw in range(0, 11):
#			if pw + aw + lw == 10:
#				cols = network_relation(pw*0.1, aw*0.1, lw*0.1)[1]
#				correct = 0
#				if (cols == np.array([2, 0, 4, 1, 3])).all():
#					correct = 1
#				solutions.append([round(pw*0.1, 1), round(aw*0.1, 1), round(lw*0.1, 1), round(network_relation(pw*0.1, aw*0.1, lw*0.1)[0], 3), correct])

for pw in range(0,101):
	for aw in range(0, 101):
		if pw + aw == 100:
			cols = network_relation(pw*0.01, aw*0.01, 0)[1]
			correct = 0
			if (cols == np.array([2, 0, 4, 1, 3])).all():
				correct = 1
			solutions.append([round(pw*0.01, 2), round(aw*0.01, 2), 0, round(network_relation(pw*0.01, aw*0.01, 0)[0], 3), correct])


def sortL(l):
	return l[4]
def sortF(l):
	return l[3]
solutions.sort(key=sortF)
solutions.sort(key=sortL)
for term in solutions:
	print(term)
