import os
import numpy as np
import pandas as pd

from utils import *

root = '/home/mrrobot/Documents'
data = os.path.join(root, 'correspondence_tests', 'data')


def main():
	#files = os.listdir(data)
	files = ['data_exp_0x3x0@0.csv', 'data_exp_4x3x1@45.csv', 'data_exp_-2x3x0@0.csv', 'data_exp_-2x3x-2@0.csv']
	n=0
	
	correspondence = pd.DataFrame(np.random.randint(0,5,size=(3, 4)), columns = [file.split('.csv')[0] for file in files], index=['Person_0', 'Person_1', 'Person_2'])
	correspondence[:] = np.nan

	for first_file in files:
		if not first_file.endswith('.csv'):
				continue
		for second_file in files:
			if not second_file.endswith('.csv'):
				continue
			
			if first_file == second_file:
				continue
			
			df_1 = get_data(first_file)
			df_2 = get_data(second_file)

			if df_1.shape[0] != df_2.shape[0]:
				print('Error')
				if df_1.shape[0] > df_2.shape[0]:
					df_2.loc[len(df_2.index)] = [None, None] 
				else:
					df_1.loc[len(df_1.index)] = [None, None] 

			temp = pd.DataFrame(np.random.random_sample(size=(3, 3)), columns = df_1['Object'].to_list(), index=df_2['Object'].to_list())

			for i in range(0,len(df_1)):
				for j in range(0,len(df_2)):
					u = df_1['3D_Bounding_Box'][i]
					v = df_2['3D_Bounding_Box'][j]
					
					dist = hausdorff(u, v)

					temp.at[df_2['Object'][j], df_1['Object'][i]] = dist[0]

			#print(temp)
			cols, rows, _ = hungarian(temp.to_numpy())
			#print(first_file.split('.csv')[0], cols)
			#print(second_file.split('.csv')[0], rows)
			#print()
			

			if n == 0:
				correspondence[first_file.split('.csv')[0]] = cols
				correspondence[second_file.split('.csv')[0]] = rows
			else: 
				correspondence = update_correspondence(cols, rows, correspondence, first_file.split('.csv')[0], second_file.split('.csv')[0])
			n+=1	

				
		files=files[1:]
	correspondence = correspondence.astype(int)
	print(correspondence)

		
main()
