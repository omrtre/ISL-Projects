import os
import numpy as np
import pandas as pd
import nums_from_string

from hungarian import linear_sum_assignment
from scipy.spatial.distance import directed_hausdorff

root = '/home/mrrobot/Documents'
data = os.path.join(root, 'correspondence_tests', 'data')

def preprocessing(string):
	lst = [nums_from_string.get_nums(str) for str in string.splitlines()]
	return np.array(lst)


def get_data(file_name, data=data):
	data_frame = pd.read_csv(os.path.join(data, file_name))
	data_frame['Object'] = file_name.split('.csv')[0] + '_' + data_frame['Unnamed: 0'].astype(str)
	df = data_frame[['Object','3D_Bounding_Box']].copy()
	
	if type(df['3D_Bounding_Box'][0]) == str:
		df['3D_Bounding_Box'] = df['3D_Bounding_Box'].map(preprocessing)

	return df


def hausdorff(u, v):
	return directed_hausdorff(u,v)


def hungarian(matrix):
	row_ind, col_ind = linear_sum_assignment(matrix)
	return col_ind, row_ind, matrix[row_ind, col_ind].sum()


def update_correspondence(cols, rows, correspondence, first_file, second_file):
	
	if not correspondence[first_file].isnull().all():
		for i in cols:
			lst_idx = np.where(cols==i)
			mat_idx = correspondence.loc[correspondence[first_file] == i].index
			correspondence.at[mat_idx, second_file] = rows[lst_idx]
	
	return correspondence
