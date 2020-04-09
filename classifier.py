import pandas as pd
import numpy as np
import math
import csv

def decision_tree():
	filename="DT_Data_CakeVsMuffin_v012_TRAIN.csv" 
	reader = pd.read_csv(filename)
	df = pd.DataFrame(reader, columns = reader.columns) 

	df = df.loc[(df['Proteins'] >= 0) & (df['Proteins'] < 11) & (df['Flour'] >= 0) & (df['Flour'] < 11) & (df['Oils'] >= 0) & (df['Oils'] < 11) &(df['Sugar'] >= 0) & (df['Sugar'] < 11)]
	df = df[623:]
	target ='None'
	with open( "MyClassification.csv", 'w+') as wp:
		wp.write("RecipeType"+"\n" )
		for ind, row in df.iterrows():
			if( row['Flour'] <=4.9 ):
				if( row['Oils'] <=3.4 ):
					if( row['Flour'] <=3.14 ):
						if( row['Proteins'] <=2.65 ):
							target = 'CupCake'
						else:
							target = 'CupCake'
					else:
						if( row['Proteins'] <=2.79 ):
							target = 'Muffin'
						else:
							target = 'Muffin'
				else:
					target = 'CupCake'
			else:
				if( row['Oils'] <=6.59 ):
					target = 'Muffin'
				else:
					if( row['Flour'] <=7.91 ):
						if( row['Sugar'] <=6.34 ):
							target = 'CupCake'
						else:
							target = 'CupCake'
					else:
						target = 'Muffin'
			wp.write(target + "\n" )

if __name__ == '__main__':
	decision_tree()