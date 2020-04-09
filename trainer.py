# __authot__ : "Pooja Kamble"
# __version__: "1.0"


# Program generates a decicion tree on a given training data, and check itself against the same data.

import pandas as pd
import csv
import numpy as np
import math
flag = False

# returns the majority class from the data, 
# if there is a tie then it will return the first foudn class with max freq
def get_majority_class(df, target):
	class_freq = df.groupby([target]).size()

	max = -99
	majority = None
	for key, val in class_freq.items():
		if val > max:
			majority = key
			max = val
	return majority


# checks if all the elements in the node belong to the same class
def stopping_criteria(df, majority_class, target, depth):
	if depth >= 10:
		return True

	if(len(df)<= 23):
		return True
	
	if len(df) <= 23:
		return True

	majority_class_portion = len((df[df[target]==majority_class]))/len(df)*100
	if majority_class_portion >= 90:
		return True
	
	return False

# trains a decision tree on given data
def decision_tree_trainer(df, target, depth):
	
	decision_tree = {}
	print(depth, "..."*depth)

	# set the majority class to return
	majority_class = get_majority_class(df, target)

	# If stopping criteria met, return the majority class
	if(stopping_criteria(df, majority_class, target, depth)):
		print("\t\t\tFound leaf node")
		return majority_class, decision_tree

	# If stopping condition not yet met:
	else:
		# call the submodule to get the best attribute with best threshold
		best_attr, best_threshold = generate_oneR(df, target)
		decision_tree['root'] = (best_attr, best_threshold)

		# Split data on found split point
		print("\n\tSplitting data on: ", best_attr, best_threshold)
		left = df.loc[(df[best_attr] <= best_threshold)]
		right = df.loc[(df[best_attr] > best_threshold)]
		
		# after splitting, perform the algorithm recursively on left split and right split
		attr_left = None
		left_subtree = None
		attr_left, left_subtree = decision_tree_trainer(left, target, depth+1)		
		print("\tLeft on recursion...")
		if(len(left_subtree) <= 0):
			decision_tree['left'] = attr_left
		else:
			decision_tree['left'] = left_subtree
		
		print("\tRight on recursion...")
		attr_right, right_subtree= decision_tree_trainer(right, target, depth+1)
		if(len(right_subtree) <= 0):
			decision_tree['right'] = attr_right
		else:
			decision_tree['right'] = right_subtree
		
		# return the majority class and decision tree
		return majority_class, decision_tree


# The function checks all attributes-all threshold and returns the best attribute with best threshold
def generate_oneR(df, target_variable):
	# creates frequency matrix for every value of every attribute
	freq_matrix = create_freq_matrix(df, target_variable)
	
	# Calls the function to get the best attribute for prediction of target
	best_attr, best_split_threshold, entropy = calculate_best_attr(df,freq_matrix, target_variable)

	return best_attr, round(best_split_threshold,2)

# This is a helper funciton to generate freq matrix, it returns the target frequency for given dataframe
def helper_freq_matrix(df, target_variable):

	target_freq = {'CupCake': 0, 'Muffin':0}
	freq = df.groupby([target_variable]).size()

	for key, val in freq.items():
		ind = key
		val = val
		cnt = val
		if key not in target_freq.keys():
			target_freq[key] = 0
		# if val not in target_freq.keys():
		# 	target_freq[ind][val] = 0
		target_freq[key] = cnt
	return target_freq


# returns the frequency matrix for all unique attrinute value for target
def create_freq_matrix(df, target_variable):

	freq_matrix = {}
	
	# for all the attributes from 1 to #of attributes:
	# for all unique calues generate the target frequency table
	for attr in df.columns[:-1]:
		freq_matrix[attr] = {}

		for val in df[attr]:
			# le: less than , gt: greater than
			freq_matrix[attr][val] = {'le': {}, 'gt': {}, 'le_cnt': 0, 'gt_cnt':0}

			# split data less than and greater than attr-value
			df_le = df.loc[(df[attr] <= val)]
			df_gt = df.loc[(df[attr] > val)]

			# genereate freq matrix for both splits
			freq_matrix[attr][val]['le'] = helper_freq_matrix(df_le, target_variable)
			freq_matrix[attr][val]['gt'] = helper_freq_matrix(df_gt, target_variable)
			freq_matrix[attr][val]['le_cnt'] = len(df_le)
			freq_matrix[attr][val]['gt_cnt'] = len(df_gt)
	return freq_matrix


# Tie breaker(aims towards(50/50))
def tie_breaker(df, x_attr, x_threshold, y_attr, y_threshold):

	# returns which attribute better split the data (aims towards(50/50))
	x = df.loc[(df[x_attr] <= x_threshold)]
	y = df.loc[(df[y_attr] <y_threshold)]
	x_split = (len(x)/len(df)*100)
	y_split = (len(y)/len(df)*100)

	print("There is a tie, breaking tie....")

	if((x_split%50) < (y_split%50)):
		return x_attr, x_threshold
	else:
		return y_attr, y_threshold


# finds out the best attribute and best threshold with the least entropy amongst all
# returns the best split
def calculate_best_attr(df, class_freq_dict, target_variable):
	
	# setting best split to initial setup
	best_attr = None
	best_split_threshold = None
	least_entropy_so_far = 999
	
	# for all attributes 1 to #of attrinutes, find the best split
	for keys, val in class_freq_dict.items():
		if(keys == target_variable):
			continue
		if(val.keys()):
			# get a thershold with a least entropy
			least_entropy, threshold = get_entropy_and_threshold(val)

			# If tie
			if(least_entropy == least_entropy_so_far):
				best_attr,best_split_threshold =  tie_breaker(df, best_attr, best_split_threshold, keys, threshold)
				if(best_attr == keys):
					least_entropy_so_far = least_entropy

			# else update the best split
			elif( least_entropy < least_entropy_so_far):
				least_entropy_so_far = least_entropy
				best_attr = keys
				best_split_threshold = threshold

	return best_attr, best_split_threshold, least_entropy

# calculate the entropy of split
def get_entropy_for_split(x,y):
	
	if max(x,y) < 5:
		return 9999
	entropy = 0
	total = x+y

	if total==0:
		return 9999
	try:
		if(x==0):
			entropy += 0
		else:
			entropy += (-1 * ((x/total)* (math.log2((x/total)))))
		if(y==0):
			entropy += 0
		else:
			entropy += (-1 * ((y/total)* (math.log2((y/total)))))
	except:
		print("Exception while calculating entropy")
		print(((x/total)* (math.log2((x/total)))))
		print(((y/total)* (math.log2((y/total)))))
	return entropy

# checks for the best threshold with a least entropy
def get_entropy_and_threshold(each_val_target_freq):
	least_entropy = 999
	best_threshold = None

	for each_val, target_freq in each_val_target_freq.items():

		total_rows_this_val = target_freq['le_cnt']+target_freq['gt_cnt']

		le_prob = target_freq['le_cnt']/total_rows_this_val
		gt_prob = target_freq['gt_cnt']/total_rows_this_val

		# get the entropy of split by current value:each_val
		split_entropy_le = le_prob * get_entropy_for_split(target_freq['le']['CupCake'], target_freq['le']['Muffin'])
		split_entropy_gt = gt_prob * get_entropy_for_split(target_freq['gt']['CupCake'], target_freq['gt']['Muffin'])
		
		# total entropy for value
		split_entropy_total = split_entropy_le + split_entropy_gt 
		
		# update the best_split
		if split_entropy_total < least_entropy:
			least_entropy = split_entropy_total
			best_threshold = each_val
	return least_entropy, best_threshold


# It genereates a python program with decision rules and predict target for training data
def rule_emitter(decision_tree, target, majority_class):
	
	# writine decision rules in .py file
	file_writer = "classifier.py"
	f = open(file_writer,'+w')
	
	# importing python libraries
	f.write("import pandas as pd" +"\n" + "import numpy as np"+"\n"+"import math"+"\n"+ "import csv" "\n\n")
	f.write("def decision_tree():\n\t")
	
	# Filename on which decision tree needs to be run
	filename = "DT_Data_CakeVsMuffin_v012_TRAIN.csv"
	
	f.write("filename"+ "=\""+filename+"\" \n\t")
	f.write("reader = pd.read_csv(filename)\n\t")
	f.write("df = pd.DataFrame(reader, columns = reader.columns) \n\n\t")
	f.write("df = df.loc[(df['Proteins'] >= 0) & (df['Proteins'] < 11) & (df['Flour'] >= 0) & (df['Flour'] < 11) & (df['Oils'] >= 0) & (df['Oils'] < 11) &(df['Sugar'] >= 0) & (df['Sugar'] < 11)]\n")
	f.write("\tdf = df[623:]\n")

	# Result file for target prediction
	file_result = "MyClassification.csv"

	f.write("\ttarget =\'"+ str(majority_class)+"\'\n")
	f.write("\twith open( \""+file_result+"\", 'w+') as wp:\n")
	f.write("\t\twp.write(\""+target +"\"+\"\\n\" )")
	
	# create IF_ELSE statements from decision tree created
	rule = create_rule(decision_tree, 3)
	
	# for all attributes, run the decision rule generated
	f.write("\n\t\tfor ind, row in df.iterrows():\n")
	f.write(rule)
	f.write("\t\t\twp.write(target + \"\\n\" )")
	
	# main
	f.write("\n\nif __name__ == '__main__':\n\tdecision_tree()")
	


# This is wlll generate IF_ELSE rules based on the decision tree which is created
def create_rule(node,depth, else_flag = False):
	
	rule = ""
	
	# Preorder traversal: root-left-right
	if(isinstance(node,dict)):
		if(else_flag):
			rule += "\t"*(depth-1) +"else:\n"+"\t"*depth+"if( row[\'" + node['root'][0]+"\'] <="+ str(node['root'][1])+" ):\n"
		else:
			rule += "\t"*depth+"if( row[\'" + node['root'][0]+"\'] <="+ str(node['root'][1])+" ):\n"
		
		# recursion on left
		rule+= create_rule(node['left'], depth+1)
		
		# recursion on right
		rule+= create_rule(node['right'], depth+1, True)
		
		return rule
	# leaf node
	else:
		if(else_flag):
			rule += "\t"*(depth-1) +"else:\n"+"\t"*depth+ "target = \'" +str(node)+"\'\n"
		else:
			rule += "\t"*depth + "target = \'" +str(node)+"\'\n"
		return rule


# given a result file and reference df dataframe, this will return accuracy of predicted results
def calculate_accuracy(resultfile, target_variable, df):

	df = df.loc[(df['Proteins'] >= 0) & (df['Proteins'] < 11) & 
					(df['Flour'] >= 0) & (df['Flour'] < 11) & 
					(df['Oils'] >= 0) & (df['Oils'] < 11) &
					(df['Sugar'] >= 0) & (df['Sugar'] < 11)]
	
	true_positives = 0
	true_negatives = 0
	

	reader1 = pd.read_csv(resultfile)

	# create dataframe using read csv and csv headers, 
	# this will allow easier operations on data
	df2 = pd.DataFrame(reader1, columns = reader1.columns)

	# compares result target value with actual value from df
	ind1 = 0
	for ind, items in df[target_variable].iteritems():
		if(items == df2[target_variable][ind1]):
			true_positives+=1
		else:
			true_negatives += 1
		ind1+= 1

	accuracy = (true_positives/(true_positives+true_negatives)*100)
	return round(accuracy,2)

# main function 
def main():
	print("***Decision Tree Menu***")
	print("1. Train decision tree")
	print("2. Generate decision rules python file")
	print("3. Run decision tree on data")
	print("4. Calculte accuracy")
	print("5. Quit")
	choice = input("Please enter your choice: ")
	

	ch = int(choice)
	decision_tree = {}

	# while user exits
	while(ch!= 5):

		target = 'RecipeType'
		majority_class = None
		train_file = "DT_Data_CakeVsMuffin_v012_TRAIN.csv"

		# create dataframe using read csv and csv headers, 
		# this will allow easier operations on data
		reader = pd.read_csv(train_file)
		df = pd.DataFrame(reader, columns = reader.columns)

		# filter out out-of-range values:
		# range: 0-10
		df = df.loc[(df['Proteins'] >= 0) & (df['Proteins'] < 11) & 
					(df['Flour'] >= 0) & (df['Flour'] < 11) & 
					(df['Oils'] >= 0) & (df['Oils'] < 11) &
					(df['Sugar'] >= 0) & (df['Sugar'] < 11)]
		
		# create a decision tree
		if(ch == 1):
			print("Creation of Decision tree in progress...")
			majority_class, decision_tree = decision_tree_trainer(df[:623], target, 0)
			flag = True
			print("\n\nDecision tree generated:\n\t\t",decision_tree,"\n\n")

		# genereate deciion rule file from decision tree made
		elif(ch==2):
			rule_emitter(decision_tree, target, majority_class)
			print("\n\tclassifier.py successfully generated..!")

		# Run the decision tree over training data
		elif(ch == 3):
			if (flag):
				print("\n\n\tPlease execute the following command to run decision tree on the data,\n\t\t come back to check the accuracy!")
				print("\n\tpython3 classifier.py\n")
				exit(0)
			else:
				print("Please train decision tree first!!")
		
		# check for accuracy of the decision tree 
		elif(ch == 4):	
			file_result = "MyClassification.csv"
			print("\n\t\tAccuracy for decision tree: ",calculate_accuracy(file_result, target, df[623:]),"%\n")

		# loop till exit
		print("***Decision Tree Menu***")
		print("1. Train decision tree")
		print("2. Generate decision rules python file")
		print("3. Run decision tree on data")
		print("4. Calculte accuracy")
		print("5. Quit")
		ch = int(input("Please enter your choice: "))

# starting point: calls the main function
if __name__ == '__main__':
	main()