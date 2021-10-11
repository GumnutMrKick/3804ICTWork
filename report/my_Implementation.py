# imports
import numpy as np
import pandas as pd

# these imports do not make up the creation of the tree
# as required by the task the implementation of the tree itself
# is from scratch without the aid of decision tree packages

# for splitting up the training data
from sklearn.model_selection import train_test_split
# for testing the effectiveness
from sklearn.metrics import accuracy_score

# class definitions
class TreeNode:

	# constructor
	def __init__(self):

		# initialise variables
		self.attribute = None




# the decision tree maker
class DecisionTreeMaker:
	
	# constructor
	def __init__(self, training_file):

		# initialise variables
		self.in_file_name = None
		self.x_test = None
		self.y_test = None
		self.attribute_names = None
		self.attribute_count = None
		self.dataset = None
		self.tree_root = None
		self.branch_count = None
		self.node_count = None

		# record name
		self.in_file_name = training_file

		print("reading in input file...")

		# read in the data from the csv
		csv_filename = self.in_file_name + ".csv"
		data = pd.read_csv(csv_filename)

		# read in the attribute names
		self.attribute_names = data.columns.values

		# split the data into independant and independant
		X = data.iloc[:, :-1].values
		Y = data.iloc[:, -1].values.reshape(-1,1)

		# split up the data into a training set and a test set
		X_train, self.X_test, Y_train, self.Y_test = train_test_split(X, Y, test_size=.1)
		
		# create dataset
		self.dataset = np.concatenate((X_train, Y_train), axis=1)

		# record attribute count
		instances, attributes = X.shape
		
		self.attribute_count = attributes
		self.branch_count = [0] * attributes

		# announce progress
		print("succesfully read in data set with ", instances, " instances and ", (self.attribute_count+1), "attributes")

	# main functions

	# starts off the process of building the tree
	def buildTree(self):

		print("beginning to build the tree...")

		self.tree_root = self.buildBranch(self.dataset)

		total = 0

		# get total branch count
		for branches in self.branch_count:
			
			total+=branches

		print("tree build complete making a tree containing", branches, " and ", self.node_count, " nodes")

	# supporting functions

	# a recurrsive function which builds out the tree from a given node position with using
	# the given data
	def buildBranch(self, dataset, depth):

		# split the dataset into independant and dependant variables
        X, Y = dataset[:,:-1], dataset[:,-1]

		# get instance count
		instance_count, unused = np.shape(X)

		# increase node count
		self.node_count+=1

		# if there are is enough instances to split and the attributes have not run out




# Y_pred = classifier.predict(X_test)

# accuracy_score(Y_test, Y_pred)



	# starts the initial process of building the tree
	# def buildTree(self):
	# 	print("build tree")


	#
	def buildBranch(self):
		print("making the tree")

		self.makeBestSpilt()

	# finds and makes the best split in the given dataset
	# returning all relevant information regarding it
	def makeBestSplit(self):
		
		# initialise mesure of improvement
		print("making best split")
		


	def printTree(self):
		print("print tree")

	def saveTree(self):
		print("save tree")












# beginning of program

# variable definition
in_file = "categorised"

# construct tree maker object
tree_maker = DecisionTreeMaker(in_file)

# build the tree
tree_maker.buildTree()