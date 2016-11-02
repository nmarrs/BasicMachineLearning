"""
This is a basic machine learning program using Scikit learn libraries. 
This program uses supervised learning to predict the classification of new data. 

Supervised Learning Recipe
Step 1 - Collect training data (examples of the problems to be solved)
Step 2 - Train classifier (Classifier is a box of rules, use learning algorithm to train it)
Step 3 - Make predictions

Apple vs Orange? 
For example, the label 0 could be an apple and the label 1 could be an orange. 
For features, 1 could represent a smooth texture, while 0 represents a  bumpy texture. 
The 140, 130, 150, and 170 features could represent the weight of the object in grams. 
"""

from sklearn import tree 
#using scikit learning libraries

features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1 , 1]
#Training data 

"""
Example training data using description
features = [[140 grams, smooth], [130 grams, smooth], [150 grams, bumpy], [170 grams, bumpy]]
labels = [apple, apple, orange, orange]
"""

clf = tree.DecisionTreeClassifier()
#Creating a Decision Tree Classifier which acts a box of rules
#Think yes / no branching 

clf = clf.fit(features, labels)
#Training algorithm fit (finds patterns in data) from SciKit library 

print (clf.predict([[160, 0]]))
#Making a prediction on new data set 

"""
This prediction will return [1] (Orange) 
The weight of 160 grams falls between the training data weight values associated with the orange label (150-170)
The bumpy texture (0) matches the training data values associated with the orange label (0, 0)
Thus, the program predicts that the object based on new data is an orange. Neato burrito! 
"""

print (clf.predict([[135, 1]]))
#Making a prediction on new data set 

"""
This prediction will return [0] (Apple) 
The weight of 135 grams falls between the training data weight values associated with the apple label (130-140)
The smooth texture (1) matches the training data values associated with the apple label (1, 1)
Thus, the program predicts that the object based on new data is an apple. SciKit rules!  
"""