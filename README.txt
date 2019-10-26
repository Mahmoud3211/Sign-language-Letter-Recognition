###
Our problem is that deaf individuals in America have always hard time describing what they want to someone who is not familiar with ASL. So,
the goal is to build a model that can recognize ASL Letters in real time so that it makes it easier for them to communicate in simpler and more efficient way,
 the tasks involved are the 


### Requirements
these libraries must be pre-installed:

numpy
matplotlib
pandas 
os
keras
sklearn
pickle
seaborn
opencv3

### Operation
ASL_Recognition.py is the file where the training process took place it's also availible as python notebook which i recommend to use,
Live Hand Recognition.py is a script that opens webcam and translates the sign letters that are positioned in the specified block

note: make sure that you download the dataset from kaggle at this link (https://www.kaggle.com/grassknoted/asl-alphabet) and place the asl_alphabet_train and asl_alphabet_test
folders in the same directories as these scripts

note: the Local2.h5 and VGG16.h5 are the saved weights of the best trained models these are used in the Live Hand Recognition.py file that uses one of them to 
make predictions and they must remain at the same directory

note: after running the Live Hand Recognition.py file to quit press Enter button


### developed by :
 Mahmoud Nada