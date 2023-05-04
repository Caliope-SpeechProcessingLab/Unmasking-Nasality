# This script will run the DNN models using cross-evaluation.
# If everything is ok, you will find the DNN models in the folder ModelosDNN
# In the folder for each model, you will find 
# 	- informe.txt (summary of the results for the 5 folds)
# 	- in the folder for each fold: a file named figCM+MODELNAME that contains the confussion matrix 

sh dnn1_Train_and_Save_Exp1.sh

