# This script will:

# 1)  create the DNN models using one adult speech corpora 

# sh dnn1_Train_and_Save_Exp2.sh

# 2) Test de models with the clinical database (BDsTesteo). Posteriors 
# will be saved to Resultados folder

sh dnn2_Load_and_Test.sh  

# 3) Compute the nasal probability per speaker using Mathad's metric. 
# This will go through all the (kfold) models 

sh dnn3_reportMATHAD.sh

# 4) (Optional) Collapse the  results for  5 kfolds

# python dnn4_promediarResultados_kFolds.py 

# This will create a "MODELNAME+ASICA_resumen.txt" file per speech database and microphone. It will compute also
# the Pearson correlation with the SLP's hypernasality score. The SLP data is kept in the file 
# resultados/resultados_nlace_perceptual_detallado.txt

# 5) Extract the statistics into a single file and save it to resultados/results_exp2.txt 

python dnn5_stats.py


# If everything goes well you will see that the correlation is the same as in our paper. Note however that the
# results need not be identical! 
