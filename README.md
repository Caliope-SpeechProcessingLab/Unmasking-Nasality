# Unmasking-Nasality

This repository contains all data used in Ignacio Moreno-Torres, Andrés Lozano, Rosa Bermúdez, Josué Pino, María Dolores García Méndez, Enrique Nava. Unmasking nasality to assess hypernasality. DOI: to be asigned


To execute both experiment run:

- **experiment1.sh**: This script will run the DNN models using cross-evaluation. If everything is ok, you will find the DNN models in the folder ModelosDNN
In the folder for each model, you will find 
 	- informe.txt (summary of the results for the 5 folds)
 	- in the folder for each fold: a file named figCM+MODELNAME that contains the confussion matrix 

- **experiment2.sh**: This script will do:
  - Create the DNN models using one adult speech corpora. 
  - Test de models with the clinical database (BDsTesteo). Posteriors probabilities will be saved to Resultados folder.
  - Compute the nasal probability per speaker using Mathad's metric. This will go through all the (kfold) models 
  - (Optional) Collapse the  results for  5 kfolds. This will create a "MODELNAME+ASICA_resumen.txt" file per speech database and microphone. It will compute also the Pearson correlation with the SLP's hypernasality score. The SLP data is kept in the file resultados/resultados_nlace_perceptual_detallado.txt
 - Extract the statistics into a single file and save it to resultados/results_exp2.txt 

If everything goes well you will see that the correlation is the same as in our paper. Note however that the results need not be identical! 


The main folders in repository are:

- **BDsTrain** : Train databases for experiment 1 and 2.

- **BDsTesteo** : Test databases for experiment 2 with mouth, nose and monophonic signal.

- **MFFCDistance** : MFCC distance calculation.

- **modelosDNN** : DNN models created during execution.

- **Resultados** : Main results.

