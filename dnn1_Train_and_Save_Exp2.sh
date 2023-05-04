# Script llamar al script python que entrena una DNN y la guarda
# Los modelos se guardan en la ruta indicada abajo en en "saveModelDir". 

# El archivo indcado en "archivoConfigDNN" tiene la  configuración de la RN.
# Se puede usar el valor por defecto. 

# Dentro de la carpeta de modelosDNN, aparecerá una carpeta con el nombre de la BDtrain empleada y:
# - Una copia del archivo de configuración empleado, donde se ha añadido la ruta a la BD de entramiento usada
# - Un gráfico de Matriz de confusión y otro de Loss/accuracy/val-loss/val-accuracy
# - El modelo guardado
# Notas: 
# - Si numClases=3 el modelo se entrena para: oral, nasal, silencio
# - Si numClases=5 el modelo se entrena para: vocal oral, consoante oral, vocal nasal, consonante nasal, silencio

# - En el archivo de configuración por defecto se indica el parámetro EarlyStopping y patience=5, por lo 
# - que a la que empiece a bajar val-loss, el entrenamiento se detiene

# Añado un parámetro "mfcc0 que indica si debe usarse o no ese coeficiente"

##################
### Config común
##################
baseDir=$(pwd) 
trainDir="/BDsTrain/experim2/" # Subcarpeta con BDs de entrenamiento
archivoConfigDNN="config4capas.cfg"
mfcc0=1
numKfold=5
numClases=5
nCanales=1

listaCorpus=(Adultos1b3)
micros=(Nose Mouth)

for corpus in ${listaCorpus[@]}
	do
		for micro in ${micros[@]}
			do
				if [ ${micro} = "NoseMouth" ]
				then
					nCanales=2
				else
					nCanales=1
				fi
				saveModelDir="modelosDNN" # Subcarpeta en la que se guardan los modelos DNN creados
				trainDBcsvFile=${micro}_${corpus}.csv # BD de entrenamiento


				rutaCompletaConfigDNN=${baseDir}"/"${archivoConfigDNN}
				rutaCompletaBDTrain=${baseDir}${trainDir}${trainDBcsvFile}
				python -m dnn1_Train_and_Save ${rutaCompletaBDTrain} ${saveModelDir} ${rutaCompletaConfigDNN} ${numClases} ${numKfold} ${mfcc0} ${nCanales}	
			done
		done
