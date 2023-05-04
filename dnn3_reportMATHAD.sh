
# Este script tomas archivos .csv con las probabilidades posteriores de una DNN de Asica y crea 
# un nuevo informe con la siguiente información:  

# 1) T5_Mathad: Ratio de nasales (calculado según Mathad et al. (2021) en la Tarea T5  
# 2) T6_Mathad: como arriba pero para la tarea T6  
# 3) Mathad: como arriba pero en todos los enunciados de T5 y T6

# Además, si el parámetro hacerCorrelacion vale 1, se hará una correlación de Pearson entre estas columnas
# y las correspondientes del archivo /Resultados/nasalance.csv

# Tiene dos bucles, uno para indicar los micrófonos, y otro para indicar los kfold a los que se aplica

# El resultado se guarda en un informe llamado con el Parámetro 1 más "_ASICA"

# Este script bash llama a dnn3_reportMathad.py con estos parámetros:

# 1. rutaRelInformeOriginal: nombre del .csv 

# 2. targetUtt: qué enunciados se van a analizar

# 3. Hacercorrelación: 1 = sí se calcula la correlación. 0 = No se calcula. 

# Como los modelos se han creado 5 veces, la misma evaluación se realiza 5 veces (una por kfold)

targetUtt="ASICA" # o cualquier otro enunciado ASICA
hacerCorrelacion=1

listaCorpus=(Adultos1b3)
micros=(Nose Mouth)

# Subcarpeta en la que están los enunciados
rutaRelComun="/resultados/"


for corpus in ${listaCorpus[@]}

	do

	for micro in ${micros[@]}
		do
		informe=${micro}_${corpus}
		kfolds=(0 1 2 3 4) # cada número corresponde a un kfold
		# Bucle con asignación de variables y llamada a Python (NO MODIFICAR)
		for kfold in ${kfolds[@]}
			do
			rutaRelInformeOriginal=${rutaRelComun}${informe}_kfold${kfold}.csv
			python -m dnn3_reportMATHAD ${rutaRelInformeOriginal} ${targetUtt} ${hacerCorrelacion}
			done
		done
	done