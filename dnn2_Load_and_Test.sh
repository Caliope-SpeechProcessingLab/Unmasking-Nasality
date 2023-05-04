# Llama al script que se encarga de cargar un testear una DB con un modelo de DNN guardado. 
# Es preciso indicar el nombre del archivo de resultados. En la carpeta del modelo
# debe haber un archivo de configuración

baseDir=$(pwd)  

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# EVALUACIONES EXPERIMENTO2
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

carpetaModelosDNN="/modelosDNN/"
rutaBDsTest="/BDsTesteo/"
rutaInformes="/resultados/"

# NinosEspana

listaCorpus=(Adultos1b3)
micros=(Nose Mouth)

	for corpus in ${listaCorpus[@]}
	do
		for micro in ${micros[@]}
		do
			bdTest=BDTesteo_Exp2_${micro}.csv
			modeloDNNbase=${micro}_${corpus}/kfold
			nombreInformeTmp=${micro}_${corpus}
			if [ ${micro} = "NoseMouth" ]
				then
					nCanales=2
				else
					nCanales=1
				fi

			# Bucel para los 5 kfold 
			lista=(0 1 2 3 4)

			for kfold in ${lista[@]}
			do
				modeloDNN=${modeloDNNbase}${kfold}
				nombreInforme=${nombreInformeTmp}_kfold${kfold}.csv
				rutaCompletaModeloDNN=${baseDir}${carpetaModelosDNN}${rutaEstosModelosDNN}${modeloDNN}"/"
				rutaCompletaBDTesteo=${baseDir}${rutaBDsTest}${bdTest}
				rutaCompletaInformeResultados=${baseDir}${rutaInformes}${nombreInforme}
				python -m dnn2_Load_and_Test ${rutaCompletaModeloDNN} ${rutaCompletaBDTesteo} ${rutaCompletaInformeResultados} ${numCanales}

			done 
		done 
	done 
