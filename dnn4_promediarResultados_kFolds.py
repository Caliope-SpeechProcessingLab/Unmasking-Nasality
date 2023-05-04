import os
import csv
import pandas as pd

baseDir=os.getcwd()
inicioInforme=baseDir+"/resultados/"

listaCorpus=["Adultos2","Ninos"]


micros=["Nose_","Mouth_","NoseMouth_"]
serie=["_kfold0","_kfold1","_kfold2","_kfold3","_kfold4"]

for corpus in listaCorpus:

	for micro in micros:

		n=0

		for fold in serie:

			archivo=inicioInforme+micro+corpus+fold+"_ASICA.txt"
			print("Archivo actual:", archivo)
			dfActual=pd.read_csv(archivo, delimiter='\t')
			if n==0:
				df0=dfActual
			elif n==1:
				df1=dfActual
			elif n==2:
				df2=dfActual
			elif n==3:
				df3=dfActual
			elif n==4:
				df4=dfActual
			n=n+1

		df_resumen=dfActual

		columnas=list(df_resumen.columns)

		for col in df_resumen.columns[1:]:	
			col1=df0[col]
			col2=df1[col]
			col3=df2[col]
			col4=df3[col]
			col5=df4[col]

			columnaPromedio=(col1+col2+col3+col4+col5)/5
			df_resumen[col]=columnaPromedio

		nomFinal=inicioInforme+micro+corpus+"_resumen"+"_ASICA.txt"
		print("Nom final: ",nomFinal)


		df_resumen.to_csv(nomFinal, sep ='\t',index=False)
