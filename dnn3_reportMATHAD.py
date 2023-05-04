# Este scrip se describe en dnn_reportASICA.sh

import os, sys
import pandas as pd
import numpy as np
import math
from scipy.stats import pearsonr

def suma(lista1,lista2):
	if len(lista1)!=len(lista2):
		print("Error: las listas son diferentes")
		exit()
	else:
		listaSuma=[]
		for i in range(len(lista1)):
			listaSuma.append(lista1[i]+lista2[i])
	return(listaSuma)


# Calcula el número de segmentos nasales en un enunciado. Lo que equivale a contar en una serie
# de N probabiliades (0 a 1) cuantas subseries de tamaño minimoVentanas al menos tienen minimaProbabilidadNasalidad
def nasalesEnEnunciado(listaPNasal,minimoVentanasNas,minimaProbabibNasalidad,espacioEntreNasales):
	# Asumo tres estados posibles
	# 0: no ha aparecido ninguna ventana con prob superior a mínima
	# 1: ha aparecido entre 1 y minVentanasNas con prob > mínima
	# 2: ya hemos pasado minVentanasNas con prob > mínima. Tras N ventanas en este estado pasamos a 0

	estado=0  
	numCN=0
	numVentanasNas=0
	for probabVentana in listaPNasal:
		if estado==0:
			if probabVentana>minimaProbabibNasalidad:
				estado=1
				numVentanasNas=1
		elif estado==1:
			if probabVentana>minimaProbabibNasalidad:
				if numVentanasNas>=minimoVentanasNas:
					numCN=numCN+1
					estado=2
					numVentanasNas=0
				else:
					numVentanasNas=numVentanasNas+1
		elif estado==2:
			if numVentanasNas>espacioEntreNasales:
				estado=0
				numVentanasNas=0
			else:
				numVentanasNas=numVentanasNas+1
	return numCN


# Suma dos listas obteniendo una tercera lista

def sumaListas(lista1,lista2):

	if len(lista1)!= len(lista2):
		print("Error: las listas tienen tamaño diferente")
		exit()

	l3=[]
	for i in range(len(lista1)):
		l3.append(lista1[i]+lista2[i])

	return l3

# suma los elementos de una lista, pasándolos previamente a int/float
def sumaStr(tipo,t2):
	suma=0
	for i in t2:
		if tipo=="i":
			suma=suma+int(i)
		elif tipo=="f":
			suma=suma+float(i)
	return suma


# Para la lista de N enunciados "enunList" de un locutor "sp" del datafram "testResult" devuelve
# dos listas de N elementos con el número de NV  NC (en cada enunciado)

def nasalesEnLocutor(testResult,sp,enunList,minimoVentanasVN,minimoVentanasCN,minimaProbabibNasalidad,espacioEntreNasales):
	numVNEnun=[]
	numCNEnun=[]
	for enun in enunList:
		# print("Enun: ",enun)
		temp=testResult.query('speaker == @sp and utt == @enun')
		listaCNasal=temp['PPNC']
		listaVNasal=temp['PPNV']
		numConsonantes=nasalesEnEnunciado(listaCNasal,minimoVentanasCN,minimaProbabibNasalidad,espacioEntreNasales)
		numVocales=nasalesEnEnunciado(listaVNasal,minimoVentanasVN,minimaProbabibNasalidad,espacioEntreNasales)
		numVNEnun.append(numVocales)
		numCNEnun.append(numConsonantes)

	return numVNEnun, numCNEnun

# Esta función se usa para seleccionar enunciados de las tareas T5 o T6
def	seleccionaEnunciados(enunList,tarea):
	nuevaLista=[]
	for e in enunList:
		if e.startswith(tarea):
			nuevaLista.append(e)
	return nuevaLista

# Calcula el porcentaje de frames nasales (ignora los silencios). 
# La fórmula para saber si un frame es oral o nasal sigue a Mathad et al. (2021)
def mathad(testResult,sp,listaEnun):
	numNasal=0
	numOral=0
	numSil=0

# print("sp: ",sp)
#	print("listaEnun: ",listaEnun)
	registrosEnunHablante=testResult.query('speaker == @sp and utt== @listaEnun')

	for i in range(len(registrosEnunHablante)):	
		frame=registrosEnunHablante.iloc[i]
		nc=frame[1]
		nv=frame[2]
		oc=frame[3]
		ov=frame[4]
		sil=frame[5]
		serie=np.array([nc, nv, oc, ov, sil])
		pCons=math.log(nc/oc)
		pVow=math.log(nv/ov)

		if serie.argmax()==4:
			numSil += numSil
		elif (pCons>1 or pVow>1):
			numNasal = numNasal+1
		else:
			numOral = numOral+1
	if (numNasal + numOral) == 0:
		ratioNasal = 0
	else:
		ratioNasal = int(round(numNasal/(numOral+numNasal),2)*100)
#		ratioNasal = numNasal/(numOral+numNasal)

# 	print("ratioNasal: ",ratioNasal)
	return ratioNasal

def main(rutaInforme, enunList, hacerCorrelacion):

	# 1. Se cargan los resultados de la DNN
	testResult= pd.read_csv(rutaInforme, delimiter=',')

	# Se crea la estructura vacía, solo con columnas de locutor y NivelCapsa
	informe_df=pd.DataFrame(columns=[])

	# Se obtiene la lista de enunciados y locutores
	
	if targetUtt=="ALL":
		enunList=testResult['utt'].unique()
	speakerList=testResult['speaker'].unique()

	informe_df.insert(0,'Locutor',"")

# Se añaden 3 columnas para las medidas de Mathad

	col_Mathad="T5_Mathad"
	informe_df[col_Mathad] = ""
	col_Mathad="T6_Mathad"
	informe_df[col_Mathad] = ""
	col_Mathad="Mathad"
	informe_df[col_Mathad] = ""

	for sp in speakerList:
		
		print("Locutor: ",sp)
	
		enunT5=seleccionaEnunciados(enunList,"T5")
		enunT6=seleccionaEnunciados(enunList,"T6")

		ratioNasalT5=mathad(testResult,sp,enunT5)
		ratioNasalT6=mathad(testResult,sp,enunT6)
		ratioNasalTotal=mathad(testResult,sp,enunList)

		fila=[]
		fila.append(sp)
		fila.append(ratioNasalT5)
		fila.append(ratioNasalT6)
		fila.append(ratioNasalTotal)

		informe_df.loc[len(informe_df)]=fila
	

	# rutaNuevoInformeCSV=rutaInforme[0:-4]+"_TMP.txt"
	# print("Se guarda el informe: ",rutaNuevoInformeCSV)
	# informe_df.to_csv(rutaNuevoInformeCSV, sep ='\t',index=False)
	# exit()

	if hacerCorrelacion==1: # se añaden las correlaciones

		baseDir=os.getcwd()
		rutaNasalanciasPerc=baseDir+"/resultados/resultados_nlace_perceptual_detallado.txt"

		# Lista de campos usados para hacer las correlaciones: "Locutor","NlaceT5","NlaceT6","NlaceTodo","PercHayN","PercSever","PercFormula"

		print("rutaNasalancias: ",rutaNasalanciasPerc)

	# Se lee el dataframe con Nasalancias
		nasalanciasPerc=pd.read_csv(rutaNasalanciasPerc, delimiter='\t')

		# Nombres de las columnas para Nasalancia y ASICA (para hacer la correlación)
		columnasASICA=["T5_Mathad","T6_Mathad","Mathad"]
		columnasPerceptual=["T5Suma2","T6Suma2","TodoSuma2"]
		pearsonR_Percep=["rPearson_Percep"]
		pearsonS_Percep=["sPearson_Percep"]

		for c in range(len(columnasASICA)):

			r, s = pearsonr(informe_df[columnasASICA[c]], nasalanciasPerc[columnasPerceptual[c]])
			pearsonR_Percep.append(r)
			pearsonS_Percep.append(s)

		informe_df.loc[len(informe_df)]=pearsonR_Percep
		informe_df.loc[len(informe_df)]=pearsonS_Percep

# 	# Se guarda el informe de resultados con la clase
	rutaNuevoInformeCSV=rutaInforme[0:-4]+"_ASICA.txt"
	print("Se guarda el informe: ",rutaNuevoInformeCSV)
	informe_df.to_csv(rutaNuevoInformeCSV, sep ='\t',index=False)

	nomLista=rutaInforme.split('/')
	nom=nomLista[-1]

# fin def

if (len(sys.argv)!=4):
	print("Num args.: ",len(sys.argv))
	print("Sintaxis incorrecta. Debe ser: ")
	print("python -m dnn_reportASICA rutaInformeResultados targetUtt correlNasalancia") 
	print("donde")
	print("rutaInformeResultados indica un nombre y ruta del archivo que se analizará")
	print("targetUtt puede ser ASICA, o bien NoAsica. Si es ASICA se analizan los enunciados ASICA")
	print("Ejemplos:")
	print("\t- python -m dnn_reportASICA rutaInformeResultados ASICA 1")
	print("\t- python -m dnn_report rutaInformeResultados NoAsica 0")
	exit();

# print("arg0: ",sys.argv[0])
# print("arg1: ",sys.argv[1])
# print("arg2: ",sys.argv[2])
# print("arg3: ",sys.argv[3])
# print("arg4: ",sys.argv[4])
# exit()

enunciadosASICA=["T0unocinco","T1unodiez","T2pa","T2pi","T2ta","T2ti","T2ka","T2ki","T3f","T3s","T4a","T5moto","T5boca","T5piano","T5pie","T5nino","T5llave","T5luna","T5campana","T5indio","T5dedo","T5gafas","T5silla","T5cuchara","T5sol","T5casa","T5pez","T5jaula","T5zapatos","T6ElBebevaBien","T6UyAhiHayAlgo","T6AlaliLunaLeen","T6AlGato","T6ADavid","T6SusiSaleSola","T6FaliFueFeria","T6ChuchuChelo","T6LosZapatos","T6LaJirafaJesus","T6TomateToda","T6TomateToda","T6PapaPuedePelar","T6QuiqueCoge","T6AMiMamaMe","T6ElneneNoCanta"]
enunciadosASICAOrales=["T2pa","T2pi","T2ta","T2ti","T2ka","T2ki","T4a","T5boca","T5pie","T5llave","T5dedo","T5gafas","T5silla","T5cuchara","T5sol","T5casa","T5pez","T5jaula","T5zapatos","T6UyAhiHayAlgo","T6AlGato","T6ADavid","T6PapaPuedePelar","T6QuiqueCoge"]
enunciadosASICAOrales=["T5boca","T5pie","T5llave","T5dedo","T5gafas","T5silla","T5cuchara","T5sol","T5casa","T5pez","T5jaula","T5zapatos","T6UyAhiHayAlgo","T6AlGato","T6ADavid","T6PapaPuedePelar","T6QuiqueCoge"]


if sys.argv[2]=="ASICA":
	targetUtt=enunciadosASICAOrales
else:
	targetUtt="ALL"

rutaInformeResultados=sys.argv[1]
baseDir=os.getcwd()
rutaAbsInforme=baseDir+rutaInformeResultados

if os.path.exists(rutaAbsInforme):
	print("Se va a analizar el informe: ",rutaAbsInforme)
else:
	print("El informe de resultados indicado no existe: ",rutaAbsInforme)
	exit()

hacerCorrelacion=int(sys.argv[3])

main(rutaAbsInforme,targetUtt,hacerCorrelacion)

