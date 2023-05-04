# Este script  crea una base de datos a partir de 
# de pares de archivos ubicados en la "carpetaAudios": archivos .wav 
# y sus correspondientes textGrid. Los TextGrid tienen este formato:

#  Primer tier: Palabras  (palabra. Este tier no sería necesario)
#  Segundo tier: fonemas 
#  Tercer tier: Nasalancia del fonema
#  Cuarto tier: RMS del intervalo del fonema 
#  Nota: Para calcular la RMS, el script de Praat normaliza el audio con este comando 
# Scale intensity: 65

#  Se crea una BD .csv donde cada fila tiene los coef MFCC, Delta y DeltaDelta de una 
#  ventana de 25ms de uno de los audios y su clase (NV, OV, NC, OC, SL). La base de
# datos resultante se guardará en la carpeta y con el nombre espedificado en:
#
#	carpetaBaseDatos = 
#	nombreBaseDatos = 
# 
# Nota: Si la BD existe, no la crea, sino que añade los datos nuevos (basándose en el nombre del locutor)
#
# Cada archivo WAV lo divide en los intervalos indicados en el tier PHONES. Y de cada
# intervalo toma ventanas de 25ms (con 10 ms de salto) de las que calcula 
# los coeficientes MFCC, Delta y DeltaDelta. Además se calcula la clase asociada a
# cada intervalo (y por tanto a las ventanas de esse intervalo). Sobre esto ver la función:  
# def subclas(phone, nasalance, rms):
# 
# IMPORTANTE: 
# La clasificación un intervalo como SL (silencio), se hace en función del valor RMS 
# calculado en el script de etiquetado automático (Nasalance_per_tier_v7.script) 

import textgrid
from pathlib import Path
import pandas as pd 
import numpy as np
import os
import sys

import csv

from scipy.io.wavfile import write
from scipy.io import wavfile 

from python_speech_features import mfcc
from python_speech_features import delta

import scipy.io.wavfile as wav


# Función que toma una señal y su frecuencia devuelve un df con sus 39 features mfcc

def allFeatures(rate,sig):
    
#    print(f'Duracion = {sig.shape[0]/rate} , Frecuencia de Muestreo = {rate} [=] Muestras/Seg' \
#     f', Wav format = {sig.dtype}')
    mfcc_feat = mfcc(sig, samplerate=rate, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=1024, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=False, winfunc=np.hamming)
    delta_feat = delta(mfcc_feat, 2)
    deltaDelta_feat = delta(delta_feat, 2)

    df_mfcc = pd.DataFrame(mfcc_feat)  #converting it to a pandas dataframe
    df_delta = pd.DataFrame(delta_feat)
    df_deltaDelta = pd.DataFrame(deltaDelta_feat)

    numVentanas = len(df_mfcc[0]) 
    indice=range(numVentanas)
    
    df_mfcc.insert(0,'Row', indice)
    df_delta.insert(0,'Row', indice)
    df_deltaDelta.insert(0,'Row', indice)

    df_mfcc.rename(columns={0: 'mfcc0', 1: 'mfcc1',2: 'mfcc2',3: 'mfcc3',4: 'mfcc4',5: 'mfcc5',6: 'mfcc6',7: 'mfcc7',8: 'mfcc8',9: 'mfcc9',10: 'mfcc10',11: 'mfcc11',12: 'mfcc12'}, inplace=True)
    df_delta.rename(columns={0: 'delta0', 1: 'delta1',2: 'delta2',3: 'delta3',4: 'delta4',5: 'delta5',6: 'delta6',7: 'delta7',8: 'delta8',9: 'delta9',10: 'delta10',11: 'delta11',12: 'delta12'}, inplace=True)
    df_deltaDelta.rename(columns={0: 'deltaDelta0', 1: 'deltaDelta1',2: 'deltaDelta2',3: 'deltaDelta3',4: 'deltaDelta4',5: 'deltaDelta5',6: 'deltaDelta6',7: 'deltaDelta7',8: 'deltaDelta8',9: 'deltaDelta9',10: 'deltaDelta10',11: 'deltaDelta11',12: 'deltaDelta12'}, inplace=True)

    df_mfcc.merge(df_delta, on='Row')
    df_mfcc.merge(df_deltaDelta, on='Row')

    merged1 = pd.merge(left=df_mfcc,right=df_delta)
    merged2 = pd.merge(left=merged1,right=df_deltaDelta)

    return merged2



# Función que define los valores de la primera columna 
# De momento uso el formato que uso Andrés
# revisado_nombre_Texto_ventana_Clase
# revisados_AguilarJimenez_Vanero_w000001_SL

def defineColumnaRow(feat, sc, nomArchivo, contador):

	nuevaFila=[]
	for c in feat['Row']:

		cadena=""
		ventana=str(contador+c)
		if len(ventana) == 1:
			ventana = "00000"+ventana
		elif len(ventana) == 2:
			ventana = "0000"+ventana
		elif len(ventana) == 3:
			ventana = "000"+ventana
		elif len(ventana) == 4:
			ventana = "00"+ventana
		elif len(ventana) == 5:
			ventana = "0"+ventana
		elif len(ventana) == 6:
			ventana = ventana
		else:
			Print("Error: hay más de 999999 ventanas. Corrige el contador de ventanas en la función defineColumnaRow")
			exit()
		if nomArchivo.endswith(".wav"):
			nomArchivo=nomArchivo[:-4]
		if len(nomArchivo.split('_'))==1: 
			nomArchivo=nomArchivo+"_textos"		
		cadena=nomArchivo+"_w"+ventana+"_"+sc
		nuevaFila+=[cadena]

	featNuevo=feat.copy()
	featNuevo['Row'] = nuevaFila

	# print("sc: ",sc)
	# print(featNuevo['Row'])

	return featNuevo

# Dada una cadena cadena="revisados_"+nomArchivo[0:-4]+"_w"+ventana+"_"+sc
# recupera el id del locutor
def extraeLocutor(c):
	limite=c.find('_')
	locutor=c[:limite]
	return locutor

# Obtiene una lista de locutores (con el fin de no dupicar datos)

def calculaLocutores(featuresTodosIntervalos):
	listaLocutores=[]
	columnaLocutores=featuresTodosIntervalos['Row']
	r=0
	for c in columnaLocutores:
		# locutor=columnaLocutores[11:-11]
		# print("Locutor de la columna Row: ", c)

		locutor=extraeLocutor(c)
		# print("Locutor extraido: ", locutor)
		if listaLocutores==[]:
			listaLocutores.append(locutor)
#			r=r+1
		elif locutor in listaLocutores:
			r=r+1
		else: 
			listaLocutores.append(locutor)
#			r=r+1
		listaLocutores.sort()	
	return listaLocutores


# Calcula la clase de sonido: SL, OC, OV, NC, NV (y las variantes NVp y NVn)
def subclas30(phone, prevPhone,nextPhone,rms,minRMS):

    vocales = ['a','e','i','o','u','j','w','a+','e+','i+','o+','u+','aI','eI','oI','uI','aU','eU','iU','oU']
    consOral = ['b', 'c', 'd̪','d', 'D', 'f', 'k', 'l', 'p', 'r', 'rf', 's', 'tʃ','tS', 't̪','t', 'w', 'x', 'ç', 'ð', 'ɟ', 'ɟʝ', 'ɡ','g','G', 'ɣ', 'ɾ', 'ʃ', 'ʎ', 'ʝ','L', 'β', 'θ','T']
    consNasal = ['ɲ', 'm', 'n', 'ŋ','n~'] 

    subclase30=""

# Para identificar la clase NV se tiene en cuenta si prevPhone o nextPhone son nasales.
# Las etiquetas NVn t NVp se usaran para calcular el fragmeno exacto que se usa como nasal
# pero la etiqueta guardada en la BD será NV
    if  phone in vocales:
    	if prevPhone in consNasal and nextPhone in consNasal:
    		subclase30="NV"
    	elif prevPhone in consNasal:
    		subclase30="NVp"
    	elif nextPhone in consNasal:
    		subclase30="NVn"
    	else:
    		subclase30="OV"
    elif phone in consOral:
        subclase30="OC"
    elif phone in consNasal:
        subclase30="NC"
    elif phone=="spn":
        print("Aviso: Palabra no etiquetada (spn)")
        subclase30=""        
    elif (phone=="" and rms < minRMS):
        subclase30="SL"
    elif (phone=="" and rms > minRMS):
        subclase30=""
    else:
    	print("Aviso: el símbolo /",phone,"/anotado en el textGrid no está en el diccionario.")
    	print("y se ignora. Símbolos admitidos: ")
    	print("    - Vocales: ", vocales)
    	print("    - Consonantes orales: ",  consOral)
    	print("    - Consonantes nasales: ", consNasal)
    	subclase30=""
    
    return subclase30

# Añado esta función para poder informar del número de ventanas del la BD
def data_preprocessing(df):
    
    # hay diferentes tipos de clases que podemos clasificar OV, OC, SL, NC, NV 
    df['class'] = df['Row'].apply(lambda x: x.split('_')[-1])
    
    df['class_four'] = df['class'].map({'OV': 'OV', 'OC': 'OC',
                                        'NC': 'NC', 'NV': 'NV'})    
    
    df['silence0_sound1'] = df['class'].map({'SL': '0', 'OV': '1', 'OC': '1', 'NC': '1', 'NV': '1'})
    df['silence_nasal_oral'] = df['class'].map({'SL': 'silence', 'OV': 'oral', 'OC': 'oral', 'NC': 'nasal', 'NV': 'nasal'})
    
    df['vowel0_consonant1'] = df['class'].map({'OV': '0', 'OC': '1', 'NC': '1', 'NV': '0'})
    df['ov0_oc1'] = df['class'].map({'OV': '0', 'OC': '1'})
    df['nv0_nc1'] = df['class'].map({'NV': '0', 'NC': '1'})

    df['oral0_nasal1'] = df['class'].map({'OV': '0', 'OC': '0', 'NC': '1', 'NV': '1'})
    df['ov0_nv1'] = df['class'].map({'OV': '0', 'NV': '1'})
    df['oc0_nc1'] = df['class'].map({'OC': '0', 'NC': '1'})

    
    df['window'] = df['Row'].apply(lambda x: x.split('_')[-2])
    
    # identificador del audio que dicen
    df['text'] = df['Row'].apply(lambda x: x.split('_')[-3])
    df['text'] = df['text'].replace({'Vanero':'Joyero', 'HOM':'Joyero', 'joyero':'Joyero',
                                 'VientoN':'Viento', 'VientoV':'Viento', 'ElViento':'Viento',
                                 'HayAlgo':'HayAlgoAhi', 'hayAlgo':'HayAlgoAhi','iphone':'CuatroTextos'
                                })
    
    df['speaker'] = df['Row'].apply(lambda x: ''.join(x.split('_')[1:-3])) # evita errores con nombres duplicados
    
    return df


# Fin def 

# Si balanceado == True .Se selecciona mismo núm de ventanas de cada clase
balanceado=True
# Si numClases es 3: Oral/Nasal/SL. SI es 5: NC, NV, OC, OV, SL 
numClases=5

def main(rutaAudios, rutaBaseDatos, baseDatos):

	baseDir=Path.cwd()

	# Valor mínimo de RMS para ser clasificado como silencio
	minRMS=30

	# tiers del textGrid
	tierWords=0
	tierPhones=1
	tierNlce=2
	tierRMS=3

	totalVentanas=0 

	# Archivos WAV (se supone que cada WAV tendrá su textGrid correspondiente)
	os.chdir(rutaAudios)
	fileDir = Path.cwd()
	audiosAsica = fileDir.glob('*.wav')
	audiosAsica=sorted(audiosAsica)
	if audiosAsica==[]:
		print("No hay archivos .wav en la carpeta: ",rutaAudios)
		print("Mete ahí los archivos .wav y sus correspondientes textGrid y vuelve a ejecutar el scipt")
		exit()
	os.chdir(baseDir)

	# Base de datos
	os.chdir(rutaBaseDatos)
	print("Base de datos: ",baseDatos)
	if os.path.exists(baseDatos):
		nomBDstr=str(baseDatos)
		print("La BD existe")
		featuresTodosIntervalos=pd.read_csv(baseDatos)
		listaLocutores=calculaLocutores(featuresTodosIntervalos)
		print("Lista de locutores analizados anteriormente:")
		for l in listaLocutores:
			print(l)
		totalVentanas=len(featuresTodosIntervalos['Row'])
		print("Número de ventanas en la BD original: ",totalVentanas)

	else:
		print("La BD no existe. Se crea una nueva")
		listaLocutores=[]
		featuresTodosIntervalos=[]
		totalVentanas=0
	os.chdir("..")
	os.chdir("..")

	audioProcdesados=0
	totalAudios=len(audiosAsica)

	ventanasNuevas=0

	#  Empiezan los cálculos para cada archivo
	for file in audiosAsica:
		fileStr=str(file)
		contador=1 # Sirve para contar ventanas de cada archivo

		ruta, archivoAudio = os.path.split(file)
		textGridStr=fileStr.replace(".wav",".TextGrid")
		rate,sig = wavfile.read(file)

		if archivoAudio.endswith(".wav"):
			nomLocutor=extraeLocutor(archivoAudio[:-3])

		print("Nuevo locutor: ", nomLocutor)
		if (nomLocutor in listaLocutores):
			print("-- Locutor repetido. Pasamos al siguiente")
			continue
		if (os.path.isfile(textGridStr)):
			grid=textgrid.TextGrid.fromFile(textGridStr)
			rutaT, archivoT = os.path.split(textGridStr)
			print("-- Leído el textGrid: ",archivoT)
		else:
	  		rutaT, archivoT = os.path.split(textGridStr)
	  		print("No se ha podido abrir el textGrid: ",textGridStr)
	  		print("-- Se ignora el wav. Pasamos al siguiente")
	  		continue

		numV_OC=0
		numV_OV=0
		numV_NC=0
		numV_NV=0
		numV_SL=0

		# Bucle para calcular ventanas de cada clase
		
		nIntervalos=len(grid[tierRMS].intervals)-1

		for intervalo in (range(nIntervalos)):
			phone=grid[tierPhones][intervalo].mark
			if intervalo>0:
				prevPhone=grid[tierPhones][intervalo-1].mark
			else:
				prevPhone=""
			nextPhone=grid[tierPhones][intervalo+1].mark
			rms=int(grid[tierRMS][intervalo].mark)
			if len(grid[tierNlce][intervalo].mark)>0:
				nlceText=grid[tierNlce][intervalo].mark
				nlceText=nlceText.split('.')[0]
				nlce=int(nlceText)
			else:
				nlce=0
			sc=subclas30(phone, prevPhone,nextPhone,rms,minRMS)
			min=grid[tierRMS][intervalo].minTime
			max=grid[tierRMS][intervalo].maxTime
			duracion=max-min		
			if duracion<0.026:
				continue
			porcionNasalizada=0
			if sc=="NV":
				porcionNasalizada=1
			elif (sc=="NVp" or sc=="NVn") and duracion>0.09: # Si el sonido es de más de 90ms se extraerá el 30%
				porcionNasalizada=0.3
			elif (sc=="NVp" or sc=="NVn") and duracion>0.06: # Si el sonido es de 60-90ms se extraerá el 50%
				porcionNasalizada=0.5
			elif (sc=="NVp" or sc=="NVn"): # Si es más breve se extraerá el 100%
				porcionNasalizada=1
			
			if sc=="SL":
				nv=int(round((duracion-0.015)/0.01))
				numV_SL=numV_SL+nv
			elif sc=="OV":
				nv=int(round((duracion-0.015)/0.01))
				numV_OV=numV_OV+nv
			elif sc=="OC":
				nv=int(round((duracion-0.015)/0.01))
				numV_OC=numV_OC+nv
			elif sc=="NC":			
				nv=int(round((duracion-0.015)/0.01))
				numV_NC=numV_NC+nv
			elif sc=="NVp" or sc=="NVn" or sc=="NV": 
				duracionFragmentoNasal=porcionNasalizada*duracion
				nv=int(round((duracionFragmentoNasal-0.015)/0.01))
				numV_NV=numV_NV+nv
	

		maximos=[numV_NC, numV_OC, numV_NV, numV_OV,numV_SL]
		# print(maximos)
		elementoMenor=np.argmin(maximos)
		elementoMayor=np.argmax(maximos)

		maxVentanas=maximos[elementoMenor]
		if numClases==3:
			maxVentanasSil=maxVentanas*2
		elif numClases==5:
			maxVentanasSil=maxVentanas
		else:
			print("Error: solo puede haber tres o cinco clases")
			exit()

		print("Max. NC: ", numV_NC)
		print("Max. OC: ", numV_OC)
		print("Max. NV: ", numV_NV)
		print("Max. OV: ", numV_OV)
		print("Max. SL: ", numV_SL)
		print("Max ventanas OV, NV, NC y NV: ",maxVentanas)
		print("Max ventanas SL: ",maxVentanasSil)

		numV_NC=0
		numV_OC=0
		numV_NV=0
		numV_OV=0
		numV_SL=0

		nIntervalos=len(grid[tierRMS].intervals)-1
		for intervalo in (range(nIntervalos)):
			nextPhone=grid[tierPhones][intervalo+1].mark
			prevPhone=grid[tierPhones][intervalo+1].mark
			min=grid[tierRMS][intervalo].minTime
			max=grid[tierRMS][intervalo].maxTime
			duracion=max-min
			if duracion<0.026:
				continue
			nv=int(round((duracion-0.015)/0.01))
			beginFragment=int(round(min*rate))
			endFragment=int(round(max*rate))
			fragmento=sig[beginFragment:endFragment]
			phone=grid[tierPhones][intervalo].mark
			rms=int(grid[tierRMS][intervalo].mark)
			
			sc=subclas30(phone, prevPhone,nextPhone,rms,minRMS)
			
			if sc=="SL" and numV_SL > maxVentanasSil:
				continue
			elif sc=="OC" and numV_OC > maxVentanas:
				continue	
			elif sc=="OV" and numV_OV > maxVentanas:
				continue
			elif sc=="NC" and numV_NC > maxVentanas:
				continue
			elif (sc=="NVp" or sc=="NVn" or sc=="NV") and numV_NV > maxVentanas:
				continue


	# Empieza propiamente el análisis 

			if sc=="NVn" or sc=="NVp" or sc=="NV":
				beginFragment=int(round(min*rate))
				endFragment=int(round(max*rate))
				duracion=max-min
				if sc=="NV":
					procionNasalizada=1
				elif duracion>0.09: # Si el sonido es de más de 90ms se extraerá el 30%
					porcionNasalizada=0.3
				elif duracion>0.06: # Si el sonido es de 60-90ms se extraerá el 50%
					porcionNasalizada=0.5
				else: # Si es más breve se extraerá el 100%
					porcionNasalizada=1
				duracionFragmentoNasal=duracion*porcionNasalizada		
				if sc=="NVn":
					corte=int(round((max-duracionFragmentoNasal)*rate))
					sigNasal=sig[corte:endFragment]
				elif sc=="NVp":
					corte=int(round((min+duracionFragmentoNasal)*rate))
					sigNasal=sig[beginFragment:corte]
				else:
					sigNasal=sig[beginFragment:endFragment]

				if len(sigNasal)>0:
					featuresNasal=allFeatures(rate,sigNasal) # Archivo de Audio
					featuresNasal=defineColumnaRow(featuresNasal, "NV", archivoAudio, contador)
					numV_NV=numV_NV+len(featuresNasal)
					contador=contador+len(featuresNasal)

					if len(featuresTodosIntervalos)==0:
						featuresTodosIntervalos=featuresNasal
					else:	
						featuresTodosIntervalos = pd.concat([featuresTodosIntervalos, featuresNasal])


			elif duracion>0 and len(featuresTodosIntervalos)==0 and sc!="":
				features=allFeatures(rate,fragmento) # Archivo de Audio
				features=defineColumnaRow(features, sc, archivoAudio, contador)
				contador=contador+len(features)
				featuresTodosIntervalos=features
				if sc=="OV":
					numV_OV=numV_OV+nv
				elif sc=="OC":
					numV_OC=numV_OC+nv
				elif sc=="SL":
					numV_SL=numV_SL+nv
				elif sc=="NC":
					numV_NC=numV_NC+nv

			elif duracion>0 and len(featuresTodosIntervalos)>0 and sc!="" and len(fragmento)>0:	
				features=allFeatures(rate,fragmento) # Archivo de Audio
				features=defineColumnaRow(features, sc, archivoAudio, contador)
				contador=contador+len(features)
				featuresTodosIntervalos = pd.concat([featuresTodosIntervalos, features])
				if sc=="OV":
					numV_OV=numV_OV+nv
				elif sc=="OC":
					numV_OC=numV_OC+nv
				elif sc=="SL":
					numV_SL=numV_SL+nv
				elif sc=="NC":
					numV_NC=numV_NC+nv


		totalVentanas=totalVentanas+contador
		ventanasNuevas=ventanasNuevas+contador
		print("-- Ventanas procesadas en el último audio: ",contador)
		print("-- Ventanas procesadas en total: ", totalVentanas)
			


	print("BDSalida: ",baseDatos)
	if ventanasNuevas > 0:
		featuresTodosIntervalos.to_csv(baseDatos, index=False)
		print("Se han guardado los resultados en: ", baseDatos)
	else:
		print("No hay nuevos resultados. La base de datos, ", baseDatos, "no ha cambiado")

	df = data_preprocessing(featuresTodosIntervalos)
	print("Ventanas por clase")
	print(df['class'].value_counts().sort_index(ascending=False))


# end def

print("NumArgs: ",len(sys.argv))

if (len(sys.argv)!=4):
    print("Sintaxis incorrecta. Debe ser: ")
    print("python -m dnn0a_CalcDesc_Train rutaAudios rutaBaseDatos baseDeDatos")
    exit();

rutaAudios=sys.argv[1]

rutaabsocultaAudios=os.chdir(rutaAudios)

rutaBaseDatos=sys.argv[2]

baseDatos=sys.argv[2]+sys.argv[3]


print("rutaAudios: ",rutaAudios)
print("rutaBaseDatos: ",rutaBaseDatos)
print("baseDatos: ",baseDatos)

main(rutaAudios, rutaBaseDatos, baseDatos)



