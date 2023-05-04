# Este script lee crea una base de datos a partir de 
# de pares de archivos de "carpetaAudios" (por defecto: 'audiosTest') de archivos
# y textGrid con un tier. Los audios de Testeo son los que se hicieron con la app ASICA
# por tanto deben tener 45 enunciados anotados.

# 
#  Se crea una BD .csv donde cada solo se extraen las ventanas etiquetadas en el TextGrid. 
#   De cada intervalo se extraen ventanas de 25 ms (con un paso de 10ms) 
# Cada fila de la BD final tendrá: 
# ID del locutor
# ID del enunciado
# Coef MFCC, Delta y DeltaDelta de una ventana
# 
# La base de datos resultante se guardará en la carpeta y con el nombre 
# espedificado en:
#
#	carpetaBaseDatos = 'resultadosBaseDatos';
#	nombreBaseDatos = 'BD_testeoASICA.csv';
# 
# Nota: Si la BD existe, no la crea, sino que añade los datos nuevos 
# (basándose en el nombre del locutor)
#
#
# Cada par de archivos es un WAV y un TextGrid con el mismo nombre. 
# Cada archivo WAV lo divide en los intervalos indicados en el tier 2. Y de cada
# intervalo toma ventanas de 25ms (con 10 ms de salto) de las que calcula 
# los coeficientes MFCC, Delta y DeltaDelta. Además se calcula la clase asociada a la
# cada intervalo (y por tanto a las ventanas de esse intervalo). Sobre esto ver la función:  
# def subclas(phone, nasalance, rms):
# 
# IMPORTANTE: 
# La clasificación un intervalo como SL (silencio), se hace en función del valor RMS 
# calculado en el script de etiquetado automático (Nasalance_per_tier_v7.script) 
# En ese script se normaliza la intensidad a 70 dB. Sin embargo el audio usado
# en ese script (el del Nasalance) no es el mismo que el usado aquí (aquí se usa el 
# del iPhone). Por tanto, hay que normalizar el audio del iPhone antes de pasar este
# script

import textgrid
from pathlib import Path
import pandas as pd 
import numpy as np
import os

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


def defineColumnaRow(feat, enun, archivoAudio):

	contador=1
	nuevaFila=[]
	for c in feat['Row']:
		cadena="Test_"+archivoAudio+"_a_Nas_"+enun+"_"+str(contador)
		nuevaFila+=[cadena]
		contador=contador+1

	featNuevo=feat.copy()
	featNuevo['Row'] = nuevaFila

	return featNuevo

# Dada una cadena cadena="revisados_"+nomArchivo[0:-4]+"_w"+ventana+"_"+sc
# recupera el id del locutor
def extraeLocutor(c):
	return c.split('_')[0]

def extraeNomIntervalo(c):
	return c.split('_')[1]

# Dada una anotación se indica un formato normalizado
def recodificaEnunciado(c):
	c=limpia(c)
	if c.endswith("cinco"):
		enun="T0unocinco"		
	elif c.endswith("diez"):
		enun="T1unodiez"
	elif c.startswith("pa") and c.endswith("pa") and len(c)>2 :
		enun="T2pa"
	elif c.startswith("pi") and c.endswith("pi") and len(c)>2 :
		enun="T2pi"
	elif c.startswith("ta") and c.endswith("ta") and len(c)>2 :
		enun="T2ta"
	elif c.startswith("ti") and c.endswith("ti") and len(c)>2 :
		enun="T2ti"
	elif c.startswith("ka") and c.endswith("ka") and len(c)>2 :
		enun="T2ka"
	elif c.startswith("ki") and c.endswith("ki") and len(c)>2 :
		enun="T2ki"
	elif c.startswith("f") and len(c)==1:
		enun="T3f"
	elif c.startswith("s") and len(c)==1:
		enun="T3s"
	elif c.startswith("a") and len(c)==1:
		enun="T4a"
	elif c=="moto":
		enun="T5moto"
	elif c=="boca":
		enun="T5boca"
	elif c=="piano":
		enun="T5piano"
	elif c=="pie":
		enun="T5pie"
	elif c=="niño":
		enun="T5nino"
	elif c=="llave":
		enun="T5llave"
	elif c=="luna":
		enun="T5luna"
	elif c=="campana":
		enun="T5campana"
	elif c=="indio":
		enun="T5indio"
	elif c=="dedo":
		enun="T5dedo"
	elif c=="gafas":
		enun="T5gafas"
	elif c=="silla":
		enun="T5silla"
	elif c=="cuchara":
		enun="T5cuchara"
	elif c=="sol":
		enun="T5sol"
	elif c=="casa":
		enun="T5casa"
	elif c=="pez":
		enun="T5pez"
	elif c=="jaula":
		enun="T5jaula"
	elif c.startswith("zapato"):
		enun="T5zapatos"
	elif c.startswith("el beb"):
		enun="T6ElBebevaBien"
	elif c.startswith("uy, ahí"):
		enun="T6UyAhiHayAlgo"
	elif c.startswith("Lali y Luna"):
		enun="T6AlaliLunaLeen"
	elif c.startswith("al gato"):
		enun="T6AlGato"
	elif c.startswith("a David"):
		enun="T6ADavid"
	elif c.startswith("si llueve"):
		enun="T6SiLlueveLeLlevo"
	elif c.startswith("Susi sale") or c.startswith("susi sale"):
		enun="T6SusiSaleSola"
	elif c.startswith("Fali fue"):
		enun="T6FaliFueFeria"
	elif c.startswith("Chuchu"):
		enun="T6ChuchuChelo"
	elif c.startswith("los zapatos"):
		enun="T6LosZapatos"
	elif c.startswith("la jirafa"):
		enun="T6LaJirafaJesus"
	elif c.startswith("tómate") or c.startswith("Tómate"):
		enun="T6TomateToda"
	elif c.startswith("tómate toda"):
		enun="T6TomateToda"
	elif c.startswith("papá puede"):
		enun="T6PapaPuedePelar"
	elif c.startswith("Quique"):
		enun="T6QuiqueCoge"
	elif c.startswith("mi mamá"):
		enun="T6AMiMamaMe"
	elif c.startswith("el nene"):
		enun="T6ElneneNoCanta"
	elif len(c)==0:
		enun=""
	else:
		enun=c

	return enun

# Fin def

def limpia(enun):
	while enun.startswith(" ") or enun.startswith("	"):
		enun=enun[1:]
	while enun.endswith(" ") or enun.endswith("	"):
		enun=enun[0:-1]
	return enun


# Variables globales
##Carpeta que contiene los audios
carpetaAudios = 'audiosTest/Experim2/Nas2';
carpetaAudios = 'audiosTest/Experim2/Nas1';
carpetaAudios = 'audiosTest/Experim2/NasSuma';

# Base de datos 
carpetaBaseDatos = 'BDsTesteo';
nombreBaseDatos = 'BDTesteo_Exp2_Nas2.csv';
nombreBaseDatos = 'BDTesteo_Exp2_Nas1.csv';
nombreBaseDatos = 'BDTesteo_Exp2_NasSuma.csv';

##Definición de descriptores
#featuresToCalculate = ["mfcc","delta","deltaDelta"];

# tiers del textGrid
tierEnunciados=0

contador=1 # Sirve para contar ventanas de cada archivo
totalVentanas=0 

# Archivos WAV (se supone que cada WAV tendrá su textGrid correspondiente)
os.chdir(carpetaAudios)
fileDir = Path.cwd()
audiosAsica = fileDir.glob('*.wav')
audiosAsica=sorted(audiosAsica)
if audiosAsica==[]:
	print("No hay archivos .wav en la carpeta: ",carpetaAudios)
	print("Mete ahí los archivos .wav y sus correspondientes .textGrid y vuelve a ejecutar el scipt")
	exit()
os.chdir("..")
os.chdir("..")
os.chdir("..")

# Base de datos
os.chdir(carpetaBaseDatos)
print("Base de datos: ",carpetaBaseDatos+"/"+nombreBaseDatos)
if os.path.exists(nombreBaseDatos):
	nomBDstr=str(nombreBaseDatos)
	print("La BD existe")
	featuresTodosIntervalos=pd.read_csv(nombreBaseDatos)
#	listaLocutores=calculaLocutores(featuresTodosIntervalos)
#	print("Lista de locutores analizados anteriormente:")
#	for l in listaLocutores:
#		print(l)
	totalVentanas=len(featuresTodosIntervalos['Row'])
	print("Número de ventanas en la BD original: ",totalVentanas)
# Está pendiente hacer un control para cargar nuevas ventanas (y que no se reptitan)
# De haberlos, esta lista tendría que tener la lista de locutres ya incluidos

else:
	print("La BD no existe. Se crea una nueva")
#	listaLocutores=[]
	featuresTodosIntervalos=[]
	featuresTodosIntervalos = pd.DataFrame()
	totalVentanas=0
os.chdir("..")

audioProcdesados=0
totalAudios=len(audiosAsica)

ventanasNuevas=0

#  Empiezan los cálculos para cada archivo
for file in audiosAsica:
	fileStr=str(file)
	contador=1

	ruta, archivoAudio = os.path.split(file)
	textGridStr=fileStr.replace(".wav",".TextGrid")
	rate,sig = wavfile.read(file)

	if archivoAudio.endswith(".wav"):
		archivoAudio=archivoAudio[:-4]
	print("Nuevo locutor: ", archivoAudio)
	locutor=archivoAudio.split('_')[0] + archivoAudio.split('_')[0]

#	if (locutor in listaLocutores):
#		print("-- Locutor repetido. Pasamos al siguiente")
#		continue
	if (os.path.isfile(textGridStr)):
		grid=textgrid.TextGrid.fromFile(textGridStr)
		rutaT, archivoT = os.path.split(textGridStr)
		print("-- Leído el textGrid: ",archivoT)
	else:
  		rutaT, archivoT = os.path.split(textGridStr)
  		print("No se ha podido abrir el textGrid: ",textGridStr)
  		print("-- Se ignora el wav. Pasamos al siguiente")
  		continue

	# grid=textgrid.TextGrid.fromFile(textGridStr)
	
	nIntervalos=len(grid[tierEnunciados].intervals)-1
	nEnunProcesados=0
	for intervalo in (range(nIntervalos)):
		enun=recodificaEnunciado(grid[tierEnunciados][intervalo].mark)
		enun=limpia(enun)
		if len(enun)==0:
			continue
		else:
			min=grid[tierEnunciados][intervalo].minTime
			max=grid[tierEnunciados][intervalo].maxTime
			beginFragment=int(round(min*rate))
			endFragment=int(round(max*rate))
			fragmento=sig[beginFragment:endFragment]
			features=allFeatures(rate,fragmento) # Archivo de Audio
			ventanasNuevas=+ventanasNuevas+len(features)
			features=defineColumnaRow(features, enun, archivoAudio)
			featuresTodosIntervalos = pd.concat([featuresTodosIntervalos, features])
			nEnunProcesados=nEnunProcesados+1
	print("Se han leído: ", nEnunProcesados," enunciados")


bdSalida=carpetaBaseDatos+"/"+nombreBaseDatos
if ventanasNuevas > 0:
	featuresTodosIntervalos.to_csv(bdSalida, index=False)
	print("Se han creado ", ventanasNuevas, " ventanas nuevas.")
	print("Se han guardado los resultados en: ", bdSalida)
else:
	print("No hay nuevos resultados. La base de datos, ", bdSalida, "no ha cambiado")
# print(featuresTodosIntervalos)