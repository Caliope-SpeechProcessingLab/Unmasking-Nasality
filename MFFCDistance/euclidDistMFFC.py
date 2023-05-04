# This script will compute the MFCC Euclidean distance for audio files
# stored in the folder Audios/. The results are saved to a file named MFFC_distance.txt
# 

from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import pandas as pd
from scipy.io import wavfile
from python_speech_features import mfcc
import math
import seaborn as sns
import matplotlib.pyplot as plt
  


# Calcula la distancia euclídea entre dos listas
def eucDist(p1,p2):
	sum = 0
	for i in range(len(p1)):
		elem = (p1[i]-p2[i])**2
		sum += elem
	return (sum) ** (1/2)

# Dada una frecuencia y una señal, devuelve los valores medios de sus MFCC
def mfccMean(rate,sig):
    
	mfcc_feat = mfcc(sig, samplerate=rate, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=1024, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=False, winfunc=np.hamming)
	df_mfcc = pd.DataFrame(mfcc_feat)  #converting it to a pandas dataframe
 
	
	df_mfcc.rename(columns={0: 'mfcc0', 1: 'mfcc1',2: 'mfcc2',3: 'mfcc3',4: 'mfcc4',5: 'mfcc5',6: 'mfcc6',7: 'mfcc7',8: 'mfcc8',9: 'mfcc9',10: 'mfcc10',11: 'mfcc11',12: 'mfcc12'}, inplace=True)
 
	nMFCCs=range(12) 

	medias=[]


	listaCol=['mfcc0', 'mfcc1','mfcc2','mfcc3','mfcc4','mfcc5','mfcc6','mfcc7','mfcc8','mfcc9','mfcc10','mfcc11','mfcc12']

	for m in listaCol:
		media=df_mfcc[m].mean()
		medias.append(media)
 
	return medias

ppio="Audios/stg20007"
n="n"
o="o"
finales=["_1.wav","_2.wav","_3.wav","_4.wav","_5.wav","_6.wav"]
# Vocal A nasales

vocales=["a","e","i","o","u"]


distrEucMFCC = pd.DataFrame(columns=['Vocal','Audio',"EucDistMFCC"])

todasDistEucCh0=[]
todasDistEucCh1=[]
todasDistEucCh2=[]


for vocal in vocales:
	canal="_ch0_"
	serieOralesCh0=[]
	serieNasalesCh0=[]

	numOrales=[]
	numNasales=[]

	for f in finales:
		fnameN=ppio+canal+vocal+n+f
		fnameO=ppio+canal+vocal+o+f
		fqN, sN = wavfile.read(fnameN)
		fqO, sO = wavfile.read(fnameO)
		mfccsN=mfccMean(fqN,sN)
		mfccsO=mfccMean(fqO,sO)
		serieNasalesCh0.append(mfccsN)
		serieOralesCh0.append(mfccsO)

	distEucCh0=[]
	for nas in serieNasalesCh0:
		for oral in serieOralesCh0:
			distEuc=eucDist(nas,oral)
			distEucCh0.append(distEuc)
			todasDistEucCh0.append(distEuc)
			datoNuevo=[vocal,"Monophonic",distEuc]
			distrEucMFCC.loc[len(distrEucMFCC)]=datoNuevo


	canal="_ch1_"
	serieOralesCh1=[]
	serieNasalesCh1=[]

	for f in finales:
		fnameN=ppio+canal+vocal+n+f
		fnameO=ppio+canal+vocal+o+f
		fqN, sN = wavfile.read(fnameN)
		fqO, sO = wavfile.read(fnameO)
		mfccsN=mfccMean(fqN,sN)
		mfccsO=mfccMean(fqO,sO)
		serieNasalesCh1.append(mfccsN)
		serieOralesCh1.append(mfccsO)


	distEucCh1=[]
	for nas in serieNasalesCh1:
		for oral in serieOralesCh1:
			distEuc=eucDist(nas,oral)
			distEucCh1.append(distEuc)
			todasDistEucCh1.append(distEuc)
			datoNuevo=[vocal,"Nose",distEuc]
			distrEucMFCC.loc[len(distrEucMFCC)]=datoNuevo


	canal="_ch2_"
	serieOralesCh2=[]
	serieNasalesCh2=[]

	for f in finales:
		fnameN=ppio+canal+vocal+n+f
		fnameO=ppio+canal+vocal+o+f
		fqN, sN = wavfile.read(fnameN)
		fqO, sO = wavfile.read(fnameO)
		mfccsN=mfccMean(fqN,sN)
		mfccsO=mfccMean(fqO,sO)
		serieNasalesCh2.append(mfccsN)
		serieOralesCh2.append(mfccsO)

	distEucCh2=[]
	for nas in serieNasalesCh2:
		for oral in serieOralesCh2:
			distEuc=eucDist(nas,oral)
			distEucCh2.append(distEuc)
			datoNuevo=[vocal,"Mouth",distEuc]
			distrEucMFCC.loc[len(distrEucMFCC)]=datoNuevo



distrEucMFCC.to_csv("MFFC_distance.txt",sep ='\t')
  
