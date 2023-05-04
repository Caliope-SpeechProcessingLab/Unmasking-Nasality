# Este script deberá hacer el entrenamiento y testeo con BDs diferentes. PENDIENTE


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential

from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import SGDClassifier
import io, os
import sys
from pathlib import Path


import configparser


def testModel(df_test, model):
	x_test = df_test[mfccs].copy()
	model_pred = model.predict(x_test)
	return model_pred


def data_preprocessing_Train(df):
    
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
                                 'HayAlgo':'HayAlgoAhi', 'hayAlgo':'HayAlgoAhi','iphone':'CuatroTextos',
                                 'Nas':'Testeo'
                                })
    
    df['speaker'] = df['Row'].apply(lambda x: ''.join(x.split('_')[1:-3])) # evita errores con nombres duplicados
    
    return df

def data_preprocessing_Test(df):
    
    df['utt'] = df['Row'].apply(lambda x: x.split('_')[-2])
    
    df['speaker'] = df['Row'].apply(lambda x:  x.split('_')[-5])

    df['ventana'] = df['Row'].apply(lambda x: x.split('_')[-1])
    
    return df

###### Modelo Deep Network

def make_modelRNN(archivoConfigDNN, metrics='accuracy'):

    # Se carga la configuración 
    
    print("archivoConfigDNN: ",archivoConfigDNN)

    configDNN = configparser.ConfigParser()
    configDNN.read(archivoConfigDNN)

    seccionConfiguracion = configDNN['Configuracion']
    unidades1 = int(seccionConfiguracion['unidades1'])
    unidades2 = int(seccionConfiguracion['unidades2'])
    unidades3 = int(seccionConfiguracion['unidades3'])
    dropout1=float(seccionConfiguracion['dropout1'])
    dropout2=float(seccionConfiguracion['dropout2'])
    lrate=float(seccionConfiguracion['lrate'])
    activation1=seccionConfiguracion['activation1']
    activation2=seccionConfiguracion['activation2']
    activation3=seccionConfiguracion['activation3']
    activationOut=seccionConfiguracion['activationOut']
    numCapas=int(seccionConfiguracion['numCapas'])
    if numCapas<3 or numCapas>4:
        print("Error. El campo numCapas del archivo de configuración de la RN solo puede valer 3 o 4.")
        exit()
    elif numCapas==4:
        unidades4 = int(seccionConfiguracion['unidades4'])
        dropout3=float(seccionConfiguracion['dropout3'])
        activation4=seccionConfiguracion['activation4']

    seccionEntrenamiento = configDNN['Entrenamiento']
    numEpoch = int(seccionEntrenamiento['numEpoch'])
    batchSize = int(seccionEntrenamiento['batchSize'])

    seccionBDTrain = configDNN['BDTrain']
    num_class_out = int(seccionBDTrain['numClasses'])
    bdTrainRelativa = seccionBDTrain['bdTrain']
    mfcc0 = int(seccionBDTrain['mfcc0'])
    nCanales = int(seccionBDTrain['nCanales'])

    baseDir=os.getcwd()
    bdTrain=os.path.join(baseDir,bdTrainRelativa)

    model = Sequential()

    model = Sequential()
    model.add(Dense(units = unidades1, activation=activation1))
    model.add(keras.layers.Dropout(dropout1)) 
    model.add(Dense(units = unidades2, activation = activation2))
    model.add(keras.layers.Dropout(dropout2)) 
    model.add(Dense(units = unidades3, activation = activation3))
    if numCapas==4:
        model.add(keras.layers.Dropout(dropout3)) 
        model.add(Dense(units = unidades4, activation = activation4))
    model.add(Dense(units = num_class_out, activation = activationOut))
    
    if not metrics:
        model.compile(
          optimizer=keras.optimizers.Adam(learning_rate=lrate), 
          loss='categorical_crossentropy')
    else:
        model.compile(
          optimizer=keras.optimizers.Adam(learning_rate=lrate), 
          loss='categorical_crossentropy',
          metrics=metrics)
    # end if 

    print("Cargado el modelo DNN")

    return model, bdTrain, numEpoch, batchSize, num_class_out, mfcc0, nCanales

###### Deep Network TRAIN
    

def testRNN(model,df_test):

    y_pred = model.predict(df_test)

    return y_pred


def main(rutaModelo,BDTest,rutaInforme,archivoConfigDNN):


    # Instancia del modelo

    modelRNN, BDTrain, num_epochs, num_batch_size, numClasses, mfcc0, nCanales = make_modelRNN(archivoConfigDNN, metrics='accuracy')

    print("Modelo RNN creado")

    # Carga pesos

    checkpoint_dir = Path(rutaModelo)

    print("checkpoint_dir:", checkpoint_dir)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=rutaModelo, 
        verbose=1, 
        save_weights_only=True,
        save_freq=200*num_batch_size)

    print("checkpoint_dir: ",checkpoint_dir)
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    print("Latest: ",latest)

    # latest = "resultadosCheckPoint/cp-0054.ckpt"

    modelRNN.load_weights(latest)

    print("Modelo RNN cargado con pesos anteriores: ",str(latest))

    train_df = pd.read_csv(BDTrain, delimiter=',')
    train_df = data_preprocessing_Train(train_df)

    test_df = pd.read_csv(BDTest, delimiter=',')
    test_df = data_preprocessing_Test(test_df)

    columns_to_copy_Test = ['speaker', 'utt','ventana']

    testResult= test_df[columns_to_copy_Test].copy()

    if mfcc0==1:
        primerMfcc=1
        primerDelta=14
        primerDeltaDelta=27
        primerMfcc2=40
        primerDelta2=53
        primerDeltaDelta2=66
    elif mfcc0==0: # Si se quiere ignorar la energía: mfcc0, delta0 y deltadelta0
        primerMfcc=2
        primerDelta=15
        primerDeltaDelta=28
        primerMfcc2=41
        primerDelta2=54
        primerDeltaDelta2=67
    else:
        print("Aquí nunca debería llegar")

    mfcc_names = list(train_df.columns[primerMfcc:14])
    delta_names = list(train_df.columns[primerDelta:27])
    deltaDelta_names = list(train_df.columns[primerDeltaDelta:40])

    mfccs = mfcc_names
    mfccs += delta_names
    mfccs += deltaDelta_names

    if nCanales==2: # Si la señal tiene dos canales hay que cargar las columnas del segundo canal
        mfcc_names2 = list(train_df.columns[primerMfcc2:primerMfcc2+12])
        delta_names2 = list(train_df.columns[primerDelta2:primerDelta2+12])
        deltaDelta_names2 = list(train_df.columns[primerDeltaDelta2:primerDeltaDelta2+12])
        mfccs += mfccs+mfcc_names2
        mfccs += mfccs+delta_names2
        mfccs += mfccs+deltaDelta_names2

    if numClasses==3:
        class_names = train_df['silence_nasal_oral'].unique().tolist()
    elif numClasses==5:
        class_names = train_df['class'].unique().tolist()
    else:
        print("El número de clases solo puede ser 3 o 5")
        exit()

    speaker_names = test_df['speaker'].unique()

    mfccsTest = mfccs 

    x_train = train_df[mfccsTest].copy()
    x_test = test_df[mfccsTest].copy()

    encoder_dnn = LabelBinarizer()
    
    encoder_dnn.fit(class_names)

    scaler = MinMaxScaler()
    x_train = scaler.fit(x_train)
    x_test=scaler.transform(x_test)

    y_pred_dnn= testRNN(modelRNN,x_test)

    y_pred_dnn_df = pd.DataFrame(y_pred_dnn)

    if numClasses==3:
        y_pred_dnn_df.rename(columns={0: 'PPNasal',1: 'PPOral',2: 'PPSil'},inplace=True) 
    elif numClasses==5:
        y_pred_dnn_df.rename(columns={0: 'PPNC', 1: 'PPNV',2: 'PPOC',3: 'PPOV',4: 'PPSil'},inplace=True) 

    indice=range(len(y_pred_dnn_df))

    y_pred_dnn_df.insert(0,'Row', indice)
  
    y_pred_dnn_df=y_pred_dnn_df.assign(nClasses=numClasses)

#    y_pred_dnn_str = calculaClaseResultado(y_pred_dnn, numClasses)

#    class_names=list(set(y_pred_dnn_str))

    print("Modelo RNN evaluado")

 #   testResult.insert(0,'Prediccion', y_pred_dnn_str)

    numVentanas = len(y_pred_dnn_df) 

    testResult.insert(0,'Row', indice)

    print("Insertado indice")

    suma_df=pd.merge(y_pred_dnn_df,testResult)

    print("Combinadas dataframes")
 
    rutaInformeDetallado=rutaInforme[0:-4]+".csv"

    print("rutaInformeDetallado: ",rutaInformeDetallado)

    suma_df.to_csv(rutaInformeDetallado, index=False)

    print("Guardado el informe")

# end def


if (len(sys.argv)!=5):
    print("Sintaxis incorrecta. Debe ser: ")
    print("python -m dnn_Load_and_Test rutaCompletaModeloDNN rutaCompletaBDTesteo rutaInformeResultados numCanales")
    exit();

rutaModelo=os.path.dirname(sys.argv[1])
print("Modelo: ", rutaModelo)
if os.path.exists(sys.argv[1]):
        print("Se va a cargar el modelo: ",sys.argv[1])
else:
    print("El modelo indicado no existe: ",sys.argv[1])
    exit()

BDTest=sys.argv[2]
if os.path.exists(sys.argv[2]):
        print("Se va a realizar una evaluación de la base de datos: ",sys.argv[2])
else:
    print("La BD para test no existe: ",sys.argv[2])
    exit()

rutaInforme=sys.argv[3]

if rutaModelo[:-1].endswith("/kfold"):
    rModelo=rutaModelo[:-7]
else:
    rModelo=rutaModelo

print("rModelo: ",rModelo)

nombreModelo=os.path.basename(rModelo)

print("Nombre modelo: ",nombreModelo)

configRNN=rModelo+"/"+nombreModelo+".cfg"

print("ConfigRNN: ",configRNN)

if os.path.exists(configRNN):
    print("Se va a cargar la configuración de la RN: ",configRNN)
else:
    print("la configuración de la RN no existe: ",configRNN)


main(rutaModelo, BDTest, rutaInforme, configRNN)


