# Este script deberá hacer el entrenamiento y testeo con BDs diferentes. PENDIENTE

import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import SGDClassifier
import io
import os
import configparser
import shutil


def show_results_class(n_fold,path_data, name_data, df_in, model, y_val, y_pred, c_names):
    cf_matrix = confusion_matrix(y_val, y_pred, labels=c_names)
    salida=show_confusion_matrix(n_fold,cf_matrix, c_names, model, path_data, name_data)

    modeloStr=str(model)
    modeloStr=modeloStr.split('(')[0]
    salida=modeloStr+"  "+acc_by_class(cf_matrix,c_names)
    return salida

def lista_a_cadena(lista):
    cadena=str(lista[0])
    lista1=lista[1:]
    for elem in lista1:
        cadena=cadena+'\t'
        cadena=cadena+str(elem) 
    print("cadena: ",cadena)

    return cadena


def show_confusion_matrix(n_fold,cf_matrix, class_names,model,path_data,name_data):


    plt.figure(figsize=(10,6))

    row_sum = np.sum(cf_matrix,axis=1).reshape(-1, 1)
    s = sns.heatmap(cf_matrix/row_sum, 
                fmt='.2%', annot=True, cmap='Greens', 
                yticklabels=class_names, xticklabels=class_names)

    s.set_ylabel('Etiqueta real', fontsize=14)
    s.set_xlabel('Etiq. predecida', fontsize=14)

    modeloStr=str(model)
    modeloStr=modeloStr.split('(')[0]

    fname=path_data+"/figCM_"+name_data+"_"+modeloStr+".png"
    print("Voy a guardar la figura: ",fname)

    plt.savefig(fname, dpi = None, facecolor = 'w', format = 'jpeg', bbox_inches = None, pad_inches = 0.1)


    rutaCompletaInforme=path_data[:-6]+"informe.txt"

    with io.open(rutaCompletaInforme, mode='a') as f:
        cadena=""
        if n_fold==0:
            f.write("\n************************\n")
            cadena=path_data[0:-7]+"\n"
        cadena=cadena+"\nkfold"+str(n_fold)+"\n"
        f.write(cadena)
        cadena=lista_a_cadena(class_names) 
        f.write(cadena)
        f.write("\n")
        for fila in cf_matrix:
            cadena=lista_a_cadena(fila) 
            f.write(cadena)
            f.write("\n")
    return 0

def acc_by_class(cf_matrix,class_names):
    acc_class = cf_matrix.diagonal()/cf_matrix.sum(axis=1)

    salida=""
    for label,acc_value in zip(class_names,acc_class):
        #salida=salida+label+"  "
        valor=round(acc_value*100)
        salida=salida+str(valor)+"  "

    return(salida)


def trainRNN(nCanales,mfcc0,archivoConfigDNN,k_fold,path_modelosDNN, path_dataTrain, 
    train_data, df_in, speaker_names, class_names):

    if k_fold == 1:
        kf = KFold(n_splits=5, shuffle=True, random_state = 42)
    else:
        kf = KFold(n_splits=k_fold, shuffle=True, random_state = 42)

    # Codificación clases con one-hot 
    encoder = LabelBinarizer()
    encoder.fit(class_names)

    print("Todos los locutores:",speaker_names)
    # print("Train index: ",train_index)

    #print("Calculando k-fold:\n")
    for n_fold, (train_index, val_index) in enumerate(kf.split(speaker_names)): 

        model, nEpochs, nBatch_size, patienceEarlyStopping = make_modelRNN(archivoConfigDNN,
            num_class_out=len(class_names), metrics='accuracy')

        print("BD: ",train_data)
        print("kfold: ",n_fold)
        print("Modelo RNN creado")
        print("PatienceEarlyStopping: ",patienceEarlyStopping)

    # Variables para guardar los resultados
        historial = []

        if train_data.endswith(".csv"):
            train_data=train_data[:-4]

        if k_fold==1:
            carpetaCheckPointkfold=path_modelosDNN+"/"+train_data
        else:
            carpetaCheckPointkfold=path_modelosDNN+"/"+train_data+"/kfold"+str(n_fold)

        if os.path.exists(carpetaCheckPointkfold):
            print("Ya existe la carpeta de checkpoint")
        else:
            os.makedirs(carpetaCheckPointkfold)
            print("Se crea una carpeta nueva de checkpoint")
        
        # Include the epoch in the file name (uses `str.format`)
        checkpoint_path = carpetaCheckPointkfold+"/"+"/cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create a callback that saves the model's weights every X batchSize

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, 
            verbose=1, 
            save_weights_only=True,
            save_freq=100*nBatch_size,
            save_best_only=True)

        es = EarlyStopping(monitor='val_loss', patience=patienceEarlyStopping)
        # Train the DNN

        callbacks=[cp_callback, es]

        # En el primer k_fold se hace una copia del archivo de configuración a la que se le añade info de esta prueba
        if n_fold==1:
            nomArchivoCFG=nomArchivo+".cfg"
            carpetaModelo=path_modelosDNN+"/"+train_data
            rutaCompletaConfigFile=os.path.join(carpetaModelo,nomArchivoCFG)
            shutil.copy(archivoConfigDNN,rutaCompletaConfigFile)

            baseDir=os.getcwd()
            ini=len(baseDir)+1
            rutaAbsolutaBDTrain=path_dataTrain+"/"+train_data
            rutaRelativaBDTrain=rutaAbsolutaBDTrain[ini:]
            with io.open(rutaCompletaConfigFile, mode='a') as f:
                f.write('\n\n')
                f.write("[BDTrain]\n")
                f.write("bdTrain="+rutaRelativaBDTrain+".csv\n")
                cadena="numClasses="+str(len(class_names))+"\n"
                cadena=cadena+"mfcc0="+str(mfcc0)+"\n"
                cadena=cadena+"nCanales="+str(nCanales)+"\n"
                f.write(cadena)


        # Se separan por apellidos
        df_train = df_in[pd.DataFrame(df_in['speaker'].tolist()).isin(speaker_names[train_index]).any(1).values]
        df_val = df_in[pd.DataFrame(df_in['speaker'].tolist()).isin(speaker_names[val_index]).any(1).values]

        # df_train.to_csv(nom, index=False)
        # Se generan los datos
        X_train = df_train.drop(['class','speaker'],axis=1)
        X_val = df_val.drop(['class','speaker'],axis=1)

        y_train = df_train['class'].to_numpy()
        y_val = df_val['class'].to_numpy()

        # Normalización
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Codificación clases con one-hot 
        y_train = encoder.transform(y_train)
        y_val = encoder.transform(y_val)
        
        # Peso de las clases
        y_integers = np.argmax(y_train, axis=1)
        class_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(y_integers), y=y_integers)
        d_class_weights = dict(enumerate(class_weights))

        # Entrenamiento y evaluación
        model.fit(X_train, y_train,
                    batch_size=nBatch_size,
                    epochs=nEpochs, 
                    verbose=1, 
                    validation_data=(X_val,y_val),
                    shuffle=True,
                    callbacks=callbacks
                    )
        historial.append(model.history)

        print("K fold RNN entrenado")

        # list all data in history

        model.save_weights(checkpoint_path.format(epoch=0))

        model_loss = pd.DataFrame(historial[0].history)
        model_loss.plot()

        nomArchivo=os.path.splitext(train_data)[0]

        if nomArchivo.endswith(".csv"):
            nomArchivo=nomArchivo[:-4]

        fname=carpetaCheckPointkfold+"/FigModelLoss_"+nomArchivo+".png"
        print("Voy a guardar la figura: ",fname)
        plt.savefig(fname, dpi = None, facecolor = 'w', format = 'jpeg', bbox_inches = None, pad_inches = 0.1)

        print("Pesos e historial guardados")

        y_pred = model.predict(X_val)
        y_val_str = encoder.inverse_transform(y_val)
        y_pred_str = encoder.inverse_transform(y_pred)

        out=show_results_class(n_fold,carpetaCheckPointkfold, train_data, df_in=[], model="DNN", y_val=y_val_str, y_pred=y_pred_str, c_names=class_names)

    # Se copia el archivo de configuración y se anota la BD de entrenamiento y otros parámetros

        if k_fold == 1: 
            break

    return model

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
    # df['text'] = df['Row'].apply(lambda x: x.split('_')[-3])
    # df['text'] = df['text'].replace({'Vanero':'Joyero', 'HOM':'Joyero', 'joyero':'Joyero',
    #                              'VientoN':'Viento', 'VientoV':'Viento', 'ElViento':'Viento',
    #                              'HayAlgo':'HayAlgoAhi', 'hayAlgo':'HayAlgoAhi','iphone':'CuatroTextos',
    #                              'Nas':'Testeo'
    #                             })
    
#    df['speaker'] = df['Row'].apply(lambda x: ''.join(x.split('_')[1:-3])) # evita errores con nombres duplicados
    df['speaker'] = df['Row'].apply(lambda x: ''.join(x.split('_')[-4])) # evita errores con nombres duplicados
    
    return df

# Esta función está repetida en dnn_Load_and_save.py
# Se podría sacar a un .py aparte o 
# importar este .py desde dnn_Load_and_save.py

def make_modelRNN(archivoConfigDNN, num_class_out, metrics='accuracy'):

    # Se carga la configuración del archivo de configuración

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
    patienceEarlyStopping = int(seccionEntrenamiento['earlyStoppingPatience'])

# Se crea el modelo

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

    # Se devuelve el modelo así como las variables numEpoch, batchSize, patienceEarlyStopping
    # obtenidas del archivo de configuración que se usarán para el entrenamiento
    return model, numEpoch, batchSize, patienceEarlyStopping

# end def mak_model


###### Deep Network TRAIN
    
def main(path_dataTrain,train_data,rutaModelosDNN,archivoConfigDNN,numClases,numkFold,mfcc0,nCanales) -> int:

    if train_data.endswith(".csv"):
        nombreBD=train_data[:-4]
    else:
        nombreBD=train_data

    carpetaCheckPoint=rutaModelosDNN+"/"+nombreBD
    
    if os.path.exists(carpetaCheckPoint):
        print("Ya existe la carpeta de checkpoint.")
    else:
        os.makedirs(carpetaCheckPoint)
        print("Se crea una carpeta nueva de checkpoint")


    # train data
    train_df = pd.read_csv(os.path.join(path_dataTrain, train_data), delimiter = ',')
    train_df = data_preprocessing_Train(train_df)

    if mfcc0==0: # se ignora el mfcc0, delta0 y deltadelta0
        primerMfcc=2
        primerDelta=15
        primerDeltaDelta=28
        primerMfcc2=41
        primerDelta2=54
        primerDeltaDelta2=67
    elif mfcc0==1:
        primerMfcc=1
        primerDelta=14
        primerDeltaDelta=27
        primerMfcc2=40
        primerDelta2=53
        primerDeltaDelta2=66
    else:
        print("Aquí nunca debería llegar")

    mfcc_names = list(train_df.columns[primerMfcc:14])
    delta_names = list(train_df.columns[primerDelta:27])
    deltaDelta_names = list(train_df.columns[primerDeltaDelta:40])
    mfcc_names = list(train_df.columns[primerMfcc:14])
    delta_names = list(train_df.columns[primerDelta:27])
    deltaDelta_names = list(train_df.columns[primerDeltaDelta:40])

    mfccs = mfcc_names
    mfccs += delta_names
    mfccs += deltaDelta_names

    if nCanales==2:
        mfcc_names2 = list(train_df.columns[primerMfcc2:primerMfcc2+12])
        delta_names2 = list(train_df.columns[primerDelta2:primerDelta2+12])
        deltaDelta_names2 = list(train_df.columns[primerDeltaDelta2:primerDeltaDelta2+12])
        mfccs += mfccs+mfcc_names2
        mfccs += mfccs+delta_names2
        mfccs += mfccs+deltaDelta_names2

    if numClases==3:
        mfccsTrain = mfccs + ['speaker', 'silence_nasal_oral']
        x_train = train_df[mfccsTrain].copy()
        x_train.columns = [*x_train.columns[:-1], 'class'] # last columns always has name class
    elif numClases==5:
        mfccsTrain = mfccs + ['speaker', 'class']
        x_train = train_df[mfccsTrain].copy()

    ########################
    #  RNN
    ########################

    df_in = x_train; # df_redux df_balanced
    class_names = df_in['class'].unique().tolist()
    if numClases != len(class_names):
        print("Se han especificado ",numClases," en el archivo de configuración,")
        print("pero en la BD hay ",len(class_names)," clases diferentes: ")
        print(class_names)
        exit()
    speaker_names = df_in['speaker'].unique()

    # Se creea y entrena modelo

    modelRNN = trainRNN(nCanales,mfcc0,archivoConfigDNN,numkFold,rutaModelosDNN, path_dataTrain, 
        train_data, df_in=df_in, speaker_names=speaker_names, class_names=class_names)


    return 0

# if __name__ == "__main__":
#     print("Número de parámetros: ", len(sys.argv))
#     print("Lista de argumentos: ", sys.argv)

if (len(sys.argv)!=8):
    print("Debes poner como parámetros:")
    print("\t - la ruta completa con el nombre de la BDs de entrenamiento,")
    print("\t - el nombre de la carpeta en la que se guardará el modelo de DNN")
    print("\t - el nombre del archivo de configuración de la DNN")
    print("\t - el número de clases")
    print("\t - si debe usarse o no mfcc0")
    print("\t - si la señal tiene uno o dos canales ")
    exit();

if os.path.exists(sys.argv[1]):
    nombreBD=os.path.basename(sys.argv[1])
    rutaBD=os.path.dirname(sys.argv[1])
    print("Se va a procesar la base de datos: ",sys.argv[1])
else:
    print("La BD indicada no existe: ",sys.argv[1])
    exit()

if os.path.exists(sys.argv[2]):
    print("El modelo DNN y la configuración se guardarán en la subcarpeta: ",sys.argv[2])
else:
    print("Se crea una nueva subcarpeta para guardar los modelos: ",sys.argv[2])
    os.mkdir(sys.argv[2])
rutaModelo=sys.argv[2]

if os.path.exists(sys.argv[3]):
        archivoConfigDNN=sys.argv[3]
        print("Archivo de configuración de la DNN: ",archivoConfigDNN)
else:
    print("No se ha encontrado el archivo de configuración: ",sys.argv[3])
    exit()

numClases=int(sys.argv[4])
if numClases==3:
    print("Se usarán 3 clases: oral, nasal, silence")
elif numClases==5:
    print("Se usarán 5 clases: OV, OC, NV, NC, SIL")
else:
    print("El parámetro numClases vale: ",sys.argv[4])
    print("Solo pueden especificarse 3 o 5 clases")
    exit()

numkFold=int(sys.argv[5])
if numkFold==1:
    print("Se creará un modelo")
elif numkFold==5:
    print("Se creará cinco modelos, en las subcarpetas: kfoldn")
else:
    print("kfold solo puede valer 1 o 5")
    exit()

mfcc0=int(sys.argv[6])

if mfcc0==1:
    print("Sí se incluirá mfcc0 como parámetro")
elif mfcc0==0:
    print("NO se inclurá mfcc0 como parámetro")
else:
    print("mfcc0 vale: ",mfcc0)
    print("mfcc0 solo puede ser 0 o 1")
    exit()
nCanales=int(sys.argv[7])

if nCanales==1:
    print("La señal tiene un canal")
elif nCanales==2:
    print("Señal de 2 canales")
else:
    print("El número de canales vale: ",nCanales)
    print("el parámetro nCanales solo puede ser 1 o 2")
    exit()


main(rutaBD, nombreBD, rutaModelo, archivoConfigDNN,numClases,numkFold,mfcc0,nCanales)

