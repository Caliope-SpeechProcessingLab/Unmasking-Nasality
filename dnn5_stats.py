import numpy as np
import pandas as pd
import os
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
import researchpy as rp

rutaRelInformes="/resultados/"
baseDir=os.getcwd()
rutaAbsInformes=baseDir+rutaRelInformes

todos = os.listdir(rutaAbsInformes)

# Se filtran los que no terminan en _resumen_ASICA.txt
terminanTxt=[]

for f in todos:
#     if f.endswith("_resumen_ASICA.txt"):
     if f.endswith("_ASICA.txt"):
        terminanTxt.append(f)
terminanTxt.sort()

resumen_stat = pd.DataFrame(columns=['tipoAudio','corpus','fold','enunciados','rpearson','spearson'])

for f in terminanTxt:
    largo=len(f)
    fold=f[largo-15:-10]

    informe=rutaAbsInformes+f
    informeActual_df=pd.read_csv(informe)

    print(informe)

#   Del nombre del archivo se obtienen varios datos
    datos=f[0:-4]
    nDatos=len(datos.split('_'))
    if nDatos != 4:
        print("El nombre del informe tiene ",nDatos," datos, pero debería tener 3. Se ignora")
        print(datos) 
        continue

    tipoAudio=datos.split('_')[0]
    corpus=datos.split('_')[1]

    archivoRutaAbs=rutaAbsInformes+"/"+f
    informe_df = pd.read_csv(archivoRutaAbs, delimiter='\t') 

    listaDatos=["T5_Mathad"]

    for dato in listaDatos:
        fila3_Perc=[tipoAudio]
        pearson=informe_df[dato]
        pearsonr_Perc=pearson[len(pearson)-2]
        pearsons_Perc=pearson[len(pearson)-1]

        medida=dato
        fila3_Perc.append(corpus)
        fila3_Perc.append(fold)
        fila3_Perc.append(dato)
        fila3_Perc.append(pearsonr_Perc)
        fila3_Perc.append(pearsons_Perc)        
        resumen_stat.loc[len(resumen_stat)]=fila3_Perc

rutaNuevoInforme=rutaAbsInformes+"/"+"results_exp2.txt"
resumen_stat.to_csv(rutaNuevoInforme, sep ='\t',index=False)




