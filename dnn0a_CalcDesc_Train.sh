# Este script crea todas las BDs del estudio de nasalidad DNN

##Carpeta que contiene los audios
carpetaAudiosEspana='audiosTrain/espana/';
carpetaAudiosNinos='audiosTrain/ninos/';
carpetaAudiosChile='audiosTrain/chile/';
carpetaAudiosMasninos='audiosTrain/masNinos/';
carpetaAudiosNinos2='audiosTrain/ninos2/';

carpetaActual=${carpetaAudiosNinos2}
carpetaBaseDatos="BDsTrain/"

baseDir=$(pwd)  
carpetaBaseDatos=${baseDir}/${carpetaBaseDatos}

# # Datos de adultos de España

# nombreBD='AdultosEspana.csv'
# carpetaAudios=${carpetaAudiosEspana}
# serie=(nas1 nas2 nasSuma H4n) 
# serie=(nas1 nas2 nasSuma) 
# for serie in ${serie[@]}
# do
# 	nombreBD=${serie}_${nombreBD}
# 	rutaAudios=${baseDir}/${carpetaAudios}${serie}/

# 	echo nombreBD ${nombreBD}
# 	echo rutaAudios ${rutaAudios}
# 	echo carpetaBaseDatos ${carpetaBaseDatos}

# 	python -m dnn0a_CalcDesc_Train ${rutaAudios} ${carpetaBaseDatos} ${nombreBD}
# done

# # Datos de adultos de Niños España

# nombreBD='NinosEspana.csv'
# carpetaAudios=${carpetaAudiosNinos}
# serie=(nas1 nas2 nasSuma) 

# for serie in ${serie[@]}
# do
# 	nombreBDn=${serie}_${nombreBD}
# 	rutaAudios=${baseDir}/${carpetaAudios}${serie}/

# 	echo nombreBD ${nombreBDn}
# 	echo rutaAudios ${rutaAudios}
# 	echo carpetaBaseDatos ${carpetaBaseDatos}

# 	python -m dnn0a_CalcDesc_Train ${rutaAudios} ${carpetaBaseDatos} ${nombreBDn}
# done

# Datos de adultos de Chile

nombreBD='Ninos2.csv'
carpetaAudios=${carpetaActual}
serie=(nas2 nas1) 

for serie in ${serie[@]}
do
	nombreBDn=${serie}_${nombreBD}
	rutaAudios=${baseDir}/${carpetaAudios}${serie}/

	echo nombreBD ${nombreBDn}
	echo rutaAudios ${rutaAudios}${serie}
	echo carpetaBaseDatos ${carpetaBaseDatos}

	python -m dnn0a_CalcDesc_Train ${rutaAudios} ${carpetaBaseDatos} ${nombreBDn}
done

