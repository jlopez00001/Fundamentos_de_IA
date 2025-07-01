#=============================================
# Agrupamiento usando conjuntos de párticulas
#============================================
# FUDAMENTOS DE IA
# JOSE JULIO LOPEZ MARQUEZ
# ESFM IPN JUNIO 2025
#============================
import pandas as pd
import numpy as np
from pso_clustering import PSOClusteringSwarm

plot = True
#====================================
# Leer datos (hoja de datos de pandas)
#=====================================
data_points = pd.read_csv('iris.txt', sep=',', header=None)
#===========================================================================
# Pasar columna 4 (comienza en 0) a un arreglo de numpy (dataframe a numpy)
#=========================================================================
clusters = data_points[4].values
#=======================================================
# Remover columna 4 de los datos (método drop de pandas)
#=====================================================
data_points = data_points.drop([4], axis=1)
#========================================================================
# Usar columnas 0 y 1 como (x,y) para graficar puntos en 2D
#==================================================================
if plot:
    data_points = data_points[[0, 1]]
#================================
# Convierte a arreglo de numpy 2d
#=================================
data_points = data_points.values
#================================
# Algorutmo PSO-CLustering
#==============================
pso = PSOClusteringSwarm(n_clusters=3, n_particles=10, data=data_points, hybrid=True)
pso.start(iteration=1000, plot=plot)

#=========================================
# Mapeon de colores a elmentos de los grupos
#===========================================
mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
clusters = np.array([mapping[x] for x in clusters])
#print('Actual classes = ', clusters)