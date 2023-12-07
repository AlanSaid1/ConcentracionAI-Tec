#Alan Said Martinez Guzman A01746210

"""
    Importamos las librerias necesarias para el preprocesamiento de los datos
    y la visualizacion de los mismos. Ademas, importamos las librerias necesarias
    para la implementacion del algoritmo KNN.

"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings('ignore')

"""
    Importamos el dataset, es necesario primero descargar el dataset y ubicarlo en la misma
    carpeta que el archivo .py. Despues, verificamos las dimensiones del dataset y visualizamos.

"""
#leer el dataset
directorio_actual = os.getcwd()
ruta_archivo = os.path.join(directorio_actual, "Aprendizaje_MaquinaAI", "Aprendizaje_Maquina_Manual", "KNNAlgorithmDataset.csv")
df = pd.read_csv(ruta_archivo)

#ver las dimensiones del dataset
df.shape

#vista del dataset
df.head()

#distribución de frecuencia de los valores de la variable objetivo
for var in df.columns:
    print (df[var].value_counts())

#preprocesamiento de los datos ---------------------------------------------------------------------------------------------
"""
    La importancia del preprocessing es que nos ayuda a preparar los datos para el modelado, ya que los datos
    pueden contener valores faltantes, valores atípicos, valores categóricos, etc. Por lo tanto, es necesario
    realizar un preprocesamiento de los datos para que el modelo pueda aprender de los datos de manera eficiente.

"""
#eliminamos la columna id ya que no aporta información
df.drop(['id'],1,inplace=True)
df.info()

#eliminamos las variables que no aportan información
df.drop(['Unnamed: 32'],1,inplace=True)

#checar si hay valores nulos en las variables
print(df.isnull().sum())

#hacer label encoding a la columna diagnosis para poder trabajar con los datos
label_encoder = LabelEncoder()
df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])
print(df['diagnosis'].head())

#visualizacion de datos ----------------------------------------------------------------------------------------------------

"""
    La visualización de datos es una parte importante del análisis de datos. Nos ayuda a comprender los datos
    y a tomar decisiones basadas en los datos. La visualización de datos también nos ayuda a detectar valores
    atípicos y valores faltantes en los datos.

"""
#histograma de las variables
plt.figure(figsize=(20,20))
df.hist(bins=20)
plt.tight_layout()
plt.show()

#correlación entre las variables, mapa de calor
plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

#declarar la variable objetivo ---------------------------------------------------------------------------------------------
"""
    La variable objetivo es la variable que queremos predecir. En este caso, la variable objetivo es la columna
    diagnosis.
"""

X = df.drop(['diagnosis'],1)
y = df['diagnosis']

#MODELO KNN -----------------------------------------------------------------------------------------------------------------

def distancia_euclidiana(x1, x2):

    """
    Calcula la distancia euclidiana entre dos puntos. La distancia euclidiana es una métrica comúnmente utilizada
    para comparar la similitud entre dos puntos. Se define como la raíz cuadrada de la suma de las diferencias

    Parameters:
    x1 (numpy.ndarray): El primer punto.
    x2 (numpy.ndarray): El segundo punto.

    Returns:
    float: Toma dos numpy arrays como entrada y devuelve la distancia euclidiana como un valor flotante.
    """
    
    return np.sqrt(np.sum((x1 - x2)**2))

def mi_knn(X_train, y_train, x_test, k):
   
    """
    Implementa un clasificador K-Nearest Neighbors (KNN) personalizado para una sola instancia de prueba.

    Parameters:
    X_train (numpy.ndarray): Conjunto de datos de entrenamiento.
    y_train (numpy.ndarray): Etiquetas correspondientes a los datos de entrenamiento.
    x_test (numpy.ndarray): Instancia de prueba a clasificar.
    k (int): Número de vecinos a considerar en KNN.

    Returns:
    int: determina la etiqueta más común entre estos vecinos y devuelve esa etiqueta como la predicción para la instancia de prueba.
    """

    distances = [distancia_euclidiana(x_test, x_train) for x_train in X_train]
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [y_train[i] for i in k_indices]
    most_common = np.bincount(k_nearest_labels).argmax()
    return most_common

# Separar los datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hacer predicciones usando knn

k = 12  # Número de vecinos

"""
    Aqui observamos que a comparación de solo poner 5 vecinos, al poner 12 vecinos la precisión aumenta, esto se debe
    a que al poner mas vecinos, el algoritmo se vuelve mas preciso, ya que al tener mas vecinos, el algoritmo se vuelve
    mas robusto y no se ve afectado por los valores atípicos.
"""

predicciones = [mi_knn(X_train.to_numpy(), y_train.to_numpy(), x_test, k) for x_test in X_test.to_numpy()]

#Generalización del modelo -------------------------------------------------------------------------------------------------
"""
    Generalizamos el modelo. La validación cruzada es una técnica que se utiliza para evaluar la capacidad de
    generalización de un modelo. La validación cruzada divide los datos en k pliegues y utiliza k-1 pliegues
    para entrenar el modelo y el pliegue restante para evaluar el modelo. Este proceso se repite k veces y
    se calcula la media de las puntuaciones de validación cruzada.
"""

num_folds = 5

# Inicializa StratifiedKFold para dividir los datos
kf = StratifiedKFold(n_splits=num_folds)

scores = []

# Realiza la validación cruzada
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Realiza predicciones en X_test y calcula la precisión
    predicciones = [mi_knn(X_train.to_numpy(), y_train.to_numpy(), x_test, k) for x_test in X_test.to_numpy()]
    accuracy = np.mean(predicciones == y_test)
    
    # Almacena la precisión en la lista de puntuaciones
    scores.append(accuracy)

# Imprime las puntuaciones de validación cruzada
print("Puntuaciones de Validacion Cruzada:", scores)

# Calcula la media y la desviación estándar de las puntuaciones
print("Precision Promedio:", np.mean(scores))
print("Desviacion Estandar de la Precision:", np.std(scores))

#MÉTRICAS DE EVALUACIÓN ----------------------------------------------------------------------------------------------------
"""
    Las métricas de evaluación son medidas que se utilizan para determinar el rendimiento de un modelo. Las métricas
    de evaluación se utilizan para medir la precisión, la recuperación, el valor F1, etc.

"""

# Función para calcular la matriz de confusión
def matriz_confusion(y_true, y_pred):
    """
    Calcula la matriz de confusión.

    Parametros:
    y_true (numpy.ndarray): Etiquetas reales.
    y_pred (numpy.ndarray): Etiquetas predichas.

    Return:
    numpy.ndarray: Matriz de confusión. La matriz de confusión es una herramienta que permite la visualización del
    desempeño de un algoritmo que se emplea en aprendizaje supervisado. Cada columna de la matriz representa las
    instancias en una predicción, mientras que cada fila representa las instancias en una clase real.
    """
    unique_labels = np.unique(y_true)
    matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
    for i in range(len(unique_labels)):
        for j in range(len(unique_labels)):
            matrix[i, j] = np.sum((y_true == unique_labels[i]) & (y_pred == unique_labels[j]))
    return matrix
print("Matriz de confusion:\n", matriz_confusion(y_test, predicciones))

# Función para calcular el Valor F1
def f1_Valor(y_true, y_pred):
    """
    Calcula el Valor F1.

    Parameters:
    y_true (numpy.ndarray): Etiquetas reales.
    y_pred (numpy.ndarray): Etiquetas predichas.

    Returns:
    float: Valor F1. El Valor F1 es una métrica que combina la precisión
     y la recuperación para proporcionar una medida equilibrada del rendimiento del modelo.
    """
    cm = matriz_confusion(y_true, y_pred)
    tp = cm[1, 1]  # Verdaderos positivos
    fp = cm[0, 1]  # Falsos positivos
    fn = cm[1, 0]  # Falsos negativos
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
print("valor F1:", f1_Valor(y_test, predicciones))

# Función para calcular la Recuperación
def recall_score(y_true, y_pred):
    """
    Calcula la Recuperación (Recall).

    Parameters:
    y_true (numpy.ndarray): Etiquetas reales.
    y_pred (numpy.ndarray): Etiquetas predichas.

    Returns:
    float: Recuperación (Recall). La recuperación es la proporción de verdaderos positivos que se
    identifican correctamente.
    """
    cm = matriz_confusion(y_true, y_pred)
    tp = cm[1, 1]  # Verdaderos positivos
    fn = cm[1, 0]  # Falsos negativos
    recall = tp / (tp + fn)
    return recall
print("Recuperacion:", recall_score(y_test, predicciones))