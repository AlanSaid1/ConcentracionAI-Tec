#Alan Said Martínez Guzmán A01746210

"""
    Dentro de este dataset podemos encontrar indices de masa corporal que nos ayudan
    a determinar si una persona tiene sobrepeso o no, esto con base en su altura y peso.
    De la misma forma tomando en cuenta si es hombre o mujer.
"""
import os #libreria para el manejo de archivos
import numpy as np #libreria para el manejo de arreglos
import pandas as pd #libreria para el manejo de dataframes
import matplotlib.pyplot as plt #libreria para la visualizacion de datos
import seaborn as sns #libreria para la visualizacion de datos
import warnings #libreria para ignorar warnings

from sklearn import preprocessing #libreria para hacer el preprocesamiento necesario
from sklearn.neighbors import KNeighborsClassifier #libreria para implementar el algoritmo KNN
from sklearn.metrics import accuracy_score #libreria para calcular la precision del modelo
from sklearn.model_selection import train_test_split #libreria para dividir los datos en entrenamiento y prueba

from sklearn.model_selection import cross_val_score #libreria para generalizar el modelo
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score #libreria para las metricas de evaluacion

warnings.filterwarnings('ignore')

#leer el dataset
directorio_actual = os.getcwd()
ruta_archivo = os.path.join(directorio_actual, "Aprendizaje_MaquinaAI", "Aprendizaje_Maquina_Framework", "bmi.csv")
df = pd.read_csv(ruta_archivo)

print(" --------------------")
#Imprimimos los primeros 5 datos del dataset
print(df.head())
print(" --------------------")
#Checamos si existen valores nulos en el dataset
print(df.isnull().sum())
print(" --------------------")
#Checamos el tipo de dato de cada columna, para saber si es necesario hacer un preprocesamiento especifico
print(df.info())
print(" --------------------")
#checamos si existen datos duplicados
print(df.duplicated().sum())

#preprocesamiento de los datos ------------------------------------------------------------------------------

"""
    Al observar que en el dataset tenemos un tipo de dato objeto, es necesario hacer un preprocesamiento
    para poder trabajar con los datos, con enteros. En este caso, se hizo un label encoding para poder trabajar con los datos.
"""

#Label encoder
label_encoder = preprocessing.LabelEncoder()

columna_cambiar = ["Gender"]

df["Gender"] = label_encoder.fit_transform(df["Gender"])

"""
    Este pequeño 'for' hace que se impriman los valores unicos de cada columna, para ver si funcionó de manera correcta el label encoding.
    Ya que al cambiar valores string a enteros debía de quedar de forma que el 1 representara a los hombres y el 0 a las mujeres.
"""
for col in columna_cambiar:
    print(f"valores unicos en: {col}: {df[col].unique()}")
print(" --------------------")

print(df.info())

#KNN: Declarar la variable objetivo ---------------------------------------------------------------------------
"""
    La variable objetivo es la variable que queremos predecir. En este caso, la variable objetivo es la columna
    diagnosis. Por lo tanto, la variable objetivo es la columna index.
"""
X = df.drop(["Index"], axis=1)
y = df["Index"]

#KNN: Separar los datos en entrenamiento y prueba

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear un clasificador KNN
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Entrenar el modelo con los datos de entrenamiento
knn_classifier.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = knn_classifier.predict(X_test)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
print(" --------------------")
print(f"Precision del modelo KNN: {accuracy}")

#Generalizar el modelo -----------------------------------------------------------------------------------------
"""
    Generalizamos el modelo. La validación cruzada es una técnica que se utiliza para evaluar la capacidad de
    generalización de un modelo. La validación cruzada divide los datos en k pliegues y utiliza k-1 pliegues
    para entrenar el modelo y el pliegue restante para evaluar el modelo. Este proceso se repite k veces y
    se calcula la media de las puntuaciones de validación cruzada.
"""

knn = KNeighborsClassifier(n_neighbors=5)  # Aquí usa el número de vecinos que desees

# Realiza una validación cruzada de 5-folds (puedes ajustar el número de folds según sea necesario)
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')

# Imprime las puntuaciones de la validación cruzada
print("Puntuaciones de Validacion Cruzada:", scores)

# Calcula la media y la desviación estándar de las puntuaciones
print("Precision Promedio:", scores.mean())
print("Desviacion Estandar de la Precision:", scores.std())

#Metricas de evaluacion -----------------------------------------------------------------------------------------
"""
    Las métricas de evaluación son medidas que se utilizan para determinar el rendimiento de un modelo. Las métricas
    de evaluación se utilizan para medir la precisión, la recuperación, el valor F1, etc.

"""
# Calcular la precisión
precision = accuracy_score(y_test, y_pred)
print("Precision:", precision)

# Calcular la matriz de confusión
confusion = confusion_matrix(y_test, y_pred)
print("Matriz de Confusion:\n", confusion)

# Calcular la precisión, recuperación y valor F1
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Precision:", precision)
print("Recuperacion:", recall)
print("Valor F1:", f1)