# Importamos todo lo necesario para que no haga falta correr todo el codigo de ejemplo

from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from random import randrange
from numpy import array_equal
# Dividimos en dos mitades aleatorias, iris_entreno e iris_test, el conjunto inicial
iris_total = load_iris()
# La semilla nos permite mezclar los datos y los estados de la misma manera
seed = randrange(10000)
iris_total.data = shuffle(iris_total.data, random_state = seed)
iris_total.target = shuffle(iris_total.target, random_state = seed)
half = int(len(iris_total.data)/2)
iris_entreno = load_iris()
iris_test = load_iris()
iris_entreno.data = iris_total.data[:half]
iris_entreno.target = iris_total.target[:half]
iris_test.data = iris_total.data[half:]
iris_test.target = iris_total.target[half:]
# Entrenamos un clasificador con los datos de entrenamiento y pasamos los datos de prueba
dtc = tree.DecisionTreeClassifier()
dtc.fit(iris_entreno.data, iris_entreno.target)
clasificador = dtc.predict(iris_test.data)
# Desplegamos unas estadisticas sobre el rendimiento del clasificador
aciertos = 0
errores = 0
for i in range(len(clasificador)):
  if array_equal(clasificador[i],iris_test.target[i]) == True:
    aciertos = aciertos+1
  else:
     errores = errores+1
precision = aciertos/len(clasificador)*100
print("----- Predicción completada -----")
print("Aciertos: "+str(aciertos))
print("Errores: "+str(errores))
print("Precisión: "+str(precision)+"%")
