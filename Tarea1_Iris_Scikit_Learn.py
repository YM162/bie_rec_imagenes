# Importamos todo lo necesario para que no haga falta correr todo el codigo de ejemplo

from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_digits
from sklearn.utils import shuffle
from random import randrange
from numpy import array_equal

print("----- Prueba de precisión con DecisionTreeClassifier -----")
print("¿Que base de datos desea usar?")
print("1. Iris")
print("2. Wine")
print("3. Digits")
db = int(input())

# Dividimos en dos mitades aleatorias, iris_entreno e iris_test, el conjunto inicial
if db == 1:
  total = load_iris()
  entreno = load_iris()
  test = load_iris()
elif db == 2:
  total = load_wine()
  entreno = load_wine()
  test = load_wine()
elif db == 3:
  total = load_digits()
  entreno = load_digits()
  test = load_digits()
else:
  print("Error al seleccionar la base de datos, asegurese de introducir un numero (1-3)")
# La semilla nos permite mezclar los datos y los estados de la misma manera
seed = randrange(10000)
total.data = shuffle(total.data, random_state = seed)
total.target = shuffle(total.target, random_state = seed)

half = int(len(total.data)//2)
entreno.data = total.data[:half]
entreno.target = total.target[:half]
test.data = total.data[half:]
test.target = total.target[half:]

# Entrenamos un clasificador con los datos de entrenamiento y pasamos los datos de prueba
dtc = tree.DecisionTreeClassifier()
dtc.fit(entreno.data, entreno.target)
clasificador = dtc.predict(test.data)

# Desplegamos unas estadisticas sobre el rendimiento del clasificador
aciertos = 0
errores = 0
for i in range(len(clasificador)):
  if array_equal(clasificador[i],test.target[i]) == True:
    aciertos = aciertos+1
  else:
     errores = errores+1
precision = round(aciertos/len(clasificador)*100, 2)
print("----- Predicción completada para "+str(len(clasificador))+" elementos -----")
print("Aciertos: "+str(aciertos))
print("Errores: "+str(errores))
print("Precisión: "+str(precision)+"%")
