import pandas as pd
from sklearn import linear_model 
import matplotlib.pyplot as plt

#todo carga dinamica de archivo ()
datos = pd.read_csv("dolar.csv");
# print(datos["anio"].values)
    
#todo sacar dos columnas de archivo
plt.xlabel("Anio");
plt.ylabel("Robos");
datosPrimerColumna=datos[[datos.columns[0]]]
nombrePrimerColumna=datos.columns[0];
datosSegundaColumna=datos[[datos.columns[1]]]

plt.scatter(datosPrimerColumna, datosSegundaColumna, color="red", marker="+");
reg = linear_model.LinearRegression();

#Entreno mi modelo machine learning

reg.fit(datos[[nombrePrimerColumna]], datosSegundaColumna)
#pedir por parametro
predecir = 2020

num = reg.predict([[predecir]])
plt.plot(datosPrimerColumna, reg.predict(datos[[nombrePrimerColumna]]),color='blue')
plt.show()
#¿como lo calcula?
# Y = M * X + b   m=pendiente, x=x, b=intercept

print("Coeficiente: ", reg.coef_) #m
print("Intercepta en: ",reg.intercept_) #b
print("Prediccion hecha: ", num, " = ", reg.coef_ , " * ", predecir, " + " , reg.intercept_)