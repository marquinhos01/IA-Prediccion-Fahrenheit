from sklearn import linear_model 
import pandas as pd


def devolverModelo():
    return linear_model.LinearRegression();

def cargarYentrenarModelo(modelo, archivo):
    datos = pd.read_csv(archivo)
    colum1=datos.columns[0]; #nombre columna
    colum2Datos=datos[[datos.columns[1]]] #datos de la columna
    modelo.fit(datos[[colum1]],colum2Datos);
    return datos

def predecir(modelo,valor):
    num = modelo.predict([[valor]])
    print(num)

