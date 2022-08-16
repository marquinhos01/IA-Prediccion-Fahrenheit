import sys
sys.path.append('./')

from model.linearregression import *
from view.linearregresionview import *

#creo modelo
modelo =  devolverModelo();

#cargo datos a mi modelo y lo entreno
datos = cargarYentrenarModelo(modelo,"./src/robos.csv");

datos_col1 =datos[[datos.columns[0]]]
datos_col2 =  datos[[datos.columns[1]]]

#dibujo mi recta
dibujar(datos_col1, datos_col2,modelo)

#predecir
predecir(modelo, 2022)