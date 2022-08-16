import matplotlib.pyplot as plt


def dibujar(dataX, dataY, modelo):
    plt.scatter(dataX, dataY, color="red", marker="+");
    num = modelo.predict(dataX)
    plt.plot(dataX, num,color='blue');
    plt.show()
    # print("Coeficiente: ", modelo.coef_) #m
    # print("Intercepta en: ",modelo.intercept_) #b