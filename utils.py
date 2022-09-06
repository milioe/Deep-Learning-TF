import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, y):
  """
  Imprime la toma de decisión del modelo
  Esta función ha sido adaptada de
   1. CS231n - https://cs231n.github.io/neural-networks-case-study/
   2. Made with ML basics - https://github.com/GokuMohandas/MadeWithML/blob/main/notebooks/08_Neural_Networks.ipynb
  """
  # Define los ejes de decisión del plot y crea un meshgrid (un meshgrid recibe las matrices de coordenadas de los vectores de coordenadas)
  x_min, x_max = np.array(X).min() - 0.1, 1
  y_min, y_max = 0, np.array(X).max() + 0.1
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), # linspace regresa un espacio con numeros separados
                       np.linspace(y_min, y_max, 100))
  
  
  # Crea los valores de X (Vamos a predecir sobre estos)
  x_in = np.c_[xx.ravel(), yy.ravel()] # stacking de arreglos de 2D  https://numpy.org/devdocs/reference/generated/numpy.c_.html
  
  # Crea predicciones 
  y_pred = model.predict(x_in)

  # Revisa si es un problema de multiclase 
  if model.output_shape[-1] > 1: # revisa si la dimensión final del output del modelo. is es mas grande a 1, es multiclase
    print("doing multiclass classification...")
    # Realizamos un reshape de las predicciones para poder graficarlas
    y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
  else:
    print("doing binary classifcation...")
    y_pred = np.round(np.max(y_pred, axis=1)).reshape(xx.shape)
  
  # Imprime boundary
  plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
  plt.scatter(np.array(X)[:, 0], np.array(X)[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())



def plot_boundaries(model, X_train, y_train, X_test, y_test):
  plt.figure(figsize=(12, 6)) # establece size de figura
  plt.subplot(1, 2, 1) # subplot 1 (en set de train)
  plt.title("Train")
  plot_decision_boundary(model, X=X_train, y=y_train)
  plt.subplot(1, 2, 2) # subplot 2 (en set de test)
  plt.title("Test") # titulo
  plot_decision_boundary(model, X=X_test, y=y_test)
  plt.show()




  