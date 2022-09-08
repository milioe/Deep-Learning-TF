import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix

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




def plot_decision_boundary_circles(model, X, y):
  """
  Imprime la toma de decisión del modelo
  Esta función ha sido adaptada de
   1. CS231n - https://cs231n.github.io/neural-networks-case-study/
   2. Made with ML basics - https://github.com/GokuMohandas/MadeWithML/blob/main/notebooks/08_Neural_Networks.ipynb
  """
  # Define los ejes de decisión del plot y crea un meshgrid (un meshgrid recibe las matrices de coordenadas de los vectores de coordenadas)
  x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
  y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                       np.linspace(y_min, y_max, 100))
  
  # Crea los valores de X (Vamos a predecir sobre estos)
  x_in = np.c_[xx.ravel(), yy.ravel()] # revisa si la dimensión final del output del modelo. is es mas grande a 1, es multiclase
  
  # Realizamos un reshape de las predicciones para poder graficarlas
  y_pred = model.predict(x_in)

  # Revisa si hay problemas de multiclase
  if model.output_shape[-1] > 1: # revisa si la dimensión final del output del modelo. is es mas grande a 1, es multiclase
    print("doing multiclass classification...")
    # Reshape
    y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
  else:
    print("doing binary classifcation...")
    y_pred = np.round(np.max(y_pred, axis=1)).reshape(xx.shape)
  
  # Imprime boundary
  plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
  plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())




def plot_boundaries_circles(model, X_train, y_train, X_test, y_test):
  plt.figure(figsize=(12, 6)) # establece size de figura
  plt.subplot(1, 2, 1) # subplot 1 (en set de train)
  plt.title("Train")
  plot_decision_boundary_circles(model, X=X_train, y=y_train)
  plt.subplot(1, 2, 2) # subplot 2 (en set de test)
  plt.title("Test") # titulo
  plot_decision_boundary_circles(model, X=X_test, y=y_test)
  plt.show()



def plot_loss_curves(history):
  """
  Regresa plot de cada una de las curvas de pérdida
  """ 
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Pérdida
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();



def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15): 
  """
    Crea una matriz de confusion comparando las predicciones y las y_true

  Si el argumento de clases se ha ocupado, la matriz de confusión se etiquetara, sino,
  se devolverán n numeros correspondientes a n clases

  Args:
    y_true: Etiquetas reales
    y_pred: Arreglo de etiquetas de predicción (tienen que ser del mismo shape que y_true)
    classes: Arreglo con nombre de clases
    figsize: Tamaño de la figura
    text_size: Tamaó de texto
  
  Returns:
    Plot de matriz de confusión


  """  
  # Crea matriz de confusión
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0] # find the number of classes we're dealing with

  # Plot 
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues)
  fig.colorbar(cax)

  # Si existe lista de clases
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  
  # Etiquetas de los ejes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # Creación de ejes
         yticks=np.arange(n_classes), 
         xticklabels=labels, 
         yticklabels=labels)
  
  # Etiquetas del eje
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # Umbral de colores diferentes
  threshold = (cm.max() + cm.min()) / 2.

  # Texto de cada celda
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
             horizontalalignment="center",
             color="white" if cm[i, j] > threshold else "black",
             size=text_size)

  