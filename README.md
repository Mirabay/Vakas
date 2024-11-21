# Inteligencia Artificial para el análisis de descanso de las vacas

Este proyecto incluye el código utilizado para entrenar un modelo de clasificación basado en redes neuronales, que identifica el estado de una cama entre las siguientes opciones:

- Cama vacía.
- Vaca acostada.
- Vaca de pie.

Además, se incluye el código con las funciones para cargar el modelo y utilizarlo en un entorno productivo, generando predicciones a partir de imagenes originales y guardando los resultados en un archivo CSV.

Los archivos están organizados de la siguiente manera:

- data_analysis. Contiene los archivos utilizados para realizar análisis de los datos una vez que se tiene el modelo entenado. Estos archivos no son necesarios para utilizar el modelo, pero son incluidos como una guía para analizar los datos en el futuro y como evidencia del trabajo realizado de análisis.

- deployment. Contiene los archivos que se necesitan para importar y utilizar el modelo. No contiene el archivo con los pesos del modelo entrenado. Este archivo se entrega por separado.

- training. Contiene los archvos necesarios para entrenar el modelo. Los imágenes utilizadas para el entrenamiento no se incluyen en este repositorio. También se incluye el archivo utilizado para evaluar los modelos.

- final-model. Contiene los pesos para generar nuestro modelo funcional.

**NOTA**. El repositorio no contiene todos los archivos necesarios para la ejecución de todos los códigos por limitaciones de GitHub en el tamaño de los archivos. No se incluyen los pesos generados de los modelos entrenados. Además, por cuestiones de la política de datos, no se incluyen las carpetas de imágenes utilizadas para clasificación.
