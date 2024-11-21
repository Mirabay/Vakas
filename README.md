# Inteligencia Artificial para el análisis de descanso de las vacas
 
Este proyecto incluye el código utilizado para entrenar un modelo de clasificación basado en redes neuronales, que identifica el estado de una cama entre las siguientes opciones:

- Cama vacía.
- Vaca acostada.
- Vaca de pie.

Además, se incluye el código con las funciones para cargar el modelo y utilizarlo en un entorno productivo, generando predicciones a partir de imagenes originales y guardando los resultados en un archivo CSV.

Los archivos están organizados de la siguiente manera:

- docs. Contiene los reportes realizados durante las diferentes fases del desarrollo del proyecto, en el cual se utilizo como marco de trabajo CRISP-DM. LOS DOCUMENTOS DEBEN DESCARGARSE PARA ACCEDER A LOS HIPERVINCULOS.

- deploy-architecture. Contiene el script e imagenes de prueba con el que se probo en la Raspberry Pi 3.

- data_analysis. Contiene los archivos utilizados para realizar análisis de los datos una vez que se tiene el modelo entenado. Estos archivos no son necesarios para utilizar el modelo, pero son incluidos como una guía para analizar los datos en el futuro y como evidencia del trabajo realizado de análisis.

- deployment. Contiene los archivos que se necesitan para importar y utilizar el modelo. No contiene el archivo con los pesos del modelo entrenado. Este archivo se entrega por separado.

- training. Contiene los archvos necesarios para entrenar el modelo. Los imágenes utilizadas para el entrenamiento no se incluyen en este repositorio. También se incluye el archivo utilizado para evaluar los modelos.

- final-model. Los pesos del modelo final se pueden encontrar en ---> https://tecmx-my.sharepoint.com/:u:/g/personal/a01611795_tec_mx/EdDaQ_XB4lRGk-b0rv1RWt0B-aXwl_ZeWCH9kmFimmUGGQ?e=vGV9GC

**NOTA**. El repositorio no contiene todos los archivos necesarios para la ejecución de todos los códigos por limitaciones de GitHub en el tamaño de los archivos. No se incluyen los pesos generados de los modelos entrenados. Además, por cuestiones de la política de datos, no se incluyen las carpetas de imágenes utilizadas para clasificación.
