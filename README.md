<p align="center">
  <img src="https://github.com/Mirabay/Vakas/blob/readme/miscellaneous/logo_TC/logo_TC.png" alt="Descripción de la imagen" width="400" height="400">
</p>


 # 🐄 Inteligencia Artificial para el análisis del descanso de las vacas 🐄
 
Este proyecto incluye el código utilizado para entrenar un modelo de clasificación basado en redes neuronales, que identifica el estado de una cama entre las siguientes opciones:

- Cama vacía.
- Vaca acostada.
- Vaca de pie.

Además, se incluye el código con las funciones para cargar el modelo y utilizarlo en un entorno productivo, generando predicciones a partir de imagenes originales y guardando los resultados en un archivo CSV.

---

## 📁 Estructura del Proyecto

### 1. **`docs`** 📂
- Contiene reportes generados durante las diferentes fases del proyecto, utilizando el marco de trabajo **CRISP-DM**.
- Subcarpetas:
  - **`CRISP-DM`**: Reportes detallados de cada fase del proyecto.
  - **`Manuales`**: Guías y manuales desarrollados.
  - **`Data Security`**: Politicas y planes.
- **Nota:** Los documentos deben descargarse para acceder a los hipervínculos.

---

### 2. **`miscellaneous`** 🛠️
- Contiene las carpetas con los scripts utilizados para:
  - **`deploy-architecture`**: Se utilizo para realizar la prueba de arquitectura y confirmar que nuestra solucion funcionara en el entorno de CAETEC.
  - **`data_analysis`**: Scripts utilizados para analizar los resultados generados, en entorno real.
  - **`codes`**: Script para el split de nuestro dataset.
  - **`Logos`**: Logo utilizado para nuestro proyecto de TC.

---

### 3. **`deployment`** 🚀
- Archivos necesarios para importar y utilizar el modelo en un entorno productivo.
- Se encuentra la carpeta de  **`bed-classifer`** contiene las subcarpetas de:
  -  **`bed_classifer`**: Se encuentra la arquitectura para la CNN y sus funciones para correr dicha arquitectura.
  -  **`dist`**: Paquetes para descargar nuestros archivos con el comando pip install.
  -  **`test`**: Pruebas para saber que los paquetes funcionan correctamente.
- Link del repositorio para la UI disponible [aquí](https://github.com/OshkarVTec/seleccion_camas_ui).
- **Nota:** El archivo con los pesos del modelo entrenado no está incluido por limitaciones de tamaño.

---

 ### 4. **`training`** 🏋️‍♂️
- Archivos necesarios para entrenar el modelo.
- Incluye:
  - **`SimpleCNN.py`**:Se encuentra la arquitectura para la CNN.
  - **`TestModel.ipynb`**: Nos da  nuestra matriz de confusion para saber el desepeño de nuestro modelo.
  - **`Training.py`**: Es el script que entrena nuestro modelo de CNN.
- **Nota:** Las imágenes utilizadas para el entrenamiento no están incluidas debido a limitaciones de almacenamiento.

---

### 5. **`final-model`** 🏆
- Los pesos del modelo final se encuentran disponibles [aquí](https://tecmx-my.sharepoint.com/:u:/g/personal/a01611795_tec_mx/EdDaQ_XB4lRGk-b0rv1RWt0B-aXwl_ZeWCH9kmFimmUGGQ?e=vGV9GC).

---


## 💻 Ejemplo de Uso


**NOTA**. El repositorio no contiene todos los archivos necesarios para la ejecución de todos los códigos por limitaciones de GitHub en el tamaño de los archivos. No se incluyen los pesos generados de los modelos entrenados. Además, por cuestiones de la política de datos, no se incluyen las carpetas de imágenes utilizadas para clasificación.
