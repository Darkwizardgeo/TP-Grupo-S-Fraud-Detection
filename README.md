# Trabajo Final Integrador Grupo S

![Image in a markdown cell](./assets/images/UTN-Logo.png)

## Diplomatura en Ciencia de Datos

### Integrantes:

    Sanchez, Jorge
    Sillet, Joseyra

#### Instrucciones para crear un entorno virtual y ejecutar Jupyter Notebook

Sigue los pasos a continuación para configurar un entorno virtual en Python y ejecutar un Jupyter Notebook:

#### 1. Crear un entorno virtual

1. Abre una terminal y navega al directorio donde deseas crear el entorno virtual.
2. Ejecuta el siguiente comando para crear el entorno virtual:

        python3 -m venv nombre_del_entorno

    Reemplaza `nombre_del_entorno` con el nombre que prefieras para tu entorno virtual.

#### 2. Activar el entorno virtual

- En sistemas **Linux/Mac**:
      ```
      source nombre_del_entorno/bin/activate
      ```
- En sistemas **Windows**:
      ```
      nombre_del_entorno\Scripts\activate
      ```

#### 3. Configuracion de los Dataset

Tenemos dos opciones:

- Se entrega con el repositorio dentro de la carpeta `dataset` un archivo de nombre 
`IEEE-fraud-dataset.tar.xz`, se puede descomprimir con alguna herramienta local o mediante cli. Los `.csv` luego deben ser subidos un nivel para que el acceso en la Jupiter notebook `/dataset/{archivo}.csv` sea el correcto.
- De presentar alguna dificultad en descomprimir el archivo se puede descargar desde Kaggle el dataset[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data), hay que suscribirse a la competencia.
