# Universidad de Buenos Aires
### Carrera de Especialización en Inteligencia Artificial
### Cohorte 22 - Año 2025

<br>

# Operaciones de aprendizaje de máquina
Este repositorio contiene el material de resolución del trabajo práctico final de la materia.

## Integrantes
- [SIU a2221] Cesar Octavio Mejia <cemejia555@gmail.com>
- [SIU a2222] Osvaldo Daniel Muñoz <ossiemunoz@gmail.com>
- [SIU a2227] Carlos Alberto Rivas Araque <carlos.rivas.a@gmail.com>
- [SIU a2208] Ariel Matias Cabello <arielmcabello@gmail.com>
- [SIU a2213] Ignacio Agustin Costarelli <agustin@costarellisa.com.ar>
- [SIU a2214] Alex Martín Curellich <alexcurellich@gmail.com>

## Pase a producción de un modelo de predicciones (win no win) de partidos de futbol de selecciones nacionales
- [FIFA2026_win_nowin.ipynb](notebook_examples/FIFA2026_win_nowin.ipynb) - Predicciones definitivas

<br>

# Requerimientos
- Docker, Airflow, MLflow, MinIO, FastAPI, Redis, 
- Python >=3.11,<3.13 (requerido por vizdoom)  
- Numpy, Pandas, SciPy
- Matplotlib, Seaborn
- Scikit-Learn

# Instalación del entorno

Si estás en Linux o MacOS, en el archivo `.env`, reemplaza `AIRFLOW_UID` por el de tu usuario o alguno que consideres oportuno (para encontrar el UID, usa el comando `id -u <username>`). De lo contrario, Airflow dejará sus carpetas internas como root y no podrás subir DAGs (en `airflow/dags`) o plugins, etc.

# Crear los directorios necesarios
mkdir -p airflow/{dags,logs,config,plugins,secrets}

# Iniciar los servicios
docker-compose --profile all up -d 

# OPTCIONAL: En caso de falla en airflow, ejecutar:
docker-compose down
docker rmi extending_airflow:latest
docker-compose --profile all build --no-cache
# y de nuevo
docker-compose --profile all up -d

# Probar los servicios
Airflow: http://localhost:8080 (user: airflow, password: airflow)
MinIO: http://localhost:9001 (access key: minio, secret key: minio123)
MLflow: http://localhost:5050
FastAPI: http://localhost:8800

# OPCIONAL: ver logs para troubleshooting, por ejemplo hay alguna falla en mlflow
docker logs mlflow

# Importante a continuación:
Entrar a MinIO y subir el archivo archives/results.csv en el bucket data, carpeta "datasets"
También se crean en el bucket data las carpetas output y processed  

## Componentes dentro de airflow

Con el objetivo de realizar el despliegue, el código está dividido en los siguientes componentes:

- `airflow/dags/etl_pipeline.py`: Dag con el Pipeline de ETL en interacción con MLflow y MinIO.
- `airflow/dags/refactor/etl.py`: ETL para leer el dataset desde s3 y preprocesar los datos, deja los archivos preprocesados en el bucket data, carpeta "preprocessed".

# WIP 
- `airflow/dags/refactor/train_model.py`: Entrenamiento del modelo, se contempla usar como parte de siguientes pasos.

# WIP
- `airflow/dags/refactor/test_model.py`: Testeo del modelo, se contempla usar como parte de siguientes pasos.

