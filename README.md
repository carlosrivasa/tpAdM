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
- Docker, el cual instala imágenes de Airflow, MLflow, MinIO, FastAPI, Redis
- Python >=3.11,<3.13 (requerido por vizdoom)  
- Numpy, Pandas, SciPy
- Matplotlib, Seaborn
- Scikit-Learn

<br>

# Instalación del entorno
Si estás en Linux o MacOS, en el archivo `.env`, reemplaza `AIRFLOW_UID` por el de tu usuario o alguno que consideres oportuno (para encontrar el UID, usa el comando `id -u <username>`). De lo contrario, Airflow dejará sus carpetas internas como root y no podrás subir DAGs (en `airflow/dags`) o plugins, etc.

# Crear los directorios necesarios
`mkdir -p airflow/{dags,logs,config,plugins,secrets}`

<br>

# Iniciar los servicios
`docker-compose --profile all up -d`

### OPCIONAL: En caso de falla en airflow, ejecutar:
`docker-compose down`
`docker rmi extending_airflow:latest`
`docker-compose --profile all build --no-cache`
#### y de nuevo
`docker-compose --profile all up -d`

<br>

# Probar los servicios
- Airflow: http://localhost:8080 (user: airflow, password: airflow)
- MinIO: http://localhost:9001 (access key: minio, secret key: minio123)
- MLflow: http://localhost:5050
- FastAPI: http://localhost:8800

## Troubleshooting
Para diagnosticar problemas en los servicios, consulta los logs correspondientes:

```bash
docker logs mlflow      # MLflow
docker logs airflow-webserver  # Airflow
docker logs fastapi     # FastAPI
docker logs minio       # MinIO
```

### Importante: Airflow
Una vez levantados los servicios, es necesario ejecutar los DAGs en Airflow para que se realicen las tareas de ETL, entrenamiento y evaluación del modelo. <br>
El orden de ejecución es:
1. `etl_pipeline` - Realiza la extracción y transformación de los datos, dejando los archivos preprocesados en el bucket de MinIO.
2. `train_model_pipeline` - Realiza el entrenamiento del modelo, búsqueda de hiperparámetros y validación, registrando los resultados en MLflow.

### Importante 2: FastAPI
Existen casos en los que FastAPI no se buildea correctamente en el primer intento. En caso de que al acceder a la API no se obtengan respuestas del endpoint `/predict`, se recomienda  destruir y luego reconstruir la imagen de FastAPI con los siguientes comandos:
`docker-compose down fastapi`
`docker-compose build --no-cache fastapi`
`docker-compose up -d fastapi`

<br>

## Componentes dentro de airflow
Con el objetivo de realizar el despliegue, el código está dividido en los siguientes componentes:

- `airflow/dags/etl_pipeline.py`: Dag con el Pipeline de ETL en interacción con MLflow y MinIO.
- `airflow/dags/refactor/etl.py`: ETL para leer el dataset desde s3 y preprocesar los datos, deja los archivos preprocesados en el bucket data, carpeta "preprocessed".
- `airflow/dags/train_model_pipeline.py`: Dag con el Pipeline de Train, búsqueda de hiperparámetros y validación del modelo.
- `airflow/dags/refactor/train_test_model.py`: Entrenamiento del modelo, búsqueda de hiperparámetros mediante un experimento de MLFlow y validación.

<br>

# Arquitectura del sistema
El sistema está compuesto por los siguientes módulos principales:
- Orquestación (Airflow + Redis)
- Modelado (MLflow)
- Almacenamiento (PostgreSQL + MinIO)
- Predicción (FastAPI)
<br>

Para más detalles, ver [docs/architecture.md](docs/architecture.md)