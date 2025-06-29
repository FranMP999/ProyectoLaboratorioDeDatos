# Utiliza una imagen base con Python instalado
FROM python:3.10-slim

#Establece el directorio de trabajo en el contenedor
WORKDIR /root/airflow

# Establece la variable de entorno AIRFLOW_HOME
ENV AIRFLOW_HOME=/root/airflow

# Actualiza los paquetes del sistema e instala curl
RUN apt-get update && apt-get install -y curl
RUN apt-get install -y  libgomp1 
RUN rm -rf /var/lib/apt/lists/*

# Instala las dependencias necesarias
RUN pip install apache-airflow==2.8.3
RUN pip install scikit-learn
RUN pip install lightgbm
RUN pip install pandas
RUN pip install pyarrow
RUN pip install numpy
RUN pip install optuna
RUN pip install shap
RUN pip install -U email-validator

# Inicializa la base de datos de Airflow
RUN airflow db init

# Expone el puerto 8080 para el servidor web de Airflow
EXPOSE 8080

# Crea el usuario admin de Airflow
RUN airflow users create --role Admin --username admin --email admin \
 --firstname admin --lastname admin --password admin

# Copia las carpetas necesarias al contenedor
COPY ./dags $AIRFLOW_HOME/dags
COPY ./train_logs $AIRFLOW_HOME/train_logs
COPY ./repositorio $AIRFLOW_HOME/repositorio
COPY ./datos_nuevos $AIRFLOW_HOME/datos_nuevos

# Comando para iniciar el servidor web y el scheduler
CMD ["sh", "-c", "airflow webserver -p 8080 & airflow scheduler"]
