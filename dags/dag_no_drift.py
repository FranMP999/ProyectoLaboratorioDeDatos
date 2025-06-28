'''
Este script debe contener la siguiente estructura:

1. (1 punto) Inicialice un DAG con fecha de inicio el 1 de octubre de 2024, el cual se debe ejecutar el día 5 de cada mes a las 15:00 UTC. Utilice un `dag_id` interpretable para identificar fácilmente. **Habilite el backfill** para que pueda ejecutar tareas programadas desde fechas pasadas.
2. (1 punto) Comience con un marcador de posición que indique el inicio del pipeline.
3. (2 puntos) Cree una carpeta correspondiente a la ejecución del pipeline y cree las subcarpetas `raw`, `preprocessed`, `splits` y `models` mediante la función `create_folders()`.
4. (2 puntos) Implemente un `Branching`que siga la siguiente lógica:
  - Fechas previas al 1 de noviembre de 2024: Se descarga solo `data_1.csv`
  - Desde el 1 de noviembre del 2024: descarga `data_1.csv` y `data_2.csv`.
  En el siguiente [enlace](https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv) puede descargar `data_2.csv`.
5. (1 punto) Cree una tarea que concatene los datasets disponibles mediante la función `load_and_merge()`. Configure un `Trigger` para que la tarea se ejecute si encuentra disponible **como mínimo** uno de los archivos.
6. (1 punto) Aplique el hold out al dataset mediante la función `split_data()`, obteniendo un conjunto de entrenamiento y uno de prueba.
7. (2 puntos) Realice 3 entrenamientos en paralelo:
  - Un modelo Random Forest.
  - 2 modelos a elección.
  Asegúrese de guardar sus modelos entrenados con nombres distintivos. Utilice su función `train_model()` para ello.
8. (2 puntos) Mediante la función `evaluate_models()`, evalúe los modelos entrenados, registrando el accuracy de cada modelo en el set de prueba. Luego debe imprimir el mejor modelo seleccionado y su respectiva métrica. Configure un `Trigger` para que la tarea se ejecute solamente si los 3 modelos fueron entrenados y guardados.
'''
import pandas as pd
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier

from airflow import DAG
#Operadores de airflow, encargados de ejecutar una tarea atómica, una Task es 
#una instancia de un operador
from airflow.operators.empty import EmptyOperator #clase dummy para iniciar o terminar el DAG
from airflow.operators.bash import BashOperator #para ejecutar comandos de consola
from airflow.operators.python import PythonOperator #para ejecutar funciones python
from airflow.operators.python_operator import BranchPythonOperator
from airflow.utils.dates import days_ago

from hiring_dynamic_functions import (
    create_folders,
    download_dataset,
    load_and_merge,
    split_data,
    train_model,
    train_model,
    evaluate_models,
)

args = {
    "owner": "Francisco Maldonado",
    "retries": 1,
    "retry_delay": timedelta(seconds=10)
}

with DAG(
    dag_id="hiring_dynamic",
    default_args=args,
    description="MLops pipeline Lab 9",
    start_date = pd.to_datetime("20241001"),
    schedule="0 15 5 * *",
    catchup=True,
) as dag:


    dummy_task = EmptyOperator(task_id="Iniciando_proceso", retries=2)

    task_create_folders = PythonOperator(
        task_ide="create_folders",
        python_callable=create_folders,
        op_kwargs={"dir_name": "{{ ds }}"},
    )

    task_data_extraction = PythonOperator(
        task_ide="data_extraction",
        python_callable=data_extraction,
        op_kwargs={"dir_name": "{{ ds }}"},
    )


    task_cleaning_and_transformation = PythonOperator(
        task_ide="cleaning_and_transformation",
        python_callable=cleaning_and_transformation,
        op_kwargs={"dir_name": "{{ ds }}"},
    )

    task_prediction_generation = PythonOperator(
        task_ide="prediction_generation",
        python_callable=prediction_generation,
        op_kwargs={"dir_name": "{{ ds }}"},
        trigger_rule="one_success",
    )

    # Función para determinar qué rama se ejecutará
    def choose_branch(ds):
        '''
      - Fechas previas al 1 de noviembre de 2024: Se descarga solo `data_1.csv`
      - Desde el 1 de noviembre del 2024: descarga `data_1.csv` y `data_2.csv`.
        '''
        if pd.to_datetime(ds) < pd.to_datetime("2024-11-01"):
            return "download_dataset1"
        else:
            return ["download_dataset1", "download_dataset2"]

    # Branching task
    branch_task = BranchPythonOperator(
        task_id='branch_task',
        python_callable=choose_branch,
        provide_context=True,
    )

    task_split_data_and_transform = PythonOperator(
        task_ide="split_data_and_transform",
        python_callable=split_data_and_transform,
        op_kwargs={"dir_name": "{{ ds }}"},
    )
            
    task_model_retraining = PythonOperator(
        task_ide="model_retraining",
        python_callable=model_retraining,
        op_kwargs={"dir_name": "{{ ds }}"},
    )



    dummy_task >> task_create_folders >> task_data_extraction >> \
        branch_task >> [
            task_cleaning_and_transformation,
            task_split_data_and_transform >> task_model_retraining >> task_cleaning_and_transformation,
            ] >> task_prediction_generation

