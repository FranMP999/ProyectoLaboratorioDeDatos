'''
En esta etapa, el equipo debe diseñar e implementar una canalización automatizada utilizando `Apache Airflow` para orquestar de forma eficiente todo el flujo de trabajo de su sistema predictivo, integrando herramientas como `MLflow`, `Optuna` y `SHAP`. Esta canalización debe contemplar el ciclo completo de modelado: desde la extracción de datos hasta la generación de predicciones y el seguimiento de resultados, de manera robusta y flexible.

Aunque en esta entrega se trabajará únicamente con los datos iniciales proporcionados, es **fundamental** que el *pipeline* sea diseñado con visión de producción, tal como se esperaría en un entorno empresarial real. Por tanto, el *pipeline* debe estar preparado para detectar *drift* en los datos y, de ser necesario, reentrenar el modelo de forma automatizada ante la llegada de nuevos datos.

**Nota importante: Tómense un tiempo para leer detenidamente esto para entender lo que esto significa. Para entrega final del proyecto se les pedirá que su pipeline reciba un nuevo conjunto de datos (por ejemplo, ustedes cuentan con datos (t) y reciben una nueva semana de datos (t+1)) y genere la predicción para la semana siguiente (t+2). <u>Repetiremos este ejercicio varias veces</u>. Tome esto en consideración para el desarrollo de esta sección.**

**BONUS**: Si deciden utilizar `MLflow`, se les otorgará entre 0 y 0.5 puntos de puntaje adicional dependiendo su uso y en qué partes lo utilizan.

Su *pipeline* debe considerar los siguientes puntos:

- **Extracción de datos**: Configurar `Airflow` para obtener automáticamente los datos y prepararlos como se realizó en la primera entrega. 
  - Asuman que los nuevos datos cuentan con la misma estructura de `transacciones.parquet`.
  - Para esta parte, pueden asumir que los datos aparecen *mágicamente* en su directorio de trabajo.
  - Pueden realizar supuestos adicionales, pero estos deben quedar evidenciados de forma clara y fácil de distinguir para su posterior revisión.
- **Limpieza y transformación**: Estandarizar y preparar los datos para su uso en el modelo. Esta etapa debe ser modular y replicable para nuevos datos.
- **Detección de drift en los datos [BONUS: 0.3 puntos]**: Desarrollar un sistema o modelo que sea capaz de distinguir cuando exista *drift* en los datos.
- **Reentrenamiento del modelo**: Implementar una rutina de reentrenamiento y trackeo de resultados. Además, esta etaba debe considerar:
    - Si deciden implementar la detección de *drift*, este paso sólo debe ser ejecutado en caso de que exista drift en los datos. En caso contrario, deben desarrollar código para que el reentrenamiento sea ejecutado de forma periódica.
    - Optimización de hiperparámetros.
    - Entrenar modelo con los hiperparámetros seleccionados usando la data nueva.
    - Trackeo de sus resultados, registrando de manera organizada la siguiente información (se recomienda muchísimo usar `MLflow`):
      - Métricas de desempeño
      - Hiperparámetros del modelo
      - Gráficos de interpretabilidad.
    - Exportar el modelo entrenado para su posterior uso (se recomienda usar `MLflow` para esto). 

    Esta lógica puede quedar implementada como tareas condicionales, *placeholders* o módulos configurables, que se activarán más adelante.
  
- **Generación de predicciones**  
  - Utilizar el mejor modelo disponible para predecir, para cada par cliente-producto, si el cliente comprará ese producto la próxima semana.  
  - **Salida**: el *pipeline* debe generar un **archivo `.csv`** que contenga **solo** las combinaciones *cliente-producto* con predicción positiva (es decir, los productos que se espera que el cliente compre). NO deben incluirse los pares con predicción negativa.


**IMPORTANTE**: Como "próxima semana" deben usar la **semana siguiente a la semana más reciente presente en los datos**. Por ejemplo, si sus datos contienen hasta la última semana del 2024, se le pide generar predicciones para la primera semana del 2025.
'''
from pathlib import Path
from datetime import timedelta
import pandas as pd

from airflow import DAG
#Operadores de airflow, encargados de ejecutar una tarea atómica, una Task es 
#una instancia de un operador
from airflow.operators.empty import EmptyOperator #clase dummy para iniciar o terminar el DAG
from airflow.operators.bash import BashOperator #para ejecutar comandos de consola
from airflow.operators.python import PythonOperator #para ejecutar funciones python
from airflow.operators.python_operator import BranchPythonOperator
from airflow.utils.dates import days_ago

from functions import (
    create_folders,
    data_extraction,
    cleaning_and_transformation,
    split_data_and_transform,
    model_retraining,
    prediction_generation,
    choose_branch,
)

args = {
    "owner": "Francisco Maldonado",
    "retries": 1,
    "retry_delay": timedelta(seconds=10)
}

with DAG(
    dag_id="pipeline_entrega2proyecto",
    default_args=args,
    description="MLops pipeline Entrega 2 Proyecto",
    start_date = pd.to_datetime("20240101"),
    schedule="0 0 * * 0",
    catchup=False
) as dag:


    dummy_task = EmptyOperator(task_id="Iniciando_proceso", retries=2)

    task_create_folders = PythonOperator(
        task_id="create_folders",
        python_callable=create_folders,
        op_kwargs={"dir_name": "{{ ds }}"},
    )

    task_data_extraction = PythonOperator(
        task_id="data_extraction",
        python_callable=data_extraction,
        op_kwargs={"dir_name": "{{ ds }}"},
    )


    task_cleaning_and_transformation = PythonOperator(
        task_id="cleaning_and_transformation",
        python_callable=cleaning_and_transformation,
        op_kwargs={"dir_name": "{{ ds }}"},
        trigger_rule="one_success",
    )

    task_prediction_generation = PythonOperator(
        task_id="prediction_generation",
        python_callable=prediction_generation,
        op_kwargs={"dir_name": "{{ ds }}"},
    )

    # Branching task
    branch_task = BranchPythonOperator(
        task_id='branch_task',
        python_callable=choose_branch,
        provide_context=True,
    )

    task_split_data_and_transform = PythonOperator(
        task_id="split_data_and_transform",
        python_callable=split_data_and_transform,
        op_kwargs={"dir_name": "{{ ds }}"},
    )
            
    task_model_retraining = PythonOperator(
        task_id="model_retraining",
        python_callable=model_retraining,
        op_kwargs={"dir_name": "{{ ds }}"},
    )



    dummy_task >> task_create_folders >> task_data_extraction >> branch_task 
    branch_task >> [task_cleaning_and_transformation, task_split_data_and_transform ]
    task_split_data_and_transform >> task_model_retraining >> task_cleaning_and_transformation
    task_cleaning_and_transformation >> task_prediction_generation
