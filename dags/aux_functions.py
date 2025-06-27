'''
Indicaciones relevantes de la entrega:

Tanto el archivo de transacciones actualizado en cada iteración, como
el modelo actualizado en cada reentrenammiento serán guardado en el directorio
de trabajo `dags`, pero en producción deberían ser subidos a algún repositorio.
Lo mismo con los archivos de clientes y productos.

Mientras que todos los archivos propios de la iteración serán guardados
en una carpeta con nombre dado por la fecha de ejecución, siguiendo la práctica
usual en el curso.

Si no existe modelo en el directorio de trabajo `dags` el pipeline revertirá
a una iteración con reentrenamiento para generarlo.
'''

import os
import subprocess
from pathlib import Path
import time
import datetime
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pickle import dump, load


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score

from lightgbm import LGBMClassifier
import optuna
import shap

import gradio as gr

from transformation_functions import pre_construir_variables

#Configuraciones varias
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn import set_config
set_config(transform_output="pandas")
RANDOM_STATE=99

def create_folders(dir_name):
    '''
    función que crea una carpeta, la cual utilice la fecha de ejecución como 
    nombre. Adicionalmente, dentro de esta carpeta debe crear las siguientes subcarpetas:
      - raw
      - transformed
      - splits
      - models
      - files/model_tracking/
    '''
    dir_path = Path(dir_name)
    os.makedirs(dir_path, exist_ok=True)
    for subdirectory_name in [ 
        "raw",  "transformed", "preprocessed", "splits", "models",
    ]: 
        os.makedirs(dir_path / subdirectory_name, exist_ok=True)


def data_extraction(dir_name):
    '''
    **Extracción de datos**: Configurar `Airflow` para obtener automáticamente los datos y prepararlos como se realizó en la primera entrega. 
      - Asuman que los nuevos datos cuentan con la misma estructura de `transacciones.parquet`.
      - Para esta parte, pueden asumir que los datos aparecen *mágicamente* en su directorio de trabajo.
      - Pueden realizar supuestos adicionales, pero estos deben quedar evidenciados de forma clara y fácil de distinguir para su posterior revisión.
      
    EL CÓDIGO DEBERÍA EN CADA ITERACIÓN DESCARGAR transacciones.parquet HASTA T, clientes.parquet Y productos.parquet
    DESDE UN REPOSITORIO EXTERNO Y RECIBIR LA DATA DE LA SEMANA T+1, UNIRLO AL transacciones.parquet YA EXISTENTE
    ACTUALIZAR EL REPOSITORIO EXTERNO HASTA T+1 Y TRABAJAR CON ESTO.
    EN LA ITERACIÓN ACTUAL DE LA TAREA, REEMPLAZARÉ EL REPOSITORIO EXTERNO CON EL DIRECTORIO DE TRABAJO, EN EL CUAL 
    SUPONDRÉ QUE EXISTEN ARCHIVOS clientes.parquet, productos.parquet Y transacciones.parquet HASTA T Y
    UN transacciones_nuevo.parquet, CON LA DATA DE LA SEMANA T+1.
    '''
    dir_path = Path(dir_name)
    assert (dir_path).exists(), f"El directorio principal no ha sido creado."

    # Esto debería descargarse de un repositorio
    df_clientes = pd.read_parquet("clientes.parquet")
    df_productos = pd.read_parquet("productos.parquet")
    df_transacciones_viejo = pd.read_parquet("transacciones.parquet")

    #Esto debería descargarse de otro lado
    df_transacciones_nuevo = pd.read_parquet("transacciones_nuevo.parquet")

    df_transacciones = pd.concat([df_transacciones_viejo, df_transacciones_nuevo ])

    df_transacciones.to_csv(dir_path / "raw"/ "transacciones.csv")
    df_clientes.to_csv(dir_path / "raw"/ "clientes.csv")
    df_productos.to_csv(dir_path / "raw"/ "productos.csv")

    #Esto debería subirse al repositorio
    df_transacciones.to_parquet("transacciones.parquet")


def cleaning_and_transformation(dir_name):
    '''
    **Limpieza y transformación**: Estandarizar y preparar los datos para su 
    uso en el modelo. Esta etapa debe ser modular y replicable para nuevos datos.
    '''
    dir_path = Path(dir_name)
    for subdir in ["raw", "transformed"]:
        assert (dir_path/ subdir).exists(), f"El directorio {subdir} no ha sido creado."

    df_clientes = pd.read_csv(dir_path / "raw"/ "clientes.csv").dropna()
    df_transacciones = pd.read_csv(dir_path / "raw"/ "transacciones.csv").drop_duplicates()
    df_productos= pd.read_csv(dir_path / "raw"/ "productos.csv")

    df_cruce = cruzar_frames(df_clientes, df_productos, df_transacciones)
    construir_variables(df_cruce).to_csv(dir_path / "transformed"/ "transformed_data.csv")
    

def drift_detection():
    pass

def split_data_and_transform(dir_name, random_seed=RANDOM_STATE):
    '''
    Función para el reentrenamiento. Se preocupa tando del holdout
    como de aplicar la transformación a los dataset de train y test.
    '''
    
    dir_path = Path(dir_name)
    for subdir in ["raw", "splits"]:
        assert (dir_path/ subdir).exists(), f"El directorio {subdir} no ha sido creado."

    df_clientes = pd.read_csv(dir_path / "raw"/ "clientes.csv").dropna()
    df_transacciones = pd.read_csv(dir_path / "raw"/ "transacciones.csv").drop_duplicates()
    df_productos= pd.read_csv(dir_path / "raw"/ "productos.csv")


    X, y = cruzar_frames_X_y(df_clientes, df_productos, df_transacciones)

    split_data_names = [
        "data_1_X_train.csv",
        "data_1_X_test.csv",
        "data_1_y_train.csv",
        "data_1_y_test.csv",
    ]
    for name, data in zip (
        split_data_names,
        train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_seed)
    ):
        construir_variables(data).to_csv(dir_path / "splits" / name, index=False)

        
def model_retraining(dir_name, random_seed=RANDOM_STATE):
    '''
    - **Reentrenamiento del modelo**: Implementar una rutina de reentrenamiento 
    y trackeo de resultados. Además, esta etapa debe considerar:
        - Si deciden implementar la detección de *drift*, este paso sólo debe 
          ser ejecutado en caso de que exista drift en los datos. En caso 
          contrario, deben desarrollar código para que el reentrenamiento sea 
          ejecutado de forma periódica.
        - Optimización de hiperparámetros.
        - Entrenar modelo con los hiperparámetros seleccionados usando la data 
          nueva.
        - Trackeo de sus resultados, registrando de manera organizada la 
          siguiente información (se recomienda muchísimo usar `MLflow`):
            - Métricas de desempeño
            - Hiperparámetros del modelo
            - Gráficos de interpretabilidad.
        - Exportar el modelo entrenado para su posterior uso (se recomienda 
          usar `MLflow` para esto). 

        Esta lógica puede quedar implementada como tareas condicionales, 
        *placeholders* o módulos configurables, que se activarán más adelante.
    '''

    dir_path = Path(dir_name)
    split_data_names = [
        "data_1_X_train.csv",
        "data_1_X_test.csv",
        "data_1_y_train.csv",
        "data_1_y_test.csv",
    ]
    X_train, X_test, y_train, y_test = [
        pd.read_csv(dir_path / "splits" / name)
        for name in split_data_names
    ]

    var_categoricas = [
            "customer_type",
            "brand",
            "category",
            "sub_category",
            "segment",
            "package",
            "purchased_last_week",
            "purchased_last_month",
            "purchased_ever",
    ]

    var_numericas = [
            "Y",
            "X",
            "num_deliver_per_week",
            "size",
            "weekly_avg_distinct",
            "avg_purchase_period",
    ]


    def objective(trial):
        # Inserte su código acá

        # Defininmos los hiperparámetros a tunear
        lgbm_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }
        ohe_params = {
            "min_frequency": trial.suggest_float("min_frequency", 0.0, 1.0)
        }
        robustsclr_params = {
            "quantile_range": (trial.suggest_int("q_min", 10, 50), trial.suggest_int("q_max", 60, 90))
        }

        # Creamos y entrenamos el pipeline
        column_transformers= ColumnTransformer([
                    ('numerical', RobustScaler(**robustsclr_params), var_numericas),
                    (
                        'categorical',
                        OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist', **ohe_params),
                        var_categoricas
                        ),
                    ], verbose_feature_names_out=False).set_output(transform='pandas')

        trial_pipeline = Pipeline([
            ('column_transformer', column_transformers),
            ("classifier", LGBMClassifier(random_state=random_seed, **lgbm_params))
                              ])
        trial_pipeline.fit(
            X_train, y_train,
        )
        #Guardamos el pipeline en el trial
        trial.set_user_attr("pipeline", trial_pipeline)

        # Obtenemos la predicción de validación
        yhat = trial_pipeline.predict(X_test)

        return f1_score(y_test, yhat)

    study = optuna.create_study( direction="maximize")
    study.optimize(objective, timeout=15*60)

    print("Número de trials:", len(study.trials))
    print("Mejor valor f1_score:", study.best_trial.value)
    print("Mejor hiperparámetros:", study.best_params)

    best_model = study.best_trial.user_attrs["pipeline"]
    model = best_model.fit(pd.concat([X_train, X_test]))


    model_id = datetime.datetime.today().strftime("%H%m%s")
    model_name =  f"pipe_{model_name}_{model_id}.joblib"
    joblib.dump(model, dir_path / "models" / "model.joblib")

    #Esto debería subirse al repositorio
    joblib.dump(model, "model.joblib")

def prediction_generation(data_dir):
    '''
    - **Generación de predicciones**  
      - Utilizar el mejor modelo disponible para predecir, para cada par cliente-producto, si el cliente comprará ese producto la próxima semana.  

    Por indicación del docente, filtrará la combinación de clientes-productos 
    en el que su modelo prediga que se efectuará una compra (y = 1) 

    El mejor modelo será extraído del directorio del repositorio ( provisionalmente directorio de trabajo).

    Finalmente la predicció será guardada en el directorio de trabajo.

    '''
    dir_path = Path(dir_name)
    model_path = "model.joblib"
    assert (dir_path/ "transformed").exists(), f"El directorio {"transformed"} no ha sido creado."
    assert Path(model_path).exists(), f"No hay modelo en el directorio de trabajo."

    data = pd.read_csv(dir_path / "transformed"/ "transformed_data.csv")

    # Esto debería descargarse de un repositorio
    pipeline = joblib.load(model_path)

    y_pred = pipe.predict(data)
    filtered_prediction = data[y_pred == 1][["customer_id", "product_id"]]

    # Esto debería guardarse en un repositorio
    filtered_prediction.to_csv("predicciones.csv", index=False)



if __name__ == "__main__":
    print("We're working for you")

