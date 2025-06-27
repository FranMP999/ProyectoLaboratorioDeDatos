'''
Se quiere crear variables que capturen 
 **preferencias históricas de consumo** inferidas por patrones de compra anteriores.
Se crearan variables binarias que indiquen si es que el producto fue comprado la última semana,
el último mes y cuántas veces fue comprado históricamente.
 En estas variables los 
nulos serán imputados como ceros, pues corresponden a ausencia de compras.

Dado que sólo nos interesa clasificar si es que hubo o no compra, ignoraremos
el id de la orden y el tamaño del bulto (`items`). Además order_id podría causar problemas
de información cruzada entre distintos productos.

Se añadirán además como variables la cantidad de productos distintos que 
compra semanalmente cada cliente, y el periodo de compra promedio del producto.
En el primer caso los nulos se imputarán como 0, pues son clientes sin registro de compra,
en el segundo caso se imputarán como el periodo total del dataset, pues si no tienen registro
de compra suficiente para calcular un promedio se puede modelar como un dato
con periodo muy alto, y el valor máximo del periodo total nos permite mantenernos en el rango natural
de los datos.
'''

from datetime import timedelta
import numpy as np
import pandas as pd


RANDOM_STATE=99

def avg_period(lista_fechas):
    '''
    Calcula el periodo promedio dada una lista de fechas
    '''
    return (lista_fechas - np.roll(lista_fechas,1))[1:].mean()

def weekly_avg_distinct_df(df_in):
    '''
    Función que dado un dataframe de variables, entrega una columna
    de cuantos productos distintos compra semanalmente cada cliente
    '''
    return (
        df_in[["customer_id","product_id"]].merge(
            df_in[["customer_id", "product_id", "purchase_date"]].explode("purchase_date")
            .groupby(['customer_id', pd.Grouper(key='purchase_date', freq='W-MON')]).nunique()
            .reset_index()
            .groupby(["customer_id"])["product_id"].mean()
            .to_frame().reset_index().rename(columns={"product_id": "weekly_avg_distinct"}),
        how="left", on="customer_id").fillna(0)
    )


def avg_purchase_period_df(df_in): 
    ''' 
    Función que dado un dataframe de variables, entrega una columna con 
    el periodo de recompra promedio para cada SKU.
    ''' 
    max_period = (
        df_in.purchase_date.dropna().explode.max()
        - df_in.purchase_date.dropna().explode.min()
        ).days
    return (
        df_in[["customer_id","product_id"]].merge(
            df_in[["product_id", "purchase_date"]].dropna().explode("purchase_date")
                .drop_duplicates()
                .sort_values(by="purchase_date")
                .groupby("product_id").agg(avg_period)
                .map(lambda x: x.days)
                .reset_index().rename(columns={"purchase_date": "avg_purchase_period"}),
        how="left", on="product_id")
        .fillna( max_period)
    )

def construir_variables(df_in):
    ''' 
    Función que realiza la construcción de variables sobre el dataframe de variables.
    ''' 
    ultima_fecha = df_in.purchase_date.dropna().explode().max()
    
    return df_in.assign(
        purchased_last_week=(
            df_in.purchase_date
            .map(lambda x: x.max() >= (ultima_fecha - timedelta(weeks=1)), na_action="ignore")
            .fillna(0)
            .astype(int)
        ),
        purchased_last_month=(
            df_in.purchase_date
            .map(lambda x: x.max() >= (ultima_fecha - timedelta(days=30)), na_action="ignore")
            .fillna(0)
            .astype(int)
        ),
        purchased_ever=(~df_in.purchase_date.isna()).astype(int),
    ).merge(
        weekly_avg_distinct_df(df_in), how="left", on=["customer_id","product_id"]
    ).merge(
        avg_purchase_period_df(df_in), how="left", on=["customer_id","product_id"]
    )

def cruzar_frames(df_clientes, df_productos, df_transacciones):
    return (
        df_clientes
        .drop(columns= [
            "region_id",
            "zone_id",
            "num_visit_per_week"
        ])
        .merge(
            df_productos,
            how="cross")
        .merge(
            df_transacciones
            .groupby(["customer_id", "product_id"]).agg(list).map(lambda x: np.array(x)),
            how="left", on=["customer_id", "product_id"])
    )

def cruzar_frames_X_y(df_clientes, df_productos, df_transacciones):
    X = (
        df_clientes
        .drop(columns= [
            "region_id",
            "zone_id",
            "num_visit_per_week"
        ])
        .merge(
            df_productos,
            how="cross")
        .merge(
            df_transacciones
            [df.purchase_date <= (df.purchase_date.max() - timedelta(weeks=1))] # RESTRICCIÓN TEMPORAL DE TRAINING
            .groupby(["customer_id", "product_id"]).agg(list).map(lambda x: np.array(x)),
            how="left", on=["customer_id", "product_id"])
    )

    y = (
        df_clientes
        .drop(columns= [
            "region_id",
            "zone_id",
            "num_visit_per_week"
        ])
        .merge(
            df_productos,
            how="cross")
        .merge(
            df_transacciones,
            how="left", on=["customer_id", "product_id"])
        .query(f"purchase_date > '{(df.purchase_date.max() - timedelta(weeks=1))}'") # RESTRICCIÓN TEMPORAL DE TESTING
        .assign(label=1)
        [["customer_id", "product_id", "label"]]
        .merge(
            df_real[["customer_id", "product_id"]],
            how="right", on=["customer_id", "product_id"])
        .drop_duplicates()
        .fillna(0)
    )
    return X, y
