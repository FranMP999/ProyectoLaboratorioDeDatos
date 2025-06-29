import numpy as np
import gradio as gr
import json
from utils import get_backend_prediction

#Documento con los valores input posibles en base al dataset de entrega1
with open('input_values.json') as json_file:
    input_values = json.load(json_file)


#Clasificación de variables por tipos
var_categoricas = [
    "customer_type",
    "brand",
    "category",
    "sub_category",
    "segment",
    "package",
]
var_booleanas = [
    "purchased_last_week",
    "purchased_last_month",
    "purchased_ever",
]
var_enteras = [
    "num_deliver_per_week",
    "avg_purchase_period",
]
var_float = [
    "X",
    "Y",
    "size",
    "weekly_avg_distinct",
]

with gr.Blocks(theme = gr.themes.Default()) as demo:
    gr.Markdown(
    """
    # Interfaz de Usuario Para Entrega 2 de Proyecto
    Esta herramienta esta diseñada para predecir las compras semanales para la empresa **SodAI Drinks 🥤**.
    ## Cómo usar esta interfaz?
    Usar esta herramienta es fácil! Sólo debes seguir los siguientes pasos:
    1. Fijar los vavlores de variables asociadas al historial de transacciones para el cliente y producto que se quieran predecir.
    2. Obtener la predicción de si en la semana siguiente habrá una compra del producto asociada al cliente.
    
    Eso es todo! 
    """)

    inputs = []
    with gr.Row():
        label = gr.Text(label = 'Predicción de Compra') 
        button = gr.Button(value = 'Predecir!')
    with gr.Row():
        with gr.Column():
                # Slider para las numéricas, checkbox para las binarias, dropdown o radio para las categóricas
            for variable in var_categoricas:
                with gr.Row():
                    if variable == "brand":
                        inputs.append(gr.Dropdown(
                            input_values[variable], label=variable, info="Seleccione según corresponda",
                        ))
                    else:
                        inputs.append(gr.Radio(
                            input_values[variable], label=variable, info="Seleccione según corresponda",
                            value=input_values[variable][0]
                        ))

        with gr.Column():
            with gr.Row():
                for variable in var_booleanas:
                    inputs.append( gr.Checkbox(
                        label=variable, info="Seleccione según corresponda",
                        value=True,
                    ))
            with gr.Row():
                for variable in var_enteras:
                    values = np.array(input_values[variable])
                    min, max, mean = values.min(), values.max(), values.mean()
                    inputs.append(gr.Slider(
                        label = variable, step=1,
                        minimum = min, maximum = max, value = min,
                    ))
                for variable in var_float:
                    values = np.array(input_values[variable])
                    min, max, mean = values.min(), values.max(), values.mean()
                    inputs.append( gr.Slider(
                        label = variable,
                        minimum = min, maximum = max, value = mean,
                    ))
    

    # setear interactividad
    outputs = [label]

    # obtener predicción desde el backend
    button.click(fn = get_backend_prediction, inputs = inputs, outputs = outputs) # esta linea invoca el backend

    demo.launch(server_name="0.0.0.0", server_port=7860)
