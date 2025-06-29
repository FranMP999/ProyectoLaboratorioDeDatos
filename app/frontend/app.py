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

with gr.Blocks(theme = gr.themes.Base()) as demo:
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

    with gr.Row():
        with gr.Column():
            inputs = {}
            # Slider para las numéricas, checkbox para las binarias, dropdown o radio para las categóricas
            for variable in var_categoricas:
                inputs[variable] = gr.Dropdown(
                    input_values[variable], label=variable, info="Seleccione según corresponda"
                )

            for variable in var_booleanas:
                inputs[variable] = gr.Checkbox(
                    label="Morning", info="Seleccione según corresponda"
                )
            for variable in var_enteras:
                values = np.arra(input_values[variable])
                min, max, mean = values.min(), values.max(), values.mean()
                inputs[variable] = gr.Slider(
                    label = variable, step=1,
                    minimum = min, maximum = max, value = mean,
                )
            for variable in var_float:
                values = np.arra(input_values[variable])
                min, max, mean = values.min(), values.max(), values.mean()
                inputs[variable] = gr.Slider(
                    label = variable,
                    minimum = min, maximum = max, value = mean,
                )

        with gr.Column():
            label = gr.DataFrame(label = 'Predicción de Compra') 
    
    with gr.Row():
        button = gr.Button(value = 'Predecir!')

    # setear interactividad
    outputs = [label]

    # obtener predicción desde el backend
    button.click(fn = get_backend_prediction, inputs = input, outputs = outputs) # esta linea invoca el backend

    demo.launch(server_name="0.0.0.0", server_port=7860)
