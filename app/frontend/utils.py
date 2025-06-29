import requests
import os

# referencia a localhost si se ejecuta localmente, usa la variable de ambiente BACKEND_URL si se ejecuta en un contenedor
url_base = os.getenv("BACKEND_URL", "http://localhost:8000") # url del backend

# primero definimos una función para obtener respuestas del back a través de requests
def get_backend_prediction(input_dic):

    # payload
    data = input_dic
    url = url_base + "/predict" # url del back end
    response = requests.post(url, json = data) # código de respuesta
    label = response.json().get("label") # obtener contenido de la respuesta

    return label
