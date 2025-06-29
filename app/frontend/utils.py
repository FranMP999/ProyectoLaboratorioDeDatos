import requests
import os

# referencia a localhost si se ejecuta localmente, usa la variable de ambiente BACKEND_URL si se ejecuta en un contenedor
url_base = os.getenv("BACKEND_URL", "http://localhost:8000") # url del backend

# primero definimos una función para obtener respuestas del back a través de requests
def get_backend_prediction(
    customer_type: str,
    brand: str,
    category: str,
    sub_category: str,
    segment: str,
    package: str,
    purchased_last_week: bool,
    purchased_last_month: bool,
    purchased_ever: bool,
    num_deliver_per_week: int,
    avg_purchase_period: int,
    X: float,
    Y: float,
    size: float,
    weekly_avg_distinct: float,
):


    # payload
    data = {
        "customer_type": str(customer_type),
        "brand": str(brand),
        "category": str(category),
        "sub_category": str(sub_category),
        "segment": str(segment),
        "package": str(package),
        "purchased_last_week": bool(purchased_last_week),
        "purchased_last_month": bool(purchased_last_month),
        "purchased_ever": bool(purchased_ever),
        "num_deliver_per_week": int(num_deliver_per_week),
        "avg_purchase_period": int(avg_purchase_period),
        "X": float(X),
        "Y": float(Y),
        "size": float(size),
        "weekly_avg_distinct": float(weekly_avg_distinct),
        }
    url = url_base + "/predict" # url del back end
    response = requests.post(url, json = data) # código de respuesta
    label = response.json().get("label") # obtener contenido de la respuesta

    return label
