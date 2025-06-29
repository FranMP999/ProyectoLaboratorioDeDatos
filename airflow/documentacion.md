# Documentación Pipeline Airflow

Comando docker para ejecución
```
docker run -p 8080:8080 -v "./resultados:/root/airflow/resultados" entrega2-pipeline
```

## Indicaciones relevantes de la entrega

* Tanto el archivo de transacciones actualizado en cada iteración, como
el modelo actualizado en cada reentrenamiento serán guardado en el directorio
local `repositorio`, pero en producción deberían ser subidos a algún repositorio.
Lo mismo con los archivos de clientes y productos.
Los datos nuevos serán importados del directorio local `datos_nuevos`, simulando
una fuente cualquiera de la que debieran ser obtenidos.

* Todos los archivos propios de la ejecución del pipeline serán guardados
en una carpeta con nombre dado por la fecha de ejecución, siguiendo la práctica
usual en el curso.

* Las predicciones serán guardadas en el directorio de trabajo.
Por indicación del docente, en este archivo resultante se filtrará la 
combinación de clientes-productos en el que su modelo prediga que se efectuará una compra (y = 1) 

* Si no existe modelo en el repositorio externo, el pipeline revertirá
a una iteración con reentrenamiento para generarlo.
El trackeo de resultados se almacena en el directorio `train_logs`
que debe ser montado en un volumen de docker.

## Explicación del DAG
  * Una descripción clara del `DAG`, explicando la funcionalidad de cada tarea y cómo se relacionan entre sí.
  * Diagrama de flujo del *pipeline* completo.
  * Una representación visual del `DAG` en la interfaz de `Airflow`.
  * Explicación de cómo se diseñó la lógica para integrar futuros datos, detectar *drift* y reentrenar el modelo.
  * Todos estos puntos deben estar contenidos en un archivo markdown junto a la carpeta de esta sección.
