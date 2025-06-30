# Conclusiones

En esta sección deben reflexionar sobre los aprendizajes, desafíos y oportunidades del enfoque `MLOps` aplicado. Sus reflexiones deben ser escritas en un archivo `conclusiones.md`

Algunas ideas que pueden abordar:
- ¿Cómo mejoró el desarrollo del proyecto al utilizar herramientas de *tracking* y despliegue?
- ¿Qué aspectos del despliegue con `Gradio/FastAPI` fueron más desafiantes o interesantes?
- ¿Cómo aporta `Airflow` a la robustez y escalabilidad del pipeline?
- ¿Qué se podría mejorar en una versión futura del flujo? ¿Qué partes automatizarían más, qué monitorearían o qué métricas agregarían?

## Comentarios Generales
### Dificultades
En el desarrollo de la parte entregada del proyecto se identifica el desafío 
que supone la implementación de un despliegue con miras a la facilidad de mantención
y escalabilidad del modelo. Se enfrentan diversas dificultades relativas a la implementación
de un proyecto con interfaces de alto nivel en comunicación mutua, **Airflow**, **mlfow**,
**Docker**, **FastAPI**, **Gradio** son herramientas cuyo manejo requiere un conocimiento
y adaptabilidad mayor que un pipeline escrito sólo con librerías clásicas de machine-learning
en python, no sólo por tener interfaces y sintaxis propias, sino por ser librerías
en constante actualización, con comandos, objetos y métodos que periodicamente
quedan desfasados, y el versionamiento siempre es una dificultad logística al trabajar con python.

**mlflow** en particular demostró en la implementación  ser una librería que sólo
llegó a dificultar el desarrollo y oscurecer el código, empeorando la experiencia
respecto a hacer un tracking de resultados guardando directamente archivos de la forma usual.

### Beneficios
No obstante la razón por la cual estas herramientas son valiosas en el toolkit
de un cientista de datos queda de manifiesto una vez desarrollado el pipeline.
Si bien aumentan la dificultad de implementación en un principio, permiten
la automatización ordenada y eficiente del proceso de monitorio, así como entregar
una interfaz amigable y directa con la cual interactuar con las predicciones para el
usuario final.
## Comentarios específicos
### Gradio/FastAPI
El desarrollo de la app para el proyecto resultó sorprendentemente intuitiva
y sencilla, teniendo en cuenta que no se disponía de experiencia previa alguna
en el desarrollo web. Tanto Gradio como FastAPI demuestran ser herramientas
directas, desarrolladas con el objetivo de facilitar el despliegue de proyectos
similares al tratado y cumplen con esto.

En la experiencia personal lo más interesante
de estas herramientas es haber constituido un primer acercamiento (muy básico)
al desarrollo web y al funcionamiento de los request htttp.

### Airflow
Airflow no es una herramienta cómoda de implementar
por tener una sintaxis verbosa, en la que cada función utilizada en el proceso
requiere que se le asocie un task mediante varias líneas de código, resultando
en un archivo de código extenso redundante frente a los módulos de python
que ya se escriben para desarrollar el pipeline.  Y es que finalidad de todas las tareas
definidas es poder especificar un dag que se escribe en un par de líneas al final del archivo,
por lo que todo el resto del archivo de dag se experimenta en el desarrollo como la repetición de definiciones de funciones python ya escritas.

Sin embargo el resultado es de suma utilidad para el monitoreo del pipeline en producción, permmitiendo dejar de antemano todas las rutinas programadas.

## Por mejorar
Principlamente queda pendiente el flujo real de los datos en producción, pensando en un repositorio de datos externo en algún servicio de nube 
o servidores propios de la empresa, en los que sean almacenadas las predicciones acumuladas con fines de monitoreo y también el 
acumulado de los datos transaccionales. Así como la interacción con la entidad encarda de generar los datos mes a mes.

Además de esto queda pendiente la conexión del pipeline de producción con la app, ojalá mediante docker compose,
de manera que la app siempre tenga acceso a la versión de modelo más actualizada.
