# Práctica 4: Detección y Seguimiento de Vehículos y Matrículas - Visión por Computador

Repositorio de la Práctica 4 (Parte A - Detección) de la asignatura Visión por Computador (ULPGC). El proyecto, desarrollado mediante el cuaderno interactivo `VC_P4.ipynb`, implementa un sistema automatizado de videovigilancia capaz de detectar y realizar seguimiento (tracking) de personas y vehículos, extraer la región de interés (ROI) de los coches y localizar sus matrículas utilizando modelos personalizados basados en redes neuronales (YOLO / RT-DETR).

## Estructura y Requisitos

El entorno de desarrollo requiere Python (se recomienda versión 3.9.5 por compatibilidad futura con OCR) y las siguientes dependencias estándar y de Deep Learning. Para asegurar el correcto funcionamiento del cuaderno, es necesario crear un entorno e instalar las librerías ejecutando los siguientes comandos:

```bash
conda create --name VC_P4 python=3.9.5
conda activate VC_P4
pip install ultralytics
pip install opencv-python numpy pandas matplotlib
pip install lapx  # Requerido para algoritmos de tracking en Ultralytics
pip install tensorboard faster_coco_eval  # Requerido si se evalúa RT-DETR
```

El proyecto se organiza de la siguiente manera:

- **`VC_P4.ipynb`**: Cuaderno principal que contiene el pipeline de detección general, extracción de matrículas, seguimiento de objetos y generación de reportes (CSV y Vídeo).
- **`runs/`**: Directorio generado automáticamente por YOLO que almacena las gráficas de validación, matrices de confusión y métricas del entrenamiento del modelo de matrículas.
- **`assets/`**: Directorio destinado a almacenar capturas de demostración y resultados visuales para este documento.
- **`datasets/`**: Carpeta (o enlace) con las imágenes y etiquetas `.txt` utilizadas para el entrenamiento del detector de matrículas.
- **`detecciones_matriculas.csv`**: Archivo de salida con el registro frame a frame de los objetos seguidos.

## Desarrollo de las Tareas y Demostrador

El cuaderno aborda la detección integral combinando un modelo preentrenado general y un modelo de entrenamiento propio, cumpliendo con los requisitos de la práctica:


## Instrucciones de Uso

Para evaluar el proyecto correctamente y asegurar la estabilidad del entorno:

1. Clonar el repositorio y asegurar que los modelos `.pt` (`yolo11n.pt` y el modelo entrenado de matrículas) se encuentren en la raíz del proyecto.
2. Iniciar la sesión en Jupyter Notebook y abrir el archivo `VC_P4.ipynb`.
3. Descargar el archivo de test [video](https://drive.google.com/file/d/1jQa2iCTIDOyzp85Me7nk2g7a3YAy2zcL/view?usp=sharing) y añadirlo a la carpeta assets.
4. Ejecutar las celdas de forma secuencial. El procesamiento del vídeo está configurado sin `show=True` por cada frame interno para no colapsar la RAM de Jupyter.
5. El proceso generará automáticamente el vídeo renderizado final y el archivo `detecciones_matriculas.csv` en el directorio de trabajo. Para detener un proceso en vivo (webcam), presionar la tecla `ESC`.

## Declaración de Uso de IA

Conforme a las directrices de la asignatura, se declara el uso de herramientas de Inteligencia Artificial bajo un rol estricto de consultoría técnica. Su aplicación se ha limitado a:


## Contribuidores
| Nombre | GitHub |
| :--- | ---: |
| Ricardo García Rodríguez | [Ricardo428](https://github.com/Ricardo428) |
| Juan Daniel López Melainine | [0nick3r](https://github.com/0nick3r) |