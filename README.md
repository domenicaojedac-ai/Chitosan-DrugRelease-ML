# A Hybrid Computational–Experimental Approach for Predicting Encapsulation Efficiency and pH-Dependent Drug Release in Chitosan-Based Hydrogel Systems

Este repositorio contiene el framework desarrollado para mi Trabajo de Integración Curricular en la **Universidad de Investigación de Tecnología Experimental Yachay**, dentro de la Escuela de Ciencias Biológicas e Ingeniería.

---

## Resumen del Proyecto
Este estudio presenta un enfoque híbrido que combina la recopilación de datos de literatura científica con modelos de **Machine Learning (ML)** y validación experimental para optimizar sistemas de entrega de fármacos. Se desarrollaron modelos capaces de predecir la **Eficiencia de Encapsulación (EE%)** y el perfil de **liberación acumulada de fármacos** dependiente del pH en hidrogeles basados en quitosano.

El sistema fue validado en laboratorio mediante la síntesis de perlas de quitosano cargadas con paracetamol y reticuladas con TPP por gelificación iónica.

## Características Principales
- **Modelado Predictivo:** Uso de algoritmos de regresión supervisada (**XGBoost**, **Random Forest** y **ElasticNet**).
- **Arquitectura Robusta:** Implementación de `RegressorChain` para capturar la dependencia temporal en curvas de liberación.
- **Interfaz Gráfica (GUI):** Aplicación de escritorio intuitiva construida en Python (Tkinter) para simular formulaciones sin necesidad de programar.
- **Optimizador de Formulaciones:** Herramienta integrada para encontrar las condiciones ideales según el objetivo de liberación deseado.

## Estructura del Repositorio
Para garantizar la reproducibilidad, el proyecto se organiza de la siguiente manera:

*   `app/`: Código fuente de la interfaz gráfica y metadatos de la aplicación.
*   `data/`: Datasets procesados y estandarizados extraídos de la literatura científica (39 artículos).
*   `models/`: Modelos pre-entrenados y serializados (.pkl / .joblib) listos para producción.
*   `training/`: Scripts de Python (.py) que contienen el análisis de datos y el proceso de entrenamiento de los modelos.
*   `LICENSE`: Licencia MIT de uso académico.

##  Instalación y Uso
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/domenicaojedac-ai/Chitosan-DrugRelease-ML.git

2. Instalar dependencias necesarias:
    ```bash
    pip install -r requirements.txt

3. Ejecutar la aplicación:
    ```bash
   python app/interfaz.py

*(Nota: Para usuarios que deseen el ejecutable directo, consultar la sección de [Releases](../../releases))*

## Autoría y Créditos
- **Autora:** Domenica Pamela Ojeda Cedeño
- **Asesor:** PhD. Christian Rafael Quijia Quezada
- **Co-Asesor:** MSc. Kevin Raúl Landazuri Cortez

## Licencia
Este proyecto está bajo la Licencia MIT. Para más detalles, ver el archivo [LICENSE](LICENSE).
