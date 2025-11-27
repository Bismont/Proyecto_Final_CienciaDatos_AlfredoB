Los inputs que ocupan este proyecto pueden obtenerse en

https://drive.google.com/drive/folders/13CceLMygg2dvC1p8NlnSsdqid2EKAjaO?usp=sharing

# Análisis emocional de titulares periodísticos (México 2024–2025)

Este repositorio contiene el código, datos procesados y resultados del proyecto:
**“Análisis emocional de titulares de medios nacionales mediante lexicones, PCA y un modelo LSTM con embeddings de dominio”.**

El objetivo es caracterizar la carga emocional de titulares y estudiar diferencias editoriales entre medios usando métodos combinados de NLP clásico y redes neuronales.

## Estructura del repositorio

```
get_data/                  # Scraper de titulares vía API
input_data/                # Datos crudos, limpios, balanceados y con PCA
plots/                     # Gráficos generados (PCA, betas, matrices, etc.)
project_library/           # Módulos: limpieza, PCA, modelo LSTM
dashboard_noticias.py      # Dashboard interactivo (Dash + Plotly)
proyecto_final.ipynb       # Notebook principal del análisis
sentinel_mx_lstm.pth       # Modelo entrenado (PyTorch)
```

##  1. Obtención de datos
`get_data/news_data.py` implementa un scraper para recolectar titulares por fecha y dominio.
Incluye normalización básica y guardado incremental en CSV.

## 2. Preprocesamiento + emociones NRC
`project_library/data_handling.py` realiza:

- limpieza y normalización del texto  
- cálculo de emociones NRC (anger, joy, trust, fear…)  
- valence y arousal promedio  
- reglas de decisión para etiquetas: **negativo, neutro, positivo**

El dataset final incluye PCA de emociones y clases supervisadas.

## 3. Modelo LSTM con embedding de dominio
`project_library/neural_network.py`:

- Bi-LSTM → vector oculto final  
- Embedding del medio → vector de 3 dimensiones (uno por clase)  
- Predicción mediante:  

```
logits = W h_text + β_d_i
```

Donde βᵢ cuantifica el efecto estilístico del medio en la clasificación.

Entrenamiento:

- Adam  
- Cross-entropy  
- 12 epochs  
- Gradient clipping  
- CUDA opcional  

Modelo final guardado en `sentinel_mx_lstm.pth`.

## 4. PCA de emociones y PCA de estados ocultos
`project_library/pca.py` aplica:

- PCA a vectores emocionales NRC  
- PCA a los estados ocultos del LSTM (representación semántica final)

## 5. Dashboard interactivo
`dashboard_noticias.py` incluye:

- resumen mensual por medio  
- valence & arousal  
- mapa PCA interactivo  
- heatmap NRC  
- wordcloud por mes y por sentimiento  

Ejecutar:

```
python dashboard_noticias.py
```

## 6. Resultados principales (en `/plots`)
- PCA emocional (lexicográfico)  
- PCA de estados ocultos del LSTM  
- Matriz de confusión  
- Curvas de entrenamiento  
- Heatmap de betas  

## Instalación rápida

```
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Notas
Este repo combina NLP clásico y moderno para analizar narrativas emocionales en medios mexicanos.  
Incluye herramientas interpretativas y visuales, así como un modelo entrenado listo para reproducir resultados.
