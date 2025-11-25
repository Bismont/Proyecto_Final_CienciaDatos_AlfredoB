from sklearn.decomposition import PCA
import plotly.express as px


EMOTIONS = [
    'anger', 'anticipation', 'disgust', 'fear', 'joy',
    'negative', 'positive', 'sadness', 'surprise', 'trust'
]


def run_emotion_pca(df):
    """
    Corre PCA sobre las 10 emociones NRC para reducir a 2 componentes.
    Agrega al DataFrame columnas:
      - pca_comp1
      - pca_comp2
    También regresa el modelo PCA por si quieres inspeccionar loadings.
    """

    # Matriz de 10 emociones
    X = df[[f"e_{emo}" for emo in EMOTIONS]].values

    # PCA a 2 componentes para visualización
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)

    df["pca_comp1"] = pcs[:, 0]
    df["pca_comp2"] = pcs[:, 1]

    print("\nVarianza explicada por PC1 y PC2:")
    print(pca.explained_variance_ratio_)

    return df, pca

def plot_pca_emotions(df):
    fig = px.scatter(
        df,
        x="pca_comp1",
        y="pca_comp2",
        color="domain",
        opacity=0.35,
        hover_data=["title_norm", "date", "valence", "arousal"],
        title="PCA emocional de titulares (10 emociones NRC)"
    )
    fig.update_layout(width=900, height=600)
    fig.show()
