from collections import Counter
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import torch


def load_nrc_lexicon(filepath: str) -> dict:
    """
    Carga un NRC Lexicon en formato 'wide', donde cada emoción está en una columna,
    con valores 0/1, y la palabra en español está en la columna 'Spanish Word'.

    Regresa un diccionario:
        palabra -> set({emociones})
    """
    df = pd.read_csv(filepath, sep="\t")

    # columnas actuales que representan emociones en tu archivo:
    emotion_columns = [
        "anger", "anticipation", "disgust", "fear",
        "joy", "negative", "positive", "sadness",
        "surprise", "trust"
    ]

    lex = {}

    for _, row in df.iterrows():
        word = str(row["Spanish Word"]).strip().lower()
        emos = set()

        for emo in emotion_columns:
            try:
                if int(row[emo]) == 1:
                    emos.add(emo)
            except:
                pass  # por si hay NaNs

        if len(emos) > 0:
            lex[word] = emos

    return lex


# ======================================
# Funciones de emociones NRC
# ======================================

# Lista de emociones base de NRC (puedes ajustar)
EMOTIONS = [
    'anger', 'anticipation', 'disgust', 'fear', 'joy',
    'negative', 'positive', 'sadness', 'surprise', 'trust'
]

def tokenize(text: str):
    """Tokenización simple: sólo letras (incluye tildes)."""
    return re.findall(r"[a-zA-ZáéíóúñüÁÉÍÓÚÑÜ]+", str(text).lower())

def emotion_counts(text: str, lexicon: dict) -> dict:
    """
    Cuenta las emociones NRC en un texto, usando el diccionario.
    Regresa un dict emoción -> conteo.
    """
    words = tokenize(text)
    counts = {e: 0 for e in EMOTIONS}
    for w in words:
        if w in lexicon:
            for emo in lexicon[w]:
                if emo in counts:
                    counts[emo] += 1
    return counts

def emotion_vector(text: str, lexicon: dict) -> np.ndarray:
    """Convierte texto en vector de longitud len(EMOTIONS) con conteos."""
    counts = emotion_counts(text, lexicon)
    return np.array([counts[e] for e in EMOTIONS], dtype=float)

def valence_arousal_from_counts(counts: dict) -> tuple[float, float]:

    anger = counts["anger"]
    anticipation = counts["anticipation"]
    disgust = counts["disgust"]
    fear = counts["fear"]
    joy = counts["joy"]
    negative = counts["negative"]
    positive = counts["positive"]
    sadness = counts["sadness"]
    surprise = counts["surprise"]
    trust = counts["trust"]

    # ------------------------
    # VALENCE = lo positivo - lo negativo
    # ------------------------
    valence = (
        joy 
        + trust 
        + anticipation 
        + positive
        - (anger + fear + sadness + disgust + negative)
    )

    # ------------------------
    # AROUSAL = emociones de activación
    # ------------------------
    arousal = (
        anger 
        + fear 
        + surprise 
        + anticipation
    )

    return float(valence), float(arousal)


def valence_arousal_from_text(text: str, lexicon: dict) -> tuple[float, float]:
    """Atajo directo texto -> (valence, arousal)."""
    counts = emotion_counts(text, lexicon)
    return valence_arousal_from_counts(counts)

SPANISH_STOPWORDS = {
    "de","la","que","el","en","y","a","los","del","se","las","por","un","para","con",
    "una","su","al","lo","como","más","pero","sus","le","ya","o","este","sí","porque",
    "esta","entre","cuando","muy","sin","sobre","también","me","hasta","hay","donde",
    "han","quien","ser","son","dos","fue","era","eso","esa","ese","uno","desde","nos",
    "nosotros","ustedes","ellos","ellas","él","ella","tiene","tener","puede","pueden",
    "tras","hoy","ayer","mañana","así","además","nuevo","nueva","nuevo","nuevos",
    "mexico","méxico","mx"
}

def fast_clean_dataset(df):
    df = df.copy()

    # Normalizar títulos (minúsculas, quitar adornos)
    def normalize_title(t):
        t = str(t).lower().strip()
        t = re.sub(r"\s+\|\s+.*$", "", t)
        t = re.sub(r"\s+–\s+.*$", "", t)
        t = re.sub(r"\s+-\s+.*$", "", t)
        t = re.sub(r"\s+", " ", t)
        return t

    df["title_norm"] = df["title"].apply(normalize_title)

    # Quitar duplicados
    df = df.drop_duplicates(subset="title_norm", keep="first")

    # Eliminar registro con título vacío
    df = df[df["title_norm"].str.len() > 3]

    # Tokenización y eliminación de stopwords
    def remove_stopwords(t):
        tokens = tokenize(t)
        tokens_clean = [w for w in tokens if w not in SPANISH_STOPWORDS]
        return " ".join(tokens_clean)

    df["title_clean"] = df["title_norm"].apply(remove_stopwords)

    # Limpieza final (por si el título quedó vacío)
    df = df[df["title_clean"].str.len() > 2]

    df = df.reset_index(drop=True)
    return df

def compute_emotion_features(df, lexicon):
    """
    Agrega al DataFrame columnas:
      - e_anger, e_anticipation, ..., e_trust
      - valence
      - arousal
    """
    emotion_vectors = []
    valences = []
    arousals = []

    for title in tqdm(df["title_norm"], desc="Calculando emociones"):
        # Vector de 10 emociones
        vec = emotion_vector(title, lexicon)
        emotion_vectors.append(vec)

        # valence & arousal
        counts = {e: vec[i] for i, e in enumerate(EMOTIONS)}
        val, aro = valence_arousal_from_counts(counts)
        valences.append(val)
        arousals.append(aro)

    # Convertir vectores a columnas individuales
    emo_arr = np.vstack(emotion_vectors)
    for i, emo in enumerate(EMOTIONS):
        df[f"e_{emo}"] = emo_arr[:, i]

    df["valence"] = valences
    df["arousal"] = arousals

    return df

def balancear_fuentes(df, n_minimo=300):
    """
    Toma el mismo número de artículos por dominio.
    n_minimo = mínimo por fuente (puede ajustarse).
    """
    df_bal = (
        df.groupby("domain")
          .apply(lambda x: x.sample(min(len(x), n_minimo), random_state=42))
          .reset_index(drop=True)
    )
    return df_bal

def valence_to_label(v, t_pos=0, t_neg=0):
    if v > t_pos:
        return 2  # positivo
    elif v < t_neg:
        return 0  # negativo
    else:
        return 1  # neutro
    
from collections import Counter

def build_vocab_from_titles(df, max_size=15000, min_freq=2):
    counter = Counter()

    for t in df["title_norm"]:
        toks = tokenize(str(t))
        counter.update(toks)

    words = [(w, c) for w, c in counter.items() if c >= min_freq]
    words = sorted(words, key=lambda x: x[1], reverse=True)

    words = words[:max_size]

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for idx, (w, _) in enumerate(words, start=2):
        vocab[w] = idx

    return vocab



def encode(text, vocab, max_len=40):
    toks = tokenize(text)
    ids = [vocab.get(t, vocab["<UNK>"]) for t in toks[:max_len]]
    
    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    
    return ids, min(len(toks), max_len)

def build_domain2id(df):
    domains = sorted(df["domain"].unique())
    return {d: i for i, d in enumerate(domains)}

def preencode_dataframe(df, vocab, domain2id, max_len=40):
    X, L, D, Y = [], [], [], []

    for t, dom, val in zip(
        df["title_norm"], df["domain"], df["valence"]
    ):
        ids, length = encode(t, vocab, max_len)
        X.append(ids)
        L.append(length)
        D.append(domain2id[dom])
        Y.append(valence_to_label(val, t_pos=1, t_neg=-1))

    return (
        torch.tensor(X, dtype=torch.long),
        torch.tensor(L, dtype=torch.long),
        torch.tensor(D, dtype=torch.long),
        torch.tensor(Y, dtype=torch.long),
    )
