import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import plotly.express as px

from dash import Dash, dcc, html, Input, Output

# ======================================================
# CARGA DE DATOS
# ======================================================
df = pd.read_csv("./input_data/noticias_emociones_con_pca.csv")
df["date"] = pd.to_datetime(df["date"])
df["year_month"] = df["date"].dt.to_period("M").astype(str)
df["domain"] = df["domain"].str.lower().str.strip()

dominios = sorted(df["domain"].unique())
min_date, max_date = df["date"].min(), df["date"].max()

# ======================================================
# DASH APP
# ======================================================
app = Dash(__name__)
app.title = "Dashboard Emocional"


# ----------- Estilos globals -----------
STYLE_BOX = {
    "background": "white",
    "padding": "18px",
    "margin": "12px",
    "borderRadius": "10px",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.10)"
}


app.layout = html.Div([

    html.H1("Dashboard emocional de noticias", 
            style={"textAlign": "center", "marginTop": "20px"}),

    html.Div([

        # ---- Selección de fuente ----
        html.Div([
            html.Label("Fuente", style={"fontWeight": "bold"}),
            dcc.Dropdown(
                id="dominio",
                options=[{"label": "Todas", "value": "ALL"}] +
                        [{"label": d, "value": d} for d in dominios],
                value="ALL"
            )
        ], style={"width": "30%", "display": "inline-block"}),

        # ---- Selección de fecha ----
        html.Div([
            html.Label("Rango de fechas", style={"fontWeight": "bold"}),
            dcc.DatePickerRange(
                id="fechas",
                start_date=min_date.date(),
                end_date=max_date.date()
            )
        ], style={"width": "40%", "display": "inline-block"}),

    ], style={"textAlign": "center"}),

    # ------------ Tabs principales -----------------
    dcc.Tabs(id="tabs", value="tab-overview", children=[
        dcc.Tab(label="Resumen", value="tab-overview"),
        dcc.Tab(label="Valence", value="tab-valence"),
        dcc.Tab(label="Arousal", value="tab-arousal"),
        dcc.Tab(label="PCA emocional", value="tab-pca")
    ]),

    html.Div(id="contenido-tab", style=STYLE_BOX)

])


# ======================================================
# CALLBACK PRINCIPAL
# ======================================================
@app.callback(
    Output("contenido-tab", "children"),
    Input("tabs", "value"),
    Input("dominio", "value"),
    Input("fechas", "start_date"),
    Input("fechas", "end_date")
)
def actualizar(tab, dominio, f_ini, f_fin):

    df_f = df.copy()
    df_f = df_f[(df_f["date"] >= f_ini) & (df_f["date"] <= f_fin)]

    if dominio != "ALL":
        df_f = df_f[df_f["domain"] == dominio]

    if df_f.empty:
        return html.Div("No hay datos con estos filtros.")

    # ---------------- TAB 1: RESUMEN -----------------
    if tab == "tab-overview":
        g = df_f.groupby(["year_month", "domain"]).size().reset_index(name="count")
        fig = px.bar(
            g, x="year_month", y="count", color="domain",
            title="Número de artículos por mes y fuente"
        )
        fig.update_layout(xaxis_tickangle=-45, height=450)
        return dcc.Graph(figure=fig)

    # ---------------- TAB 2: VALENCE ------------------
    if tab == "tab-valence":
        g = df_f.groupby(["year_month", "domain"])["valence"].mean().reset_index()
        fig = px.line(
            g, x="year_month", y="valence", color="domain", markers=True,
            title="Valence promedio por mes y fuente"
        )
        fig.update_layout(xaxis_tickangle=-45, height=450)
        return dcc.Graph(figure=fig)

    # ---------------- TAB 3: AROUSAL ------------------
    if tab == "tab-arousal":
        g = df_f.groupby(["year_month", "domain"])["arousal"].mean().reset_index()
        fig = px.line(
            g, x="year_month", y="arousal", color="domain", markers=True,
            title="Arousal promedio por mes y fuente"
        )
        fig.update_layout(xaxis_tickangle=-45, height=450)
        return dcc.Graph(figure=fig)

    # ---------------- TAB 4: PCA ------------------
    if tab == "tab-pca":
        fig = px.scatter(
            df_f,
            x="pca_comp1", y="pca_comp2",
            color="domain" if dominio=="ALL" else None,
            hover_data=["title", "date"],
            title="Mapa emocional (PCA)"
        )
        fig.update_layout(height=500)
        return dcc.Graph(figure=fig)

    return html.Div("Tab no reconocida.")


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    app.run(debug=False)
