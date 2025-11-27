import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import plotly.express as px

from dash import Dash, dcc, html, Input, Output

# ===============================================
# CARGA DE DATOS
# ===============================================
df = pd.read_csv("./input_data/noticias_emociones_con_pca.csv")
df["date"] = pd.to_datetime(df["date"])
df["year_month"] = df["date"].dt.to_period("M").astype(str)
df["domain"] = df["domain"].str.lower().str.strip()

dominios = sorted(df["domain"].unique())
min_date, max_date = df["date"].min(), df["date"].max()

# ===============================================
# DASH APP (TEMA OSCURO)
# ===============================================
external_css = [
    {
        "href": "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap",
        "rel": "stylesheet"
    }
]

app = Dash(__name__, external_stylesheets=external_css)
app.title = "Dashboard Emocional"

# ---------- Estilos Globales ----------
DARK_BG = "#111111"
CARD_BG = "#1c1c1e"
TEXT_COLOR = "#E6E6E6"
ACCENT = "#4d88ff"

STYLE_BOX = {
    "background": CARD_BG,
    "padding": "22px",
    "margin": "16px",
    "color": TEXT_COLOR,
    "borderRadius": "14px",
    "boxShadow": "0 4px 12px rgba(0,0,0,0.35)",
    "border": "1px solid #222"
}

# ===============================================
# HTML Template ajustado (dropdown y datepicker oscuros)
# ===============================================
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <style>
        body {
            margin: 0;
            background-color: #111111;
        }

        /* Dropdown menú oscuro */
        .Select-control {
            background-color: #1c1c1e !important;
            border: 1px solid #333 !important;
            color: #E6E6E6 !important;
        }
        .Select-menu-outer {
            background-color: #1c1c1e !important;
            border: 1px solid #333 !important;
        }
        .Select-placeholder, .Select-value-label {
            color: #E6E6E6 !important;
        }
        .Select-option {
            background-color: #1c1c1e !important;
            color: #E6E6E6 !important;
        }

        /* DatePickerRange oscuro */
        .DateInput_input, .DateInput_input_1 {
            background-color: #1c1c1e !important;
            color: #E6E6E6 !important;
            border: 1px solid #333 !important;
        }

        .DateRangePickerInput {
            background-color: #1c1c1e !important;
            border: 1px solid #333 !important;
        }

        .CalendarDay__default {
            background-color: #1c1c1e !important;
            color: #E6E6E6 !important;
            border: 1px solid #333 !important;
        }

        .CalendarMonth_caption {
            color: #fff !important;
        }

        .CalendarDay__selected {
            background: #4d88ff !important;
            color: white !important;
        }

        .CalendarDay__hovered_span,
        .CalendarDay__selected_span {
            background-color: #2a3f70 !important;
            color: white !important;
        }
        </style>

        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

# ===============================================
# LAYOUT
# ===============================================
app.layout = html.Div(
    style={
        "backgroundColor": DARK_BG,
        "fontFamily": "'Inter', sans-serif",
        "minHeight": "100vh",
        "margin": "0",
        "padding": "20px",
        "overflowX": "hidden"
    },
    children=[
        html.H1(
            "Dashboard emocional de noticias",
            style={
                "textAlign": "center",
                "marginTop": "10px",
                "marginBottom": "25px",
                "color": TEXT_COLOR,
                "fontWeight": "700",
                "letterSpacing": "1px"
            }
        ),

        # =======================================
        #   FILTROS
        # =======================================
        html.Div([
            html.Div([
                html.Label("Fuente", style={"fontWeight": "bold", "color": TEXT_COLOR}),
                dcc.Dropdown(
                    id="dominio",
                    options=[{"label": "Todas", "value": "ALL"}] +
                            [{"label": d, "value": d} for d in dominios],
                    value="ALL",
                    style={
                        "backgroundColor": CARD_BG,
                        "color": TEXT_COLOR,
                        "border": "1px solid #333"
                    }
                )
            ], style={"width": "30%", "display": "inline-block"}),

            html.Div([
                html.Label("Rango de fechas",
                           style={"fontWeight": "bold", "color": TEXT_COLOR}),
                dcc.DatePickerRange(
                    id="fechas",
                    start_date=min_date.date(),
                    end_date=max_date.date(),
                    display_format="YYYY-MM-DD",
                    minimum_nights=0
                )
            ], style={"width": "40%", "display": "inline-block", "paddingLeft": "20px"}),
        ]),

        # =======================================
        #   TABS
        # =======================================
        dcc.Tabs(
            id="tabs",
            value="tab-overview",
            children=[
                dcc.Tab(label="Resumen", value="tab-overview",
                        style={"background": CARD_BG, "color": TEXT_COLOR},
                        selected_style={"background": ACCENT, "color": "#fff"}),

                dcc.Tab(label="Valence", value="tab-valence",
                        style={"background": CARD_BG, "color": TEXT_COLOR},
                        selected_style={"background": ACCENT, "color": "#fff"}),

                dcc.Tab(label="Arousal", value="tab-arousal",
                        style={"background": CARD_BG, "color": TEXT_COLOR},
                        selected_style={"background": ACCENT, "color": "#fff"}),

                dcc.Tab(label="PCA emocional", value="tab-pca",
                        style={"background": CARD_BG, "color": TEXT_COLOR},
                        selected_style={"background": ACCENT, "color": "#fff"})
            ]
        ),

        html.Div(id="contenido-tab", style=STYLE_BOX)
    ]
)

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
        return html.Div("No hay datos con estos filtros.", style={"color": TEXT_COLOR})

    # ========= Tema dark para todas las figuras =========
    px.defaults.template = "plotly_dark"
    px.defaults.color_discrete_sequence = px.colors.qualitative.Plotly

    # Estilo común para las figuras
    def estilo_fig(fig):
        fig.update_layout(
            paper_bgcolor=DARK_BG,
            plot_bgcolor=DARK_BG,
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.07)",
                zeroline=False
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.07)",
                zeroline=False
            ),
            font=dict(color=TEXT_COLOR)
        )
        return fig

    # ---------------- TAB 1 ----------------
    if tab == "tab-overview":
        g = df_f.groupby(["year_month", "domain"]).size().reset_index(name="count")
        fig = px.bar(
            g, x="year_month", y="count", color="domain",
            title="Número de artículos por mes y fuente",
        )
        fig.update_layout(xaxis_tickangle=-45, height=450)
        return dcc.Graph(figure=estilo_fig(fig))

    # ---------------- TAB 2 ----------------
    if tab == "tab-valence":
        g = df_f.groupby(["year_month", "domain"])["valence"].mean().reset_index()
        fig = px.line(
            g, x="year_month", y="valence", color="domain", markers=True,
            title="Valence promedio por mes y fuente"
        )
        fig.update_layout(xaxis_tickangle=-45, height=450)
        return dcc.Graph(figure=estilo_fig(fig))

    # ---------------- TAB 3 ----------------
    if tab == "tab-arousal":
        g = df_f.groupby(["year_month", "domain"])["arousal"].mean().reset_index()
        fig = px.line(
            g, x="year_month", y="arousal", color="domain", markers=True,
            title="Arousal promedio por mes y fuente"
        )
        fig.update_layout(xaxis_tickangle=-45, height=450)
        return dcc.Graph(figure=estilo_fig(fig))

    # ---------------- TAB 4 ----------------
    if tab == "tab-pca":
        fig = px.scatter(
            df_f,
            x="pca_comp1", y="pca_comp2",
            color="domain" if dominio == "ALL" else None,
            hover_data=["title", "date"],
            title="Mapa emocional (PCA)"
        )
        fig.update_layout(height=500)
        return dcc.Graph(figure=estilo_fig(fig))

    return html.Div("Tab no reconocida.", style={"color": TEXT_COLOR})


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    app.run(debug=False)
