import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin


# ============================================================
# Clase personalizada usada durante el entrenamiento
# Debe estar definida ANTES de cargar el modelo .pkl
# ============================================================

class TopNCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, top_n=15, other_label="OTROS_DISTRITOS"):
        self.top_n = top_n
        self.other_label = other_label

    def fit(self, X, y=None):
        serie = pd.Series(np.asarray(X).ravel()).astype(str).str.strip().str.upper()
        self.top_categories_ = serie.value_counts().head(self.top_n).index.tolist()
        return self

    def transform(self, X):
        serie = pd.Series(np.asarray(X).ravel()).astype(str).str.strip().str.upper()
        serie = serie.where(serie.isin(self.top_categories_), self.other_label)
        return serie.to_numpy().reshape(-1, 1)


# ============================================================
# Configuración de rutas
# ============================================================

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "modelo_final_xgboost_obesidad.pkl"
OPTIONS_PATH = BASE_DIR / "models" / "opciones_app_obesidad.pkl"


# ============================================================
# Carga de modelo y opciones
# ============================================================

@st.cache_resource
def cargar_modelo():
    return joblib.load(MODEL_PATH)


@st.cache_data
def cargar_opciones():
    return joblib.load(OPTIONS_PATH)


modelo = cargar_modelo()
opciones_app = cargar_opciones()


# ============================================================
# Configuración de la página
# ============================================================

st.set_page_config(
    page_title="Predicción de Prioridad Sanitaria",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Predicción de Prioridad Sanitaria")

st.markdown(
    """
    Esta aplicación utiliza un modelo **XGBoost Classifier optimizado**
    para predecir el nivel de prioridad sanitaria de un segmento poblacional-clínico
    registrado con obesidad diagnosticada.

    El modelo clasifica el segmento en una de tres categorías:
    **Prioridad Baja**, **Prioridad Media** o **Prioridad Alta**.
    """
)


# ============================================================
# Formulario de ingreso de datos
# ============================================================

st.subheader("Datos del segmento poblacional-clínico")

col1, col2 = st.columns(2)

with col1:
    distrito = st.selectbox(
        "Distrito de residencia",
        opciones_app["distritos"]
    )

    grupo_edad = st.selectbox(
        "Grupo de edad",
        opciones_app["grupos_edad"],
        index=opciones_app["grupos_edad"].index("30-59A")
        if "30-59A" in opciones_app["grupos_edad"] else 0
    )

with col2:
    genero = st.selectbox(
        "Género",
        opciones_app["generos"]
    )

    diagnostico = st.selectbox(
        "Diagnóstico agrupado",
        opciones_app["diagnosticos"]
    )


# ============================================================
# Diccionario de interpretación
# ============================================================

label_map = {
    0: "Prioridad Baja",
    1: "Prioridad Media",
    2: "Prioridad Alta"
}


# ============================================================
# Predicción
# ============================================================

if st.button("Predecir prioridad sanitaria"):

    nuevo_segmento = pd.DataFrame([{
        "distrito_rh_paciente": distrito,
        "grupo_edad": grupo_edad,
        "id_genero": genero,
        "diag_agrupado": diagnostico
    }])

    # Normalización textual para mantener coherencia con el entrenamiento
    for col in nuevo_segmento.columns:
        nuevo_segmento[col] = nuevo_segmento[col].astype(str).str.strip().str.upper()

    pred = modelo.predict(nuevo_segmento)[0]
    probas = modelo.predict_proba(nuevo_segmento)[0]

    # Asegurar correspondencia entre clase y probabilidad
    clases_modelo = modelo.classes_
    prob_dict = {int(clase): prob for clase, prob in zip(clases_modelo, probas)}

    prob_bajo = prob_dict.get(0, 0)
    prob_medio = prob_dict.get(1, 0)
    prob_alto = prob_dict.get(2, 0)

    prioridad = label_map.get(int(pred), str(pred))

    st.subheader("Resultado del modelo")

    if int(pred) == 2:
        st.error(f"Nivel de prioridad predicho: {prioridad}")
        recomendacion = (
            "Se recomienda priorizar este segmento para campañas preventivas, "
            "seguimiento nutricional y asignación de recursos sanitarios."
        )
    elif int(pred) == 1:
        st.warning(f"Nivel de prioridad predicho: {prioridad}")
        recomendacion = (
            "Se recomienda mantener seguimiento preventivo y monitorear la evolución "
            "del segmento."
        )
    else:
        st.success(f"Nivel de prioridad predicho: {prioridad}")
        recomendacion = (
            "Se recomienda mantener monitoreo rutinario, sin priorización inmediata."
        )

    st.markdown("### Probabilidades por clase")

    colp1, colp2, colp3 = st.columns(3)

    colp1.metric("Prioridad Baja", f"{prob_bajo * 100:.1f}%")
    colp2.metric("Prioridad Media", f"{prob_medio * 100:.1f}%")
    colp3.metric("Prioridad Alta", f"{prob_alto * 100:.1f}%")

    st.markdown("### Recomendación")
    st.info(recomendacion)

    st.markdown("### Datos ingresados")
    st.dataframe(nuevo_segmento)

    st.caption(
        "Nota: la predicción corresponde a un segmento poblacional-clínico agregado, "
        "no a un diagnóstico individual de paciente."
    )


# ============================================================
# Información técnica del modelo
# ============================================================

st.markdown("---")
st.subheader("Información del modelo")

st.markdown(
    """
    - **Modelo final:** XGBoost Classifier optimizado.
    - **Tipo de problema:** Clasificación supervisada multiclase.
    - **Variable objetivo:** `nivel_prioridad`.
    - **Clases:** Bajo, Medio y Alto.
    - **Métrica principal:** F1-score macro.
    - **Métrica crítica de negocio:** Recall de la clase Alto.
    - **Motivo:** La clase Alto representa los segmentos poblacionales con mayor prioridad sanitaria.
    """
)