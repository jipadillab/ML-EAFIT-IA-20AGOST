import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuración de la página de Streamlit ---
st.set_page_config(
    page_title="Simulación ML Supervisado",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Aplicación de Simulación de Modelos de ML Supervisados 🧠")
st.markdown("""
Esta aplicación demuestra el entrenamiento, validación y visualización de modelos de clasificación supervisados
utilizando un conjunto de datos simulado. Puedes elegir entre K-Nearest Neighbors (KNN), Árbol de Decisión,
o Clasificador Bayesiano Gausiano.
""")

# --- Generación del conjunto de datos simulado ---
@st.cache_data
def generate_simulated_data(n_samples=300, n_features=6, n_classes=2, random_state=42):
    """
    Genera un conjunto de datos simulado para tareas de clasificación.
    """
    X, y = make_classification(
        n_samples=n_samples,       # Número total de muestras
        n_features=n_features,     # Número total de características
        n_informative=n_features - 1, # Número de características informativas
        n_redundant=1,             # Número de características redundantes
        n_repeated=0,              # Número de características repetidas
        n_classes=n_classes,       # Número de clases de destino
        n_clusters_per_class=1,    # Número de grupos por clase
        class_sep=1.0,             # Separabilidad entre clases
        random_state=random_state, # Semilla para reproducibilidad
        flip_y=0.01                # Fracción de etiquetas aleatorias
    )
    # Crea un DataFrame de Pandas para mejor visualización
    feature_names = [f"Característica_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['Clase'] = y
    return df, X, y

# Generar los datos
df, X, y = generate_simulated_data(n_samples=300, n_features=6)

st.subheader("1. Vista Previa del Conjunto de Datos Simulado")
st.write(f"Conjunto de datos generado con {df.shape[0]} muestras y {df.shape[1]-1} características.")
st.dataframe(df.head()) # Mostrar las primeras filas del DataFrame

# --- División de los datos en conjuntos de entrenamiento y prueba ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

st.subheader("2. Selección del Modelo y Parámetros")

# Sidebar para selección de modelo y parámetros
st.sidebar.header("Configuración del Modelo")
model_choice = st.sidebar.selectbox(
    "Selecciona un Modelo de Clasificación:",
    ("K-Nearest Neighbors (KNN)", "Árbol de Decisión", "Clasificador Bayesiano Gausiano")
)

# --- Entrenamiento y Evaluación del Modelo ---
model = None
st.subheader("3. Resultados del Modelo")

if model_choice == "K-Nearest Neighbors (KNN)":
    st.sidebar.subheader("Parámetros de KNN")
    n_neighbors = st.sidebar.slider("Número de Vecinos (k)", min_value=1, max_value=20, value=5)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    st.info(f"Modelo seleccionado: **K-Nearest Neighbors** con k={n_neighbors}")

elif model_choice == "Árbol de Decisión":
    st.sidebar.subheader("Parámetros del Árbol de Decisión")
    max_depth = st.sidebar.slider("Profundidad Máxima", min_value=3, max_value=15, value=7)
    min_samples_leaf = st.sidebar.slider("Mínimo de Muestras por Hoja", min_value=1, max_value=10, value=3)
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
    st.info(f"Modelo seleccionado: **Árbol de Decisión** con profundidad_máxima={max_depth}, min_muestras_por_hoja={min_samples_leaf}")

elif model_choice == "Clasificador Bayesiano Gausiano":
    model = GaussianNB()
    st.info("Modelo seleccionado: **Clasificador Bayesiano Gausiano**")

if model:
    st.write("Entrenando el modelo...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("### Métricas de Evaluación")
    accuracy = accuracy_score(y_test, y_pred)
    st.metric(label="Exactitud (Accuracy)", value=f"{accuracy:.2f}")

    st.write("#### Reporte de Clasificación")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    st.write("### Matriz de Confusión")
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    cm_display = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=plt.cm.Blues, ax=ax_cm)
    st.pyplot(fig_cm)

    st.subheader("4. Visualizaciones Adicionales")

    if model_choice == "Árbol de Decisión":
        st.write("### Visualización del Árbol de Decisión")
        fig_tree, ax_tree = plt.subplots(figsize=(15, 10))
        plot_tree(model, filled=True, feature_names=df.columns[:-1].tolist(), class_names=["Clase 0", "Clase 1"], ax=ax_tree)
        st.pyplot(fig_tree)

    # Visualización de dispersión para las dos primeras características
    st.write("### Dispersión de las Dos Primeras Características (Conjunto de Prueba)")
    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x=X_test[:, 0],
        y=X_test[:, 1],
        hue=y_test,
        palette='viridis',
        marker='o',
        s=100,
        edgecolor='k',
        ax=ax_scatter
    )
    plt.title("Dispersión de Característica 1 vs Característica 2 por Clase")
    plt.xlabel("Característica 1")
    plt.ylabel("Característica 2")
    st.pyplot(fig_scatter)

    st.markdown("---")
    st.markdown("¡Experimenta con los diferentes modelos y sus parámetros en la barra lateral!")

