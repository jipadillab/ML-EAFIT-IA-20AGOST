import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuración de la página de Streamlit ---
st.set_page_config(
    page_title="Análisis Interactivo de Árboles de Decisión",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Análisis Interactivo de Árboles de Decisión 🌳")
st.markdown("""
Esta aplicación te permite generar un conjunto de datos simulado, realizar un análisis exploratorio,
**entrenar un modelo de Árbol de Decisión con parámetros personalizables**, visualizar el árbol resultante
y evaluar su desempeño en tareas de clasificación.
""")

# --- Controles de Generación de Datos en el Sidebar ---
st.sidebar.header("Generación de Datos Simulados")
n_samples = st.sidebar.slider("Número de Muestras", min_value=100, max_value=2000, value=300, step=50)
n_features = st.sidebar.slider("Número de Columnas (Características)", min_value=2, max_value=10, value=6, step=1)
n_classes = st.sidebar.slider("Número de Clases", min_value=2, max_value=5, value=2, step=1)
random_state_data = st.sidebar.number_input("Semilla Aleatoria para Datos", value=42, step=1)

# --- Generación del conjunto de datos simulado ---
@st.cache_data
def generate_simulated_data(n_samples, n_features, n_classes, random_state):
    """
    Genera un conjunto de datos simulado para tareas de clasificación.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(n_features, n_features - 1),
        n_redundant=max(0, n_features - min(n_features, n_features - 1) - 1),
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=1.0,
        random_state=random_state,
        flip_y=0.01
    )
    feature_names = [f"Característica_{i+1}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['Clase'] = y
    return df, X, y

# Generar los datos
df, X, y = generate_simulated_data(n_samples, n_features, n_classes, random_state_data)

st.subheader("1. Vista Previa del Conjunto de Datos Simulado")
st.write(f"Conjunto de datos generado con **{df.shape[0]}** muestras y **{df.shape[1]-1}** características.")
st.dataframe(df.head()) # Mostrar las primeras filas del DataFrame

# --- División de los datos en conjuntos de entrenamiento y prueba ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# --- Análisis Exploratorio de Datos (EDA) ---
st.markdown("---")
st.subheader("2. Análisis Exploratorio de Datos (EDA) 📊")

st.markdown("#### Estadísticas Descriptivas")
st.dataframe(df.describe())

st.markdown("#### Matriz de Correlación")
# Excluir la columna 'Clase' para la correlación de características
fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
sns.heatmap(df.drop('Clase', axis=1).corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
plt.title("Matriz de Correlación entre Características")
st.pyplot(fig_corr)

st.markdown("#### Distribución de Características")
selected_features_for_hist = st.multiselect(
    "Selecciona características para ver su distribución:",
    options=df.columns[:-1].tolist(),
    default=df.columns[0:min(3, df.shape[1]-1)].tolist()
)
if selected_features_for_hist:
    for feature in selected_features_for_hist:
        fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
        sns.histplot(df[feature], kde=True, ax=ax_hist, color='skyblue')
        plt.title(f"Distribución de {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frecuencia")
        st.pyplot(fig_hist)

st.markdown("#### Gráfico de Dispersión Interactivo (Primeras dos características)")
st.write("Visualización de las dos primeras características principales por clase.")
fig_scatter, ax_scatter = plt.subplots(figsize=(10, 8))
sns.scatterplot(
    x=df.iloc[:, 0], # Primera característica
    y=df.iloc[:, 1], # Segunda característica
    hue=df['Clase'],
    palette='viridis',
    marker='o',
    s=100,
    edgecolor='k',
    ax=ax_scatter
)
plt.title(f"Dispersión de '{df.columns[0]}' vs '{df.columns[1]}' por Clase")
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
st.pyplot(fig_scatter)

# --- Configuración del Modelo de Árbol de Decisión ---
st.markdown("---")
st.subheader("3. Configuración y Entrenamiento del Árbol de Decisión 🛠️")

st.sidebar.header("Parámetros del Árbol de Decisión")
# Parámetros básicos
max_depth = st.sidebar.slider("Profundidad Máxima (max_depth)", min_value=1, max_value=20, value=7, help="La profundidad máxima del árbol. Limitar esto previene el sobreajuste.")
min_samples_leaf = st.sidebar.slider("Mínimo de Muestras por Hoja (min_samples_leaf)", min_value=1, max_value=20, value=5, help="El número mínimo de muestras requeridas para estar en un nodo hoja. Un valor más alto previene el sobreajuste.")
min_samples_split = st.sidebar.slider("Mínimo de Muestras para Dividir (min_samples_split)", min_value=2, max_value=40, value=10, help="El número mínimo de muestras requeridas para dividir un nodo interno.")

# Parámetros avanzados
criterion = st.sidebar.selectbox(
    "Criterio de División (criterion)",
    ("gini", "entropy", "log_loss"),
    index=0, # gini es el predeterminado
    help="La función para medir la calidad de una división. 'gini' para impureza Gini, 'entropy' para ganancia de información, 'log_loss' para pérdida de logaritmo."
)
splitter = st.sidebar.selectbox(
    "Estrategia de División (splitter)",
    ("best", "random"),
    index=0, # best es el predeterminado
    help="La estrategia utilizada para elegir la división en cada nodo. 'best' selecciona la mejor división, 'random' selecciona la mejor división aleatoria."
)
# Nota: make_classification genera datos con características continuas, por lo que min_impurity_decrease no es directamente aplicable
# y max_features podría limitar demasiado para un número bajo de características.
# Estos podrían ser agregados si se desea mayor complejidad.

st.info("Modelo seleccionado: **Árbol de Decisión**")

# --- Entrenamiento del Modelo ---
model = DecisionTreeClassifier(
    max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,
    min_samples_split=min_samples_split,
    criterion=criterion,
    splitter=splitter,
    random_state=42 # Para reproducibilidad
)

st.write("Entrenando el Árbol de Decisión con los parámetros seleccionados...")
try:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- Resultados del Modelo ---
    st.markdown("---")
    st.subheader("4. Resultados y Desempeño del Árbol de Decisión ✅")

    accuracy = accuracy_score(y_test, y_pred)
    st.metric(label="Exactitud (Accuracy) del Modelo", value=f"{accuracy:.4f}")

    st.markdown("#### Reporte de Clasificación")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    st.markdown("#### Matriz de Confusión")
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    cm_display = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=plt.cm.Blues, ax=ax_cm)
    st.pyplot(fig_cm)

    # --- Visualización del Árbol ---
    st.markdown("---")
    st.subheader("5. Visualización del Árbol de Decisión Final 🌳")
    st.write("Aquí puedes ver la estructura del árbol de decisión que fue entrenado con tus datos y parámetros.")

    # Ajustar tamaño de la figura para que el árbol se vea bien
    fig_tree, ax_tree = plt.subplots(figsize=(25, 15)) # Aumentado el tamaño
    plot_tree(
        model,
        filled=True,
        feature_names=df.columns[:-1].tolist(),
        class_names=[str(c) for c in sorted(df['Clase'].unique())], # Nombres de clase dinámicos
        ax=ax_tree,
        fontsize=10, # Ajustado el tamaño de la fuente
        proportion=True, # Muestra la proporción de muestras en cada nodo
        rounded=True # Bordes redondeados para mejor estética
    )
    plt.title("Estructura del Árbol de Decisión", fontsize=16)
    st.pyplot(fig_tree)

except Exception as e:
    st.error(f"¡Ocurrió un error al entrenar el Árbol de Decisión! Por favor, revisa los parámetros y datos. Error: {e}")

st.markdown("---")
st.markdown("¡Experimenta con los parámetros del árbol de decisión en la barra lateral para ver cómo afectan la estructura y el desempeño!")

