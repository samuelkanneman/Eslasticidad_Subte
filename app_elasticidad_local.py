import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# Configuración de la página
st.set_page_config(
    page_title="Elasticidad Precio Demanda Subte",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Funciones auxiliares
@st.cache_data
def cargar_datos_real(ruta_archivo):
    """Carga el dataset real desde el archivo parquet"""
    try:
        df = pd.read_parquet(ruta_archivo)
        
        # Verificar que tenga las columnas necesarias
        columnas_requeridas = ['fecha', 'precio', 'pax_pago', 'mes']
        columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
        
        if columnas_faltantes:
            st.error(f"⚠️ El archivo no tiene las columnas requeridas: {columnas_faltantes}")
            return None
        
        # Convertir fecha si es necesario
        if not pd.api.types.is_datetime64_any_dtype(df['fecha']):
            df['fecha'] = pd.to_datetime(df['fecha'])
        
        return df
    except Exception as e:
        st.error(f"❌ Error al cargar el archivo: {str(e)}")
        return None

@st.cache_data
def preparar_datos(df):
    """Prepara los datos para el modelo"""
    df = df.copy()
    df['ln_q'] = np.log(df['pax_pago'])
    df['ln_p'] = np.log(df['precio'])
    df['t'] = np.arange(len(df))
    
    # Crear dummies de mes
    mes_d = pd.get_dummies(df['mes'], prefix='mes', drop_first=True)
    
    # Variables independientes
    X = pd.concat([df[['ln_p', 't']], mes_d], axis=1)
    y = df['ln_q']
    
    return X, y, df

@st.cache_data
def entrenar_modelos(X, y, split_ratio=0.6):
    """Entrena los tres modelos y devuelve resultados"""
    split_idx = int(split_ratio * len(X))
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    # Definir modelos
    modelos = {
        'OLS (Baseline)': LinearRegression(),
        'Ridge (α óptimo)': RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000], cv=5),
        'Lasso (α óptimo)': LassoCV(alphas=[0.001, 0.01, 0.1, 1, 10], cv=5, max_iter=10000)
    }
    
    resultados = {}
    predicciones_train = {}
    predicciones_test = {}
    
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        
        y_pred_test = modelo.predict(X_test)
        y_pred_train = modelo.predict(X_train)
        
        resultados[nombre] = {
            'R² (Train)': r2_score(y_train, y_pred_train),
            'R² (Test)': r2_score(y_test, y_pred_test),
            'RMSE (Test)': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'MAE (Test)': mean_absolute_error(y_test, y_pred_test),
            'Elasticidad': modelo.coef_[0],
            'Alpha': getattr(modelo, 'alpha_', 'N/A'),
            'modelo': modelo
        }
        
        predicciones_train[nombre] = y_pred_train
        predicciones_test[nombre] = y_pred_test
    
    # Cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    for nombre, modelo in modelos.items():
        scores = cross_val_score(modelo, X, y, cv=tscv, scoring='r2')
        resultados[nombre]['R² (CV)'] = scores.mean()
        resultados[nombre]['R² (CV Std)'] = scores.std()
    
    return resultados, predicciones_train, predicciones_test, split_idx, X_train, X_test, y_train, y_test

def calcular_impacto_precio(elasticidad, cambio_precio_pct):
    """Calcula el cambio en la demanda dado un cambio en el precio"""
    cambio_demanda_pct = elasticidad * cambio_precio_pct
    return cambio_demanda_pct

# ============================================
# CONFIGURACIÓN INICIAL Y CARGA DE DATOS
# ============================================

# Buscar el archivo parquet en el directorio actual
RUTA_ARCHIVO = "subte_demanda_precio_mensual.parquet"

# Verificar si existe el archivo
if not os.path.exists(RUTA_ARCHIVO):
    st.error(f"""
    ❌ **Archivo no encontrado: `{RUTA_ARCHIVO}`**
    
    Por favor, asegúrate de colocar el archivo `subte_demanda_precio_mensual.parquet` 
    en el mismo directorio que esta aplicación.
    
    **Ubicación esperada:** `{os.path.abspath(RUTA_ARCHIVO)}`
    """)
    st.stop()

# Cargar datos
df_original = cargar_datos_real(RUTA_ARCHIVO)

if df_original is None:
    st.stop()

# Mostrar información del dataset cargado
st.sidebar.success(f"✅ Dataset cargado exitosamente!")
st.sidebar.info(f"""
**📊 Información del Dataset:**
- **Observaciones:** {len(df_original)}
- **Período:** {df_original['fecha'].min().strftime('%Y-%m')} a {df_original['fecha'].max().strftime('%Y-%m')}
- **Precio promedio:** ${df_original['precio'].mean():.2f}
- **Demanda promedio:** {int(df_original['pax_pago'].mean()):,} pasajeros
""")

# Preparar datos y entrenar modelos
X, y, df = preparar_datos(df_original)
resultados, pred_train, pred_test, split_idx, X_train, X_test, y_train, y_test = entrenar_modelos(X, y)

# Sidebar
st.sidebar.title("🚇 Navegación")
pagina = st.sidebar.radio(
    "Ir a:",
    ["🏠 Inicio", "📊 Comparación Modelos", "🎯 Calculadora", 
     "📈 Series Temporales", "🔍 Sensibilidad", "🎲 Simulador", "📄 Datos"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Información del Proyecto")
st.sidebar.info("""
**Trabajo Práctico**  
Metodología de la Investigación

**Tema:** Elasticidad Precio de la Demanda del Subte

**Modelos:**
- OLS (Baseline)
- Ridge Regression
- Lasso Regression

**Dataset:** Datos reales SBASE 2014-2019
""")

# ============================================
# PÁGINA 1: INICIO
# ============================================
if pagina == "🏠 Inicio":
    st.markdown('<h1 class="main-header">🚇 Análisis de Elasticidad Precio - Demanda del Subte</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 👋 Bienvenido al Tablero Interactivo
    
    Este tablero presenta los resultados del análisis de **elasticidad precio de la demanda** del sistema de subterráneos de Buenos Aires,
    utilizando **datos reales de SBASE (2014-2019)** y tres modelos de regresión lineal con diferentes técnicas de regularización.
    """)
    
    # Métricas principales
    st.markdown("### 📊 Resumen Ejecutivo")
    
    mejor_modelo = max(resultados, key=lambda x: resultados[x]['R² (Test)'])
    elasticidad_promedio = np.mean([resultados[m]['Elasticidad'] for m in resultados.keys()])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🏆 Mejor Modelo",
            value=mejor_modelo.split(' ')[0],
            delta=f"R² = {resultados[mejor_modelo]['R² (Test)']:.4f}"
        )
    
    with col2:
        st.metric(
            label="📉 Elasticidad Promedio",
            value=f"{elasticidad_promedio:.4f}",
            delta="Demanda Inelástica" if abs(elasticidad_promedio) < 1 else "Demanda Elástica"
        )
    
    with col3:
        st.metric(
            label="🎯 RMSE (Mejor)",
            value=f"{resultados[mejor_modelo]['RMSE (Test)']:.4f}",
            delta=f"MAE: {resultados[mejor_modelo]['MAE (Test)']:.4f}"
        )
    
    with col4:
        st.metric(
            label="📊 Observaciones",
            value=len(df),
            delta=f"Test: {len(y_test)}"
        )
    
    st.markdown("---")
    
    # Interpretación
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔍 Interpretación de Resultados")
        st.markdown(f"""
        <div class="info-box">
        <b>Elasticidad Precio de la Demanda: {elasticidad_promedio:.4f}</b><br><br>
        
        ✓ La demanda es <b>{'INELÁSTICA' if abs(elasticidad_promedio) < 1 else 'ELÁSTICA'}</b><br>
        ✓ Un aumento del <b>1%</b> en el precio genera una reducción del <b>{abs(elasticidad_promedio):.2f}%</b> en la demanda<br>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📈 Performance de Modelos")
        
        # Gráfico comparativo
        df_metricas = pd.DataFrame({
            'Modelo': list(resultados.keys()),
            'R² Test': [resultados[m]['R² (Test)'] for m in resultados.keys()],
            'RMSE': [resultados[m]['RMSE (Test)'] for m in resultados.keys()]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_metricas['Modelo'],
            y=df_metricas['R² Test'],
            name='R² Test',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title='R² en Test Set',
            yaxis_title='R²',
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Tabla de resultados
    st.markdown("### 📋 Tabla Comparativa de Modelos")
    
    df_resultados = pd.DataFrame(resultados).T
    df_resultados = df_resultados.drop('modelo', axis=1)
    df_resultados = df_resultados.round(4)
    
    st.dataframe(
        df_resultados.style.highlight_max(axis=0, subset=['R² (Train)', 'R² (Test)', 'R² (CV)'], color='lightgreen')
                          .highlight_min(axis=0, subset=['RMSE (Test)', 'MAE (Test)', 'R² (CV Std)'], color='lightgreen'),
        use_container_width=True
    )
    
    # Conclusiones
    st.markdown("### 💡 Conclusiones Principales")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success(f"""
        **✓ Generalización**  
        Diferencia Train-Test: {abs(resultados[mejor_modelo]['R² (Train)'] - resultados[mejor_modelo]['R² (Test)']):.4f}  
        {'Sin overfitting significativo' if abs(resultados[mejor_modelo]['R² (Train)'] - resultados[mejor_modelo]['R² (Test)']) < 0.1 else 'Posible overfitting'}
        """)
    
    with col2:
        st.info(f"""
        **📊 Validación Cruzada**  
        R² CV (mejor): {max([resultados[m]['R² (CV)'] for m in resultados.keys()]):.4f}  
        Estabilidad confirmada con Time Series Split
        """)
    
    with col3:
        st.warning(f"""
        **🎯 Regularización**  
        Ridge α: {resultados['Ridge (α óptimo)']['Alpha']:.4f}  
        Lasso α: {resultados['Lasso (α óptimo)']['Alpha']:.4f}
        """)

# ============================================
# PÁGINA 2: COMPARACIÓN DE MODELOS
# ============================================
elif pagina == "📊 Comparación Modelos":
    st.title("📊 Comparación Detallada de Modelos")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Real vs Predicho", "📉 Residuos", "🎯 Métricas", "🔄 Cross-Validation"])
    
    # Tab 1: Real vs Predicho
    with tab1:
        st.markdown("### Predicciones en Test Set")
        
        col1, col2, col3 = st.columns(3)
        
        for col, (nombre, y_pred) in zip([col1, col2, col3], pred_test.items()):
            with col:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=y_test,
                    y=y_pred,
                    mode='markers',
                    name='Predicciones',
                    marker=dict(size=8, opacity=0.6)
                ))
                
                # Línea identidad
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Línea ideal',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title=f"{nombre}<br>R² = {resultados[nombre]['R² (Test)']:.4f}",
                    xaxis_title='ln_q Real',
                    yaxis_title='ln_q Predicho',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Residuos
    with tab2:
        st.markdown("### Análisis de Residuos")
        
        col1, col2, col3 = st.columns(3)
        
        for col, (nombre, y_pred) in zip([col1, col2, col3], pred_test.items()):
            with col:
                residuos = y_test.values - y_pred
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=y_pred,
                    y=residuos,
                    mode='markers',
                    name='Residuos',
                    marker=dict(size=8, opacity=0.6)
                ))
                
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                
                fig.update_layout(
                    title=f"{nombre}<br>MAE = {resultados[nombre]['MAE (Test)']:.4f}",
                    xaxis_title='ln_q Predicho',
                    yaxis_title='Residuos',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Métricas
    with tab3:
        st.markdown("### Comparación de Métricas")
        
        metricas = ['R² (Test)', 'RMSE (Test)', 'MAE (Test)', 'Elasticidad']
        
        for metrica in metricas:
            st.markdown(f"#### {metrica}")
            
            valores = [resultados[m][metrica] for m in resultados.keys()]
            nombres = list(resultados.keys())
            
            fig = go.Figure(go.Bar(
                x=valores,
                y=nombres,
                orientation='h',
                text=[f'{v:.4f}' for v in valores],
                textposition='auto',
            ))
            
            fig.update_layout(
                xaxis_title=metrica,
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Cross-Validation
    with tab4:
        st.markdown("### Validación Cruzada (Time Series Split)")
        
        df_cv = pd.DataFrame({
            'Modelo': list(resultados.keys()),
            'R² CV': [resultados[m]['R² (CV)'] for m in resultados.keys()],
            'Std': [resultados[m]['R² (CV Std)'] for m in resultados.keys()]
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='R² CV',
            x=df_cv['Modelo'],
            y=df_cv['R² CV'],
            error_y=dict(type='data', array=df_cv['Std']),
            text=df_cv['R² CV'].round(4),
            textposition='auto'
        ))
        
        fig.update_layout(
            title='R² en Validación Cruzada con Barras de Error',
            yaxis_title='R² (Media ± Std)',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df_cv.style.format({'R² CV': '{:.4f}', 'Std': '{:.4f}'}), use_container_width=True)

# ============================================
# PÁGINA 3: CALCULADORA DE ELASTICIDAD
# ============================================
elif pagina == "🎯 Calculadora":
    st.title("🎯 Calculadora de Elasticidad Precio")
    
    st.markdown("""
    ### ¿Cómo impacta un cambio en el precio a la demanda?
    Utiliza esta calculadora para estimar el cambio en la cantidad demandada ante variaciones en el precio.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Configuración")
        
        cambio_precio = st.slider(
            "Cambio en el precio (%)",
            min_value=-50,
            max_value=50,
            value=10,
            step=1,
            help="Selecciona el cambio porcentual en el precio"
        )
        
        modelo_seleccionado = st.selectbox(
            "Modelo para cálculo",
            list(resultados.keys()),
            help="Selecciona el modelo a utilizar para la estimación"
        )
        
        precio_actual = st.number_input(
            "Precio actual ($)",
            min_value=1.0,
            max_value=100.0,
            value=float(df['precio'].iloc[-1]),
            step=0.5
        )
        
        demanda_actual = st.number_input(
            "Demanda actual (pasajeros)",
            min_value=100000,
            max_value=50000000,
            value=int(df['pax_pago'].iloc[-1]),
            step=10000
        )
    
    with col2:
        st.markdown("### 📊 Resultados")
        
        elasticidad = resultados[modelo_seleccionado]['Elasticidad']
        cambio_demanda = calcular_impacto_precio(elasticidad, cambio_precio)
        
        nuevo_precio = precio_actual * (1 + cambio_precio/100)
        nueva_demanda = demanda_actual * (1 + cambio_demanda/100)
        
        # Mostrar resultados
        st.markdown("#### 📈 Escenario Simulado")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.metric(
                label="💰 Precio Nuevo",
                value=f"${nuevo_precio:.2f}",
                delta=f"{cambio_precio:+.1f}%"
            )
            
            st.metric(
                label="👥 Demanda Nueva",
                value=f"{int(nueva_demanda):,}",
                delta=f"{cambio_demanda:+.2f}%",
                delta_color="inverse"
            )
        
        with col_b:
            st.metric(
                label="💵 Ingreso Actual",
                value=f"${precio_actual * demanda_actual:,.0f}"
            )
            
            st.metric(
                label="💵 Ingreso Nuevo",
                value=f"${nuevo_precio * nueva_demanda:,.0f}",
                delta=f"{((nuevo_precio * nueva_demanda) / (precio_actual * demanda_actual) - 1) * 100:+.2f}%"
            )
        
        # Gráfico de comparación
        st.markdown("#### 📊 Visualización del Impacto")
        
        df_comparacion = pd.DataFrame({
            'Escenario': ['Actual', 'Nuevo'],
            'Precio': [precio_actual, nuevo_precio],
            'Demanda': [demanda_actual, nueva_demanda]
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Precio',
            x=df_comparacion['Escenario'],
            y=df_comparacion['Precio'],
            yaxis='y',
            marker_color='lightblue',
            text=df_comparacion['Precio'].round(2),
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name='Demanda',
            x=df_comparacion['Escenario'],
            y=df_comparacion['Demanda'],
            yaxis='y2',
            marker_color='lightcoral',
            text=df_comparacion['Demanda'].astype(int),
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Comparación: Actual vs Nuevo Escenario',
            yaxis=dict(title='Precio ($)', side='left'),
            yaxis2=dict(title='Demanda (pasajeros)', overlaying='y', side='right'),
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparación entre modelos
        st.markdown("#### 🔄 Comparación Entre Modelos")
        
        resultados_modelos = []
        for nombre in resultados.keys():
            elast = resultados[nombre]['Elasticidad']
            cambio_dem = calcular_impacto_precio(elast, cambio_precio)
            nueva_dem = demanda_actual * (1 + cambio_dem/100)
            
            resultados_modelos.append({
                'Modelo': nombre,
                'Elasticidad': elast,
                'Cambio Demanda (%)': cambio_dem,
                'Nueva Demanda': int(nueva_dem),
                'Nuevo Ingreso': nuevo_precio * nueva_dem
            })
        
        df_resultados_modelos = pd.DataFrame(resultados_modelos)
        st.dataframe(
            df_resultados_modelos.style.format({
                'Elasticidad': '{:.4f}',
                'Cambio Demanda (%)': '{:+.2f}%',
                'Nueva Demanda': '{:,.0f}',
                'Nuevo Ingreso': '${:,.0f}'
            }),
            use_container_width=True
        )

# ============================================
# PÁGINA 4: SERIES TEMPORALES
# ============================================
elif pagina == "📈 Series Temporales":
    st.title("📈 Análisis de Series Temporales")
    
    st.markdown("### Predicciones en Test Set")
    
    # Gráfico principal
    df_plot = df.iloc[split_idx:].copy()
    df_plot['Real'] = y_test.values
    
    for nombre, y_pred in pred_test.items():
        df_plot[nombre] = y_pred
    
    fig = go.Figure()
    
    # Serie real
    fig.add_trace(go.Scatter(
        x=df_plot['fecha'],
        y=df_plot['Real'],
        mode='lines+markers',
        name='Real',
        line=dict(color='black', width=3),
        marker=dict(size=8)
    ))
    
    # Predicciones
    colores = ['blue', 'orange', 'purple']
    for nombre, color in zip(pred_test.keys(), colores):
        fig.add_trace(go.Scatter(
            x=df_plot['fecha'],
            y=df_plot[nombre],
            mode='lines+markers',
            name=nombre,
            line=dict(color=color, width=2, dash='dash'),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title='Predicciones vs Real en Test Set',
        xaxis_title='Fecha',
        yaxis_title='ln_q (log de pasajeros)',
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Métricas por período
    st.markdown("### 📊 Estadísticas del Test Set")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📅 Período Test", f"{df['fecha'].iloc[split_idx].strftime('%Y-%m')} a {df['fecha'].iloc[-1].strftime('%Y-%m')}")
    
    with col2:
        st.metric("📊 Observaciones", len(y_test))
    
    with col3:
        st.metric("📈 Rango ln_q", f"{y_test.min():.2f} - {y_test.max():.2f}")
    
    # Gráfico de errores acumulados
    st.markdown("### 📉 Errores Acumulados por Modelo")
    
    fig = go.Figure()
    
    for nombre, y_pred in pred_test.items():
        errores_abs = np.abs(y_test.values - y_pred)
        errores_cum = np.cumsum(errores_abs)
        
        fig.add_trace(go.Scatter(
            x=df_plot['fecha'],
            y=errores_cum,
            mode='lines',
            name=nombre,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title='Errores Absolutos Acumulados',
        xaxis_title='Fecha',
        yaxis_title='Error Acumulado',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Análisis mensual
    st.markdown("### 📅 Error Promedio por Mes")
    
    df_plot['mes'] = pd.to_datetime(df_plot['fecha']).dt.month
    
    errores_mensuales = {}
    for nombre, y_pred in pred_test.items():
        df_plot[f'error_{nombre}'] = np.abs(y_test.values - y_pred)
        errores_mensuales[nombre] = df_plot.groupby('mes')[f'error_{nombre}'].mean()
    
    df_errores_mes = pd.DataFrame(errores_mensuales)
    
    fig = go.Figure()
    
    for nombre in pred_test.keys():
        fig.add_trace(go.Bar(
            name=nombre,
            x=df_errores_mes.index,
            y=df_errores_mes[nombre],
        ))
    
    fig.update_layout(
        title='MAE Promedio por Mes',
        xaxis_title='Mes',
        yaxis_title='MAE',
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# PÁGINA 5: ANÁLISIS DE SENSIBILIDAD
# ============================================
elif pagina == "🔍 Sensibilidad":
    st.title("🔍 Análisis de Sensibilidad - Regularización")
    
    st.markdown("""
    ### Impacto del Parámetro α en la Elasticidad
    Analiza cómo el parámetro de regularización (α) afecta la estimación de la elasticidad precio.
    """)
    
    # Calcular curvas de regularización
    alphas = np.logspace(-3, 3, 50)
    coefs_ridge = []
    coefs_lasso = []
    r2_ridge = []
    r2_lasso = []
    
    with st.spinner('Calculando curvas de sensibilidad...'):
        for alpha in alphas:
            # Ridge
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_train, y_train)
            coefs_ridge.append(ridge.coef_[0])
            r2_ridge.append(r2_score(y_test, ridge.predict(X_test)))
            
            # Lasso
            lasso = Lasso(alpha=alpha, max_iter=10000)
            lasso.fit(X_train, y_train)
            coefs_lasso.append(lasso.coef_[0])
            r2_lasso.append(r2_score(y_test, lasso.predict(X_test)))
    
    tab1, tab2 = st.tabs(["📉 Elasticidad vs α", "📊 R² vs α"])
    
    # Tab 1: Elasticidad vs Alpha
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Ridge")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=alphas,
                y=coefs_ridge,
                mode='lines',
                name='Elasticidad',
                line=dict(color='orange', width=3)
            ))
            
            # Línea OLS
            fig.add_hline(
                y=resultados['OLS (Baseline)']['Elasticidad'],
                line_dash="dash",
                line_color="blue",
                annotation_text="OLS"
            )
            
            # Alpha óptimo
            fig.add_vline(
                x=resultados['Ridge (α óptimo)']['Alpha'],
                line_dash="dash",
                line_color="red",
                annotation_text=f"α óptimo = {resultados['Ridge (α óptimo)']['Alpha']:.3f}"
            )
            
            fig.update_xaxes(type="log", title="Alpha (α)")
            fig.update_yaxes(title="Elasticidad (β_precio)")
            fig.update_layout(height=400, title="Ridge: Elasticidad vs α")
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Lasso")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=alphas,
                y=coefs_lasso,
                mode='lines',
                name='Elasticidad',
                line=dict(color='purple', width=3)
            ))
            
            # Línea OLS
            fig.add_hline(
                y=resultados['OLS (Baseline)']['Elasticidad'],
                line_dash="dash",
                line_color="blue",
                annotation_text="OLS"
            )
            
            # Alpha óptimo
            fig.add_vline(
                x=resultados['Lasso (α óptimo)']['Alpha'],
                line_dash="dash",
                line_color="red",
                annotation_text=f"α óptimo = {resultados['Lasso (α óptimo)']['Alpha']:.3f}"
            )
            
            fig.update_xaxes(type="log", title="Alpha (α)")
            fig.update_yaxes(title="Elasticidad (β_precio)")
            fig.update_layout(height=400, title="Lasso: Elasticidad vs α")
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: R² vs Alpha
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Ridge - R² en Test Set")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=alphas,
                y=r2_ridge,
                mode='lines',
                name='R² Test',
                line=dict(color='green', width=3)
            ))
            
            # Alpha óptimo
            idx_optimo = np.argmax(r2_ridge)
            fig.add_vline(
                x=alphas[idx_optimo],
                line_dash="dash",
                line_color="red",
                annotation_text=f"α óptimo"
            )
            
            fig.update_xaxes(type="log", title="Alpha (α)")
            fig.update_yaxes(title="R² (Test)")
            fig.update_layout(height=400, title="Ridge: Performance vs α")
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Lasso - R² en Test Set")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=alphas,
                y=r2_lasso,
                mode='lines',
                name='R² Test',
                line=dict(color='green', width=3)
            ))
            
            # Alpha óptimo
            idx_optimo = np.argmax(r2_lasso)
            fig.add_vline(
                x=alphas[idx_optimo],
                line_dash="dash",
                line_color="red",
                annotation_text=f"α óptimo"
            )
            
            fig.update_xaxes(type="log", title="Alpha (α)")
            fig.update_yaxes(title="R² (Test)")
            fig.update_layout(height=400, title="Lasso: Performance vs α")
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Información adicional
    st.markdown("### 💡 Interpretación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Ridge Regression:**
        - Reduce la magnitud de los coeficientes
        - α bajo → cercano a OLS
        - α alto → coeficientes tienden a 0
        - No hace selección de variables
        """)
    
    with col2:
        st.info("""
        **Lasso Regression:**
        - Puede reducir coeficientes exactamente a 0
        - Realiza selección de variables
        - α alto → modelo más sparse
        - Útil cuando hay muchas variables
        """)

# ============================================
# PÁGINA 6: SIMULADOR DE ESCENARIOS
# ============================================
elif pagina == "🎲 Simulador":
    st.title("🎲 Simulador de Escenarios")
    
    st.markdown("""
    ### Predice la demanda para diferentes configuraciones
    Configura el precio y el mes para obtener predicciones de los tres modelos.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Configuración del Escenario")
        
        precio_sim = st.number_input(
            "Precio ($)",
            min_value=1.0,
            max_value=50.0,
            value=float(df['precio'].iloc[-1]),
            step=0.5
        )
        
        mes_sim = st.selectbox(
            "Mes",
            range(1, 13),
            format_func=lambda x: ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                                   'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'][x-1]
        )
        
        tendencia_sim = st.slider(
            "Tendencia temporal (t)",
            min_value=0,
            max_value=200,
            value=int(df['t'].iloc[-1])
        )
        
        st.markdown("---")
        
        # Botón de predicción
        if st.button("🎯 Calcular Predicción", type="primary"):
            st.session_state.simular = True
    
    with col2:
        if 'simular' in st.session_state and st.session_state.simular:
            st.markdown("### 📊 Resultados de la Simulación")
            
            # Preparar input
            ln_p_sim = np.log(precio_sim)
            
            # Crear dummies de mes
            mes_dummies = np.zeros(11)  # drop_first=True, entonces 11 dummies
            if mes_sim > 1:
                mes_dummies[mes_sim - 2] = 1
            
            X_sim = np.concatenate([[ln_p_sim, tendencia_sim], mes_dummies]).reshape(1, -1)
            
            # Predecir con cada modelo
            predicciones_sim = {}
            for nombre, datos in resultados.items():
                modelo = datos['modelo']
                ln_q_pred = modelo.predict(X_sim)[0]
                q_pred = np.exp(ln_q_pred)
                predicciones_sim[nombre] = {
                    'ln_q': ln_q_pred,
                    'q': q_pred,
                    'ingreso': precio_sim * q_pred
                }
            
            # Mostrar resultados
            st.markdown("#### 📈 Predicciones")
            
            df_pred_sim = pd.DataFrame({
                'Modelo': list(predicciones_sim.keys()),
                'ln_q Predicho': [predicciones_sim[m]['ln_q'] for m in predicciones_sim.keys()],
                'Pasajeros Predichos': [int(predicciones_sim[m]['q']) for m in predicciones_sim.keys()],
                'Ingreso Estimado ($)': [predicciones_sim[m]['ingreso'] for m in predicciones_sim.keys()]
            })
            
            st.dataframe(
                df_pred_sim.style.format({
                    'ln_q Predicho': '{:.4f}',
                    'Pasajeros Predichos': '{:,.0f}',
                    'Ingreso Estimado ($)': '${:,.2f}'
                }),
                use_container_width=True
            )
            
            # Gráfico comparativo
            st.markdown("#### 📊 Comparación Visual")
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=df_pred_sim['Modelo'],
                y=df_pred_sim['Pasajeros Predichos'],
                text=df_pred_sim['Pasajeros Predichos'],
                texttemplate='%{text:,.0f}',
                textposition='auto',
                marker_color=['lightblue', 'lightcoral', 'lightgreen']
            ))
            
            fig.update_layout(
                title='Pasajeros Predichos por Modelo',
                yaxis_title='Pasajeros',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Estadísticas
            st.markdown("#### 📊 Estadísticas del Escenario")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                promedio_pasajeros = np.mean([predicciones_sim[m]['q'] for m in predicciones_sim.keys()])
                st.metric("👥 Pasajeros (Promedio)", f"{int(promedio_pasajeros):,}")
            
            with col_b:
                promedio_ingreso = np.mean([predicciones_sim[m]['ingreso'] for m in predicciones_sim.keys()])
                st.metric("💰 Ingreso (Promedio)", f"${promedio_ingreso:,.2f}")
            
            with col_c:
                std_pasajeros = np.std([predicciones_sim[m]['q'] for m in predicciones_sim.keys()])
                cv = (std_pasajeros / promedio_pasajeros) * 100
                st.metric("📊 Coef. Variación", f"{cv:.2f}%")

# ============================================
# PÁGINA 7: DATOS
# ============================================
elif pagina == "📄 Datos":
    st.title("📄 Exploración de Datos")
    
    tab1, tab2, tab3 = st.tabs(["📊 Dataset", "📈 Estadísticas", "🔍 Visualizaciones"])
    
    # Tab 1: Dataset
    with tab1:
        st.markdown("### 📋 Vista del Dataset")
        
        st.dataframe(
            df.style.format({
                'precio': '${:.2f}',
                'pax_pago': '{:,.0f}',
                'ln_q': '{:.4f}',
                'ln_p': '{:.4f}'
            }),
            use_container_width=True,
            height=400
        )
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Observaciones", len(df))
        
        with col2:
            st.metric("📅 Período", f"{df['fecha'].min().year} - {df['fecha'].max().year}")
        
        with col3:
            st.metric("💰 Precio Promedio", f"${df['precio'].mean():.2f}")
        
        with col4:
            st.metric("👥 Demanda Promedio", f"{int(df['pax_pago'].mean()):,}")
    
    # Tab 2: Estadísticas
    with tab2:
        st.markdown("### 📊 Estadísticas Descriptivas")
        
        st.dataframe(
            df[['precio', 'pax_pago', 'ln_q', 'ln_p']].describe().T.style.format('{:.4f}'),
            use_container_width=True
        )
        
        # Correlaciones
        st.markdown("### 🔗 Matriz de Correlación")
        
        corr_matrix = df[['precio', 'pax_pago', 'ln_q', 'ln_p', 't']].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(3),
            texttemplate='%{text}',
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title='Matriz de Correlación',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Visualizaciones
    with tab3:
        st.markdown("### 📈 Evolución Temporal")
        
        # Precio y demanda en el tiempo
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['fecha'],
            y=df['precio'],
            mode='lines+markers',
            name='Precio',
            yaxis='y',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=df['fecha'],
            y=df['pax_pago']/1000000,
            mode='lines+markers',
            name='Pasajeros (millones)',
            yaxis='y2',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title='Evolución de Precio y Demanda (2014-2019)',
            xaxis_title='Fecha',
            yaxis=dict(title='Precio ($)', side='left'),
            yaxis2=dict(title='Pasajeros (millones)', overlaying='y', side='right'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribuciones
        st.markdown("### 📊 Distribuciones")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure(data=[go.Histogram(x=df['ln_p'], nbinsx=30)])
            fig.update_layout(title='Distribución de ln(Precio)', xaxis_title='ln_p', height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure(data=[go.Histogram(x=df['ln_q'], nbinsx=30)])
            fig.update_layout(title='Distribución de ln(Demanda)', xaxis_title='ln_q', height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot
        st.markdown("### 🎯 Relación Precio-Demanda")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['ln_p'],
            y=df['ln_q'],
            mode='markers',
            marker=dict(
                size=8,
                color=df['t'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Tiempo")
            ),
            text=df['fecha'].dt.strftime('%Y-%m'),
            hovertemplate='<b>%{text}</b><br>ln_p: %{x:.3f}<br>ln_q: %{y:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='ln(Precio) vs ln(Demanda)',
            xaxis_title='ln(Precio)',
            yaxis_title='ln(Demanda)',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p><b>Trabajo Práctico - Metodología de la Investigación- Universidad del Gran Rosario</b></p>
    <p>Análisis de Elasticidad Precio de la Demanda del Subte</p>
    <p>📊 Dataset: SBASE 2014-2019</p>
    <p>Desarrollado por:</p
    <p>Federico Ford</p>
    <p>Mariana Veccio </p>
    <p>Gastón Montenegro </p>
    <p>Pedro Chincolla </p>
    <p>Samuel Kanneman </p>
</div>
""", unsafe_allow_html=True)
