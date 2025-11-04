# 🚇 Tablero de Elasticidad Precio - Demanda del Subte

## 📋 Descripción

Tablero interactivo desarrollado en **Streamlit** para visualizar y analizar los resultados del modelo de regresión lineal que estima la **elasticidad precio de la demanda** del sistema de subterráneos de Buenos Aires.

---


### 📦 **Carga Local** 
**Archivo:** `app_elasticidad_local.py`

✅ **Ventajas:**
- Carga automática al iniciar
- Más rápida (no necesitas subir el archivo cada vez)
- Ideal para desarrollo y presentaciones

❌ **Requisitos:**
- Debes colocar `subte_demanda_precio_mensual.parquet` en el mismo directorio que la app

**📝 Uso:**
```bash
# 1. Coloca tu archivo parquet junto a la app
# 2. Ejecuta:
streamlit run app_opcion_a_local.py
```

---


## 🎯 Características del Tablero

### Páginas del Tablero:

1. **🏠 Inicio**
   - Resumen ejecutivo con métricas clave
   - Mejor modelo destacado
   - Tabla comparativa de resultados
   - Interpretación de la elasticidad
   - **✨ Usa tus datos reales de SBASE**

2. **📊 Comparación de Modelos**
   - Gráficos Real vs Predicho
   - Análisis de residuos
   - Métricas detalladas (R², RMSE, MAE)
   - Resultados de validación cruzada

3. **🎯 Calculadora de Elasticidad**
   - Simulador interactivo de cambios de precio
   - Cálculo del impacto en la demanda e ingresos
   - Comparación entre los 3 modelos (OLS, Ridge, Lasso)

4. **📈 Series Temporales**
   - Predicciones vs valores reales en el tiempo
   - Errores acumulados por modelo
   - Análisis mensual de performance

5. **🔍 Análisis de Sensibilidad**
   - Curvas de regularización (Ridge y Lasso)
   - Impacto del parámetro α en la elasticidad
   - Relación α vs R²

6. **🎲 Simulador de Escenarios**
   - Configuración personalizada de precio, mes y tendencia
   - Predicciones con los 3 modelos
   - Visualización comparativa de resultados

7. **📄 Exploración de Datos**
   - Vista del dataset completo
   - Estadísticas descriptivas
   - Matriz de correlación
   - Visualizaciones temporales y distribuciones

---


## 📊 Modelos Implementados

Los tres modelos se entrenan **automáticamente** con tus datos:

1. **OLS (Ordinary Least Squares)** - Baseline
   - Regresión lineal clásica sin regularización
   - Sirve como punto de comparación

2. **Ridge Regression** 
   - Regularización L2 con α óptimo seleccionado por CV
   - Reduce la magnitud de los coeficientes
   - Evita overfitting

3. **Lasso Regression**
   - Regularización L1 con α óptimo seleccionado por CV
   - Puede realizar selección de variables
   - Coeficientes pueden llegar a 0

**🔬 Proceso de Entrenamiento:**
- Split 60/40 (train/test) temporal
- Cross-validation con Time Series Split (5 folds)
- Optimización de α con GridSearch CV
- Métricas: R², RMSE, MAE, Elasticidad

---

## 📈 Variables del Modelo

### Requeridas en el Dataset:
- `fecha`: Fecha de la observación
- `precio`: Precio del boleto del subte
- `pax_pago`: Número de pasajeros pagos (demanda)
- `mes`: Número del mes (1-12)

### Transformaciones aplicadas automáticamente:
- `ln_q = log(pax_pago)`: Variable dependiente
- `ln_p = log(precio)`: Logaritmo del precio
- `t`: Tendencia temporal (0, 1, 2, ...)
- Dummies estacionales: meses 2-12 (mes 1 como base)

**📐 Ecuación del modelo:**
```
ln_q = β₀ + β₁·ln_p + β₂·t + Σ(βᵢ·mesᵢ) + ε
```

Donde **β₁ es la elasticidad precio de la demanda** 🎯

---

## 🎓 Interpretación de la Elasticidad

La **elasticidad precio de la demanda** (β₁) mide el cambio porcentual en la cantidad demandada ante un cambio del 1% en el precio.

### 📊 Clasificación:
- **|ε| > 1:** Demanda elástica (alta sensibilidad al precio)
- **|ε| < 1:** Demanda inelástica (baja sensibilidad al precio)
- **|ε| = 1:** Elasticidad unitaria


**🚇 Implicación para el subte:**
Una demanda inelástica significa que la cantidad demandada es poco sensible a cambios en el precio, lo cual es típico del transporte público por ser un bien necesario con pocas alternativas cercanas.

---


## 📚 Tecnologías Utilizadas

- **Streamlit 1.32.0**: Framework para aplicaciones web interactivas
- **Pandas 2.1.4**: Manipulación y análisis de datos
- **NumPy 1.26.3**: Cálculos numéricos
- **Scikit-learn 1.4.0**: Modelos de Machine Learning
- **Plotly 5.18.0**: Visualizaciones interactivas
- **Matplotlib 3.8.2 & Seaborn 0.13.1**: Gráficos estadísticos

---

## 💡 Casos de Uso

1. **📊 Análisis de políticas de precios**
   - Evaluar el impacto de aumentos tarifarios
   - Optimizar la estructura de precios para maximizar ingresos
   - Proyectar demanda bajo diferentes escenarios

2. **📈 Proyecciones de demanda**
   - Estimar pasajeros para diferentes niveles de precio
   - Planificación de capacidad del sistema
   - Análisis de estacionalidad

3. **🔬 Evaluación de modelos**
   - Comparar performance de diferentes técnicas de regularización
   - Validar la estabilidad de los resultados
   - Análisis de sensibilidad a hiperparámetros

4. **🎓 Presentaciones académicas**
   - Visualizar resultados de forma profesional e interactiva
   - Facilitar la comprensión de conceptos econométricos
   - Demostrar aplicaciones prácticas de Machine Learning

---

## 📊 Ejemplo de Dataset

Tu archivo `subte_demanda_precio_mensual.parquet` debe tener esta estructura:

| periodo | mes | pax_pago  | pax_pases_pagos | pax_franq | pax_total  | fecha      | precio |
|---------|-----|-----------|-----------------|-----------|------------|------------|--------|
| 2014    | 1   | 16256557  | 7503            | 817108    | 17080168   | 2014-01-01 | 3.50   |
| 2014    | 2   | 17242544  | 4824            | 820619    | 18067987   | 2014-02-01 | 3.50   |
| 2014    | 3   | 19603417  | 101047          | 985425    | 20689889   | 2014-03-01 | 4.50   |
| ...     | ... | ...       | ...             | ...       | ...        | ...        | ...    |

**Columnas mínimas requeridas:** `fecha`, `mes`, `precio`, `pax_pago`

---

## ❓ FAQ - Preguntas Frecuentes

### ¿Los modelos se entrenan cada vez que abro la app?

Sí, pero solo la primera vez en cada sesión. Streamlit usa **caché** para que las siguientes interacciones sean instantáneas.

### ¿Puedo usar un archivo CSV en lugar de Parquet?

Sí! Solo necesitas modificar la función de carga:

```python
# Cambiar:
df = pd.read_parquet(ruta_archivo)

# Por:
df = pd.read_csv(ruta_archivo)
```

### ¿Cómo exporto las visualizaciones?

Todas las gráficas de Plotly tienen un botón 📷 en la esquina superior derecha para descargar como PNG.

### ¿Puedo agregar más modelos?

¡Sí! Solo agrega tu modelo al diccionario `modelos`:

```python
modelos = {
    'OLS (Baseline)': LinearRegression(),
    'Ridge (α óptimo)': RidgeCV(alphas=[...], cv=5),
    'Lasso (α óptimo)': LassoCV(alphas=[...], cv=5),
    'ElasticNet': ElasticNetCV(alphas=[...], cv=5),  # ← Nuevo modelo
}
```

---

## 🤝 Contribuciones

Este proyecto fue desarrollado como parte de un trabajo práctico para la materia **Metodología de la Investigación**.

---

## 📄 Licencia

Proyecto académico - Uso educativo

---
**Desarrollado para el Trabajo Final Metodología de la Investigación- Universidad del Gran Rosario**

🚇 Análisis de Elasticidad Precio de la Demanda del Subte de Buenos Aires
