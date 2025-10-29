# 🚀 Guía de Inicio Rápido

## ¿Cuál versión usar?

### 🎯 Comparación Rápida

| Característica | Opción A (Local) | Opción B (Interactiva) |
|----------------|------------------|------------------------|
| **Velocidad de inicio** | ⚡⚡⚡ Muy rápida | ⚡⚡ Media |
| **Facilidad de uso** | ⭐⭐⭐ Requiere colocar archivo | ⭐⭐⭐⭐⭐ Drag & drop |
| **Uso repetido** | ✅ Excelente | ⚠️ Requiere subir cada vez |
| **Cambiar datasets** | ❌ Manual | ✅ Fácil desde interfaz |
| **Ideal para** | Presentaciones, uso frecuente | Demos, pruebas, flexibilidad |

---

## 📦 Opción A: Instalación Local

### Paso 1: Preparar archivos
```bash
# Estructura de carpetas:
tu_carpeta/
├── app_opcion_a_local.py
├── requirements.txt
└── subte_demanda_precio_mensual.parquet  ← ¡Importante!
```

### Paso 2: Instalar dependencias
```bash
pip install -r requirements.txt
```

### Paso 3: Ejecutar
```bash
streamlit run app_opcion_a_local.py
```

### Paso 4: Abrir navegador
Automáticamente se abrirá en `http://localhost:8501`

---

## 🌐 Opción B: Instalación Interactiva

### Paso 1: Instalar dependencias
```bash
pip install -r requirements.txt
```

### Paso 2: Ejecutar
```bash
streamlit run app_opcion_b_interactiva.py
```

### Paso 3: Subir archivo
1. Se abrirá tu navegador en `http://localhost:8501`
2. Verás una pantalla de bienvenida con zona de carga
3. Arrastra o selecciona tu archivo `.parquet`
4. ¡Listo! Los modelos se entrenan automáticamente

---

## ✅ Verificación de Dataset

Tu archivo debe tener estas columnas **obligatorias**:
- ✓ `fecha`
- ✓ `precio`  
- ✓ `pax_pago`
- ✓ `mes`

**Verificar en Python:**
```python
import pandas as pd

df = pd.read_parquet('subte_demanda_precio_mensual.parquet')
print(df.columns)
print(df.head())

# Debe mostrar: fecha, precio, pax_pago, mes
```

---

## 🔧 Solución de Problemas

### Error: "Archivo no encontrado"
**Opción A:** Verifica que el archivo `.parquet` esté en el MISMO directorio que `app_opcion_a_local.py`

```bash
# Verificar estructura:
ls -la
# Deberías ver:
# app_opcion_a_local.py
# subte_demanda_precio_mensual.parquet
```

### Error: "Columnas faltantes"
Tu dataset debe tener al menos: `fecha`, `precio`, `pax_pago`, `mes`

### Error de instalación
```bash
# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### La app no se abre automáticamente
Abre manualmente tu navegador en: `http://localhost:8501`

---

## 📊 Primera Vez - Qué Esperar

### Tiempos de carga:

1. **Carga inicial de datos:** 1-2 segundos
2. **Entrenamiento de modelos:** 5-10 segundos
3. **Validación cruzada:** 10-15 segundos
4. **Total primera carga:** ~20-30 segundos

**✨ Después de la primera carga, todo es instantáneo gracias al caché de Streamlit!**

### Páginas que tardan más:
- 🔍 **Sensibilidad:** ~10 segundos (calcula 50 modelos)
- Las demás son instantáneas

---

## 🎯 Mi Recomendación Personal

### Para tu presentación académica: **Opción A** ✅

**Razones:**
1. ⚡ Carga más rápida en la presentación
2. 🎯 Sin pasos extra frente al profesor
3. 💪 Más profesional y confiable
4. 📊 Te enfocas en los resultados, no en la tecnología

### Para explorar y experimentar: **Opción B** ✅

**Razones:**
1. 🔄 Puedes probar con diferentes datasets
2. 🎨 Muestra el proceso completo
3. 📱 Más amigable para no-técnicos
4. ✨ Interfaz más "wow"

---

## 💡 Tips Pro

### 1. Testea ANTES de presentar
```bash
# Ejecuta ambas versiones para estar seguro
streamlit run app_opcion_a_local.py
```

### 2. Prepara screenshots de respaldo
Por si hay problemas técnicos en la presentación

### 3. Conoce los shortcuts de Streamlit
- `R` = Rerun app
- `C` = Clear cache
- `Ctrl+C` en terminal = Detener app

### 4. Modo presentación
```bash
# Abrir en modo fullscreen
streamlit run app_opcion_a_local.py --server.headless true
```

---

## 📚 Recursos Adicionales

- **Documentación Streamlit:** https://docs.streamlit.io
- **Scikit-learn:** https://scikit-learn.org/stable/
- **Plotly:** https://plotly.com/python/

---

## ✅ Checklist Final

Antes de tu presentación:

- [ ] Archivo `.parquet` en el directorio correcto
- [ ] Dependencias instaladas (`pip list`)
- [ ] App funciona sin errores
- [ ] Probaste todas las páginas
- [ ] Entiendes la interpretación de elasticidad
- [ ] Tienes screenshots de respaldo
- [ ] Batería/conexión eléctrica OK
- [ ] Navegador actualizado

---

## 🎓 Para la Defensa del TP

### Preguntas que te pueden hacer:

1. **¿Por qué usaste Ridge/Lasso además de OLS?**
   - Para evitar overfitting y estabilizar coeficientes
   - Cross-validation determina el α óptimo

2. **¿Qué significa elasticidad de -0.6?**
   - Demanda inelástica: 1% ↑precio → 0.6% ↓demanda
   - Típico de bienes necesarios como transporte público

3. **¿Por qué logaritmos?**
   - Modelo log-log da elasticidad directamente como coeficiente
   - Estabiliza varianza y hace relación lineal

4. **¿Cómo validaste el modelo?**
   - Train/test split temporal (60/40)
   - Time Series Cross-validation (5 folds)
   - Análisis de residuos

---

**¡Éxito en tu presentación! 🎉**

Si tienes dudas, revisa el README_COMPLETO.md para más detalles.
