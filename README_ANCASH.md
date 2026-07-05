# Modelo Predictivo de Conflictos Sociales — Región Áncash (ANTAMINA)

Documentación técnica del sistema **Track A regional de Áncash**: un modelo de
alerta temprana que estima, por Unidad de Gestión Territorial (UGT) de
ANTAMINA, la probabilidad de que ocurra una protesta nueva en los próximos
30 días.

**Alcance:** exclusivamente Áncash y las 4 UGTs de ANTAMINA. Ver `CLAUDE.md`
para el contexto de negocio y las decisiones de diseño que no deben
cambiarse sin justificación.

---

## Índice rápido

1. [El problema que resuelve](#1-el-problema-que-resuelve)
2. [Para el desarrollador que llega nuevo](#2-para-el-desarrollador-que-llega-nuevo)
3. [Unidad de análisis: ¿por qué UGT?](#3-unidad-de-análisis-por-qué-ugt-y-no-distrito-ni-departamento)
4. [Las dos capas del sistema (no confundir)](#4-las-dos-capas-del-sistema-no-confundir)
5. [Capa 1 — Índice de actividad observada](#5-capa-1--índice-de-actividad-observada)
6. [Capa 2 — El modelo predictivo](#6-capa-2--el-modelo-predictivo)
7. [Scoring semanal](#7-scoring-semanal-poner-el-modelo-a-trabajar)
8. [Dashboard](#8-dashboard)
9. [Cómo reproducir todo desde cero](#9-cómo-reproducir-todo-desde-cero)
10. [Cómo agregar una feature nueva](#10-cómo-agregar-una-feature-nueva)
11. [Fuentes evaluadas y descartadas](#11-fuentes-evaluadas-y-descartadas)
12. [Limitaciones honestas](#12-limitaciones-honestas-léase-antes-de-presentar-el-modelo-a-alguien)
13. [Trabajo pendiente / roadmap](#13-trabajo-pendiente--roadmap)
14. [Glosario rápido](#14-glosario-rápido)

---

## 1. El problema que resuelve

Antamina necesita saber, **con anticipación**, qué tan probable es que ocurra
una protesta nueva en cada una de sus zonas operativas durante el próximo mes.
No es lo mismo que "monitorear lo que ya está pasando" (eso ya existe en el
dashboard de actividad observada): el objetivo aquí es **pronosticar**.

Esto se modela como una **clasificación binaria supervisada con componente
temporal**: para una UGT y una semana dadas, predecir si habrá o no una
protesta nueva en los próximos 30 días.

---

## 2. Para el desarrollador que llega nuevo

### 2.1 Entorno

```bash
# Python 3.11+ recomendado. Crear y activar venv:
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

### 2.2 Archivos de datos necesarios

El proyecto trabaja 100% local. Los archivos de datos **no están en git**
(`.gitignore`). Un colaborador nuevo necesita obtenerlos del equipo:

| Archivo | Ubicación | Cómo obtener |
|---|---|---|
| `dataIncidentes.xlsx` | `src/data/` | BD interna de incidentes geolocalizados |
| `dataFormulario.xlsx` | `src/data/` | Log de monitoreo de noticias de Antamina |
| `dataProtestas.xlsx` | `src/data/` | Detalle de protestas con nivel de impacto |
| `dataHistoricaProtestas.xlsx` | `src/data/` | Historial de prensa 2001–2025 |
| `2025/` (carpeta) | `src/data/` | Reportes Word/PowerPoint de Antamina 2025 |
| `2026/` (carpeta) | `src/data/` | Reportes Word/PowerPoint de Antamina 2026 |
| `defensoria_historico.csv` | `data/raw/defensoria/` | Generar con `scraper_defensoria.py` (requiere conexión) |

Los archivos intermedios (`data/interim/*.parquet`, `data/processed/*.parquet`)
se regeneran ejecutando el pipeline (sección 9). El modelo entrenado
(`models/modelo_v1_track_A_ancash.pkl`) también se regenera con `train_ancash`.

### 2.3 Flujo de trabajo habitual

```
[datos fuente xlsx/docx/pptx]
        │
        ▼
[loaders: loader_*.py]  →  data/interim/*.parquet
        │
        ▼
[build_ancash.py]  →  data/processed/master_ancash_ugt.parquet  (444 filas)
        │
        ▼
[train_ancash.py]  →  models/modelo_v1_track_A_ancash.pkl
        │
        ▼
[recalibrar_umbral.py]  →  data/processed/calibracion_umbral_ancash.csv
        │
        ▼
[score_ancash.py]  →  predicción por UGT en consola
[dashboard_ancash_predictivo.py]  →  data/processed/dashboard_ancash_predictivo.html
```

### 2.4 Comandos del día a día

```bash
# Predicción de la semana actual (consola):
python -m src.scoring.score_ancash

# Dashboard interactivo (abre el navegador):
python -m src.scoring.dashboard_ancash_predictivo

# Reconstruir todo desde cero: ver sección 9
```

### 2.5 Mapa de archivos clave

```
src/
├── data/
│   ├── ancash_datos.py          ← JERARQUIA (mapa UGT→distrito), cálculo del índice observado
│   ├── loader_calendario.py     ← feriados, elecciones, fechas críticas
│   ├── loader_inei.py           ← tasa de pobreza anual (Áncash)
│   ├── scraper_defensoria.py    ← escalamiento Defensoría (requiere conexión)
│   ├── loader_oefa.py           ← denuncias ambientales OEFA/SINADA
│   ├── loader_reportes.py       ← parser de reportes Word/PowerPoint de Antamina
│   └── loader_historica.py      ← protestas históricas de prensa (2001–2025)
├── dataset/
│   └── build_ancash.py          ← tabla maestra UGT × semana, todas las features y labels
├── models/
│   ├── train_ancash.py          ← entrenamiento + go/no-go + serialización del modelo
│   └── recalibrar_umbral.py     ← calibración de UMBRAL_ALTO/MEDIO con predicciones OOF
└── scoring/
    ├── ancash_datos.py          ← índice de actividad observada (NO es predicción)
    ├── score_ancash.py          ← scoring semanal con el modelo entrenado
    ├── dashboard_ancash.py      ← dashboard HTML con Plotly (versión técnica)
    ├── dashboard_ancash_predictivo.py  ← dashboard con plantilla de diseño (presentación cliente)
    └── assets/ancash_dashboard/ ← template.html, support.js, logo

data/
├── interim/                     ← outputs de los loaders (gitignored)
├── processed/                   ← dataset maestro, resultados, dashboard HTML (gitignored)
└── raw/defensoria/              ← CSV de Defensoría (gitignored)

models/
└── modelo_v1_track_A_ancash.pkl ← modelo serializado con joblib (gitignored)
```

---

## 3. Unidad de análisis: ¿por qué UGT y no distrito ni departamento?

Se evaluaron tres niveles de granularidad espacial:

| Nivel | Problema |
|---|---|
| **Departamento** (Áncash completo) | Label casi trivial: protestas urbanas de Chimbote/Huaraz no relacionadas con Antamina "contaminan" la etiqueta, subiéndola artificialmente. |
| **Distrito** (18 distritos de interés) | Demasiado disperso: solo 9 de los 18 distritos tuvieron alguna protesta registrada. Ruido, no señal. |
| **UGT** (Unidad de Gestión Territorial) ✅ | Agrupación operativa real de Antamina. Suficiente densidad de eventos. Directamente accionable para el cliente. |

Las 4 UGTs y su jerarquía territorial (`src/scoring/ancash_datos.py`, `JERARQUIA`):

| UGT | Provincias | Distritos |
|---|---|---|
| Mina San Marcos | Huari | San Marcos, Chavín de Huántar, Huachis, San Pedro de Chana |
| Huallanca | Bolognesi | Huallanca, Aquía, Chiquián |
| Valle Fortaleza | Bolognesi, Recuay, Barranca-Lima | Cajacay, Antonio Raimondi, Colquioc, Huayllacayán, Catac, Pampas Chico, Marca, Llacllín, Pararín, Paramonga |
| Huarmey | Huarmey | Huarmey |

**UGT no respeta límites provinciales estrictos** (Valle Fortaleza cruza tres
provincias) — deliberado, porque la unidad de interés es la operación, no la
división político-administrativa.

---

## 4. Las dos capas del sistema (no confundir)

El dashboard combina dos cosas conceptualmente distintas:

| Capa | Archivo | Qué hace | Naturaleza |
|---|---|---|---|
| **Índice de actividad observada** | `ancash_datos.py` | Mide cuánta conflictividad reciente hay por distrito (fotografía del pasado) | Fórmula determinística, sin ML |
| **Modelo predictivo Track A** | `build_ancash.py` → `train_ancash.py` → `score_ancash.py` | Estima la probabilidad de una protesta nueva en los próximos 30 días | Clasificación supervisada walk-forward |

Esta sección cubre primero la capa 1 (más simple) y luego la capa 2 (el
modelo real).

---

## 5. Capa 1 — Índice de actividad observada

Archivo: `src/scoring/ancash_datos.py`. No usa machine learning, es una
fórmula determinística.

### 5.1 Fuentes

Dos fuentes, combinadas en `_cargar_incidentes_raw()`:

- **`src/data/dataIncidentes.xlsx`** — BD interna con eventos geolocalizados:
  fecha, departamento, distrito, motivo, título. Cobertura: agosto 2023 en adelante.
- **`src/data/dataFormulario.xlsx`** — log de monitoreo de noticias específico
  de Antamina (~8,400 filas, categoría "Protestas, Paros y Bloqueos"). Es un
  log de NOTICIAS — un mismo evento real puede tener varias filas (distintos
  artículos del mismo hecho) — se deduplica a 1 fila por (distrito, fecha).
  Solo se agregan combinaciones (distrito, fecha) que `dataIncidentes` no tiene,
  para no contar el mismo evento dos veces.

Ambas se filtran a `DEPTOS_VALIDOS = {"ANCASH", "LIMA"}` y se normaliza texto
(`_norm`: minúsculas + sin tildes) para machear nombres de distrito de forma
robusta.

**Hueco conocido:** ninguna fuente actual tiene eventos de Paramonga (el
distrito de Valle Fortaleza en provincia "Barranca - Lima").

### 5.2 Categorización del motivo

Cada fila tiene un campo `MOTIVO` libre. Se mapea a categoría via
`MOTIVO_A_CATEGORIA`:

- **PROTESTA**: bloqueo, reclamo, reunión, evento social, incidente fundo/mina.
- **POLITICA**: movimiento político, político-electoral.
- **VIOLENCIA**: homicidio, sicariato, secuestro, extorsión, robo.
- **OTRO**: cualquier motivo no mapeado (se descarta del índice).

### 5.3 Fórmula del índice (decaimiento exponencial + normalización)

Para cada evento, el peso es:

```
peso_evento = peso_categoría × exp(-días_desde_el_evento / 180)
```

- `peso_categoría` (constante `PESOS`): PROTESTA=1.0, POLITICA=0.8, VIOLENCIA=0.45.
- `180` días (`TAU_DIAS`): un evento de hace 180 días pesa `e⁻¹ ≈ 37%` de lo
  que pesaría si fuera hoy.

Normalización a escala 0-100:

```
score = 100 × √(min(score_crudo / percentil_95, 1))
```

- **Percentil 95** como referencia: evita que un distrito con actividad extrema
  aplaste la escala de todos los demás.
- **Raíz cuadrada**: comprime el rango, dando más resolución en la zona media.

### 5.4 Estados y acción recomendada

```
score >= 70 → Alto crítico (rojo)    → Mesa de diálogo urgente
score >= 55 → Alto (naranja)         → Intervención preventiva
score >= 35 → Medio (ámbar)          → Monitoreo cercano
score <  35 → Bajo (verde)           → Vigilancia rutinaria
```

Esto es **descriptivo**, no es la salida del modelo de ML.

---

## 6. Capa 2 — El modelo predictivo

### 6.1 Variable objetivo (target)

`y_30`: binaria, **"¿hubo al menos una protesta nueva en esta UGT durante los
30 días siguientes a esta semana?"**

También existen `y_14` y `y_60` en el dataset (misma lógica, otra ventana).
El modelo desplegado usa `y_30`.

El label sale de la **misma BD de incidentes**, filtrada solo a categoría
`PROTESTA` (VIOLENCIA es feature, no target).

### 6.2 Anti-fuga del label

Código clave (`src/dataset/build_ancash.py`, función `_labels`):

```python
ini = sv + np.timedelta64(1, "D")     # empieza el día después de la semana
fin = sv + np.timedelta64(h, "D")     # termina h días después
hay = (searchsorted(eventos, ini) < searchsorted(eventos, fin))
hay[fin > fecha_corte] = NaN          # ventana fuera del dato disponible → descartar
```

Dos reglas de anti-fuga:
1. El label mira **estrictamente hacia adelante** (nunca incluye el presente).
2. Si la ventana de 30 días supera `FECHA_CORTE`, la fila se descarta — no se
   infiere "no hubo protesta" de la ausencia de datos futuros.

### 6.3 Features (19 variables activas)

| Grupo | Variable | Qué mide |
|---|---|---|
| **Incidentes (autoregresivas)** | `inc_prot_1w` | protestas en la última semana |
| | `inc_prot_4w` | protestas en las últimas 4 semanas |
| | `inc_viol_1w`, `inc_viol_4w` | igual pero violencia |
| | `delta_prot` | aceleración: semana actual vs. promedio de las 4 previas |
| | `racha_prot` | semanas consecutivas con al menos 1 protesta |
| | `dias_desde_ultima_prot` | recencia de la última protesta conocida |
| **Calendario** | `n_feriados`, `es_semana_electoral`, `dias_hasta_eleccion`, `es_fecha_critica`, `mes`, `trimestre`, `semana_iso` | estructura temporal y política |
| **Contexto departamental** | `tasa_pobreza` | INEI, valor anual "as-of" (solo Áncash) |
| | `def_escalamiento_ancash` | Defensoría del Pueblo, Áncash, con 2 meses de rezago anti-fuga |
| | `oefa_denuncias_mineria_12m` | Denuncias ambientales OEFA/SINADA (actividad minería), ventana móvil 12 meses — **señal de contexto acumulado, no validada como predictor de corto plazo** (ver sección 11.1) |
| **Reportes Antamina** | `rep_antamina_neg_4w` | párrafos de reportes Word/PowerPoint que mencionan Antamina explícitamente CON tono de tensión (últimas 4 semanas) |
| | `rep_compromiso_4w` | párrafos con mención de mesa de diálogo/compromiso/acuerdo (últimas 4 semanas) |

**Features probadas y descartadas** (código en `build_ancash.py`, no en `FEATURES`):

| Variable | Fuente | Resultado |
|---|---|---|
| `prot_impacto_ultimo` | `dataProtestas.xlsx` (severidad) | PR-AUC 0.780 → 0.761. Degradó el modelo. |
| `hist_prot_antamina_acum`, `hist_prot_antamina_5y` | `dataHistoricaProtestas.xlsx` (prensa 2001–2025) | PR-AUC 0.8004 → 0.7962. Son constantes por UGT en la ventana 2024–2026 (todos los eventos históricos son pre-2024). Sin señal real. |

**Notas importantes sobre features:**
- `tasa_pobreza` y `def_escalamiento_ancash` son **departamentales** — la misma
  valor para las 4 UGTs de cada semana. Contexto de fondo.
- `rep_antamina_neg_4w` y `rep_compromiso_4w` son específicas por UGT (buscan
  menciones de los distritos exactos de cada UGT en el texto de los reportes).
  Solo tienen cobertura desde **enero 2025** — 2024 queda en 0.
- El rezago de 2 meses en `def_escalamiento_ancash` no es arbitrario: la
  Defensoría publica su reporte del mes M en M+1. Solo se usa el último reporte
  efectivamente publicado a cada fecha de corte.

### 6.4 Dataset maestro

```python
UGTS        = ["Mina San Marcos", "Huallanca", "Valle Fortaleza", "Huarmey"]
FECHA_INICIO = "2024-01-01"
FECHA_CORTE  = "2026-04-13"
```

Producto cartesiano UGT × semana (incluye semanas sin eventos — el modelo
necesita ver tanto positivos como negativos).

Resultado: **444 filas** (4 UGTs × ~111 semanas) en
`data/processed/master_ancash_ugt.parquet`.

Balance `y_30`: **43.9% positivos (195/444)**. Por UGT: Mina San Marcos 59%,
Huarmey 50%, Valle Fortaleza 33%, Huallanca 32%.

### 6.5 Algoritmo y pipeline

Archivo: `src/models/train_ancash.py`. Dos candidatos evaluados:

```python
"logistic_regression": Pipeline([
    SimpleImputer(strategy="median", keep_empty_features=True),
    StandardScaler(),
    LogisticRegression(class_weight="balanced", max_iter=1000),
]),
"random_forest": Pipeline([
    SimpleImputer(strategy="median", keep_empty_features=True),
    RandomForestClassifier(n_estimators=200, max_depth=4, class_weight="balanced"),
]),
```

- **`SimpleImputer(keep_empty_features=True)`**: rellena NaN con la mediana.
  `keep_empty_features` evita que columnas sin valores en un fold sean
  eliminadas silenciosamente (error real durante el desarrollo).
- **`class_weight="balanced"`**: penaliza más los errores sobre la clase
  minoritaria, evita que el modelo aprenda a "siempre decir no".
- **`max_depth=4`** en Random Forest: con 444 filas, mayor profundidad
  provocaría overfitting.

**Modelo desplegado: `logistic_regression`** — PR-AUC **0.8004**, std ±0.167
(más estable que Random Forest: 0.792 ± 0.187).

El `.pkl` guarda dos campos distintos de PR-AUC:
- `pr_auc_cv`: PR-AUC del modelo que se serializa (logistic_regression)
- `pr_auc_mejor_candidato`: PR-AUC del mejor candidato en la evaluación

Esta distinción importa porque a veces random_forest gana en PR-AUC pero se
descarta por inestabilidad — los dos campos diferirían. Actualmente coinciden
porque logistic_regression es el mejor en ambas dimensiones.

### 6.6 Validación: walk-forward, nunca k-fold aleatorio

```python
TimeSeriesSplit(n_splits=4)
```

Regla de oro del proyecto. Un k-fold aleatorio mezclaría semanas del
"futuro" en el entrenamiento → métricas infladas artificialmente. Con
`TimeSeriesSplit`, cada fold de prueba es **posterior** a su fold de
entrenamiento, simulando el uso real semana a semana.

De los 4 splits, 3 resultan válidos en cada corrida (se descarta un fold
si alguna de sus partes tiene una sola clase).

### 6.7 Métricas: PR-AUC y recall a precisión fija

- **Por qué no accuracy**: con 56% de negativos, "siempre decir no" ya da 56%
  de exactitud sin aprender nada útil.
- **PR-AUC**: resume la discriminación sobre todos los umbrales posibles, sin
  la distorsión del desbalance.
- **Recall@P50**: de todas las protestas reales, qué porcentaje captura el
  modelo cuando se exige ≥50% de precisión.

### 6.8 Punto de control go/no-go

```
Mejor baseline:    siempre_negativo = 0.5540
Mejor candidato:   logistic_regression = 0.8004
Diferencia:        +0.2464   (umbral GO: +0.05)
VEREDICTO:         GO ✅
```

El modelo debe superar al mejor baseline en al menos +0.05 de PR-AUC o no se
despliega. Si el resultado es NO-GO, no hay modelo en `models/`.

### 6.9 Umbral de alerta calibrado

Archivo: `src/models/recalibrar_umbral.py`. Calibrado el 2026-06-30 con
predicciones **out-of-fold** (OOF) del walk-forward — no se toca el conjunto
de prueba para elegir el umbral.

| Nivel | Umbral | Precisión OOF | Recall OOF | Tasa de alerta |
|---|---|---|---|---|
| **ALTO** | ≥ 80% | 75% | 69% | 59% |
| **MEDIO** | 44–79% | — | — | — |
| **BAJO** | < 44% | — | — | — |

- **ALTO ≥ 80%**: umbral elegido manualmente (el criterio automático de
  max-F1 daba 14% → 71% de tasa de alerta, operacionalmente inútil con el
  modelo anterior; con el modelo actual auto-elige 77%, razonable).
- **MEDIO ≥ 44%**: tasa histórica de protesta (43.9%) como referencia de
  "más riesgo que el promedio". Un valor en MEDIO significa que el modelo
  ve señales sobre la media, aunque no con suficiente certeza para ALTO.
- **BAJO < 44%**: el modelo estima riesgo por debajo del promedio histórico.

Si el modelo se reentrena (nuevas features, más datos), **hay que recalibrar**:
```bash
python -m src.models.recalibrar_umbral
```

### 6.10 Modelo final guardado

```python
joblib.dump({
    "model":                  modelo_final,         # entrenado sobre 444 filas completas
    "feature_cols":           FEATURES,             # 19 variables, en el orden exacto
    "track": "A", "region": "Áncash", "unidad": "UGT × semana",
    "target":                 "y_30",
    "pr_auc_cv":              0.8004,               # PR-AUC del modelo guardado
    "pr_auc_mejor_candidato": 0.8004,
    "pr_auc_baseline":        0.5540,
    "modelo_tipo":            "logistic_regression",
    "trained_on":             "<fecha>",
}, "models/modelo_v1_track_A_ancash.pkl")
```

El modelo final se reentrena sobre **todos** los datos disponibles (no solo
un fold) una vez que pasó el go/no-go — los folds sirven para medir calidad,
no para producir el modelo desplegado.

---

## 7. Scoring semanal (poner el modelo a trabajar)

Archivo: `src/scoring/score_ancash.py`. Reutiliza las **mismas funciones** de
`build_ancash.py` para construir features — no duplica lógica, garantizando
que las features de "hoy" se calculen exactamente igual que en entrenamiento.

```python
def predecir(lunes_actual=None):
    paquete = joblib.load(MODELO_PATH)
    df = calcular_features_actuales(lunes_actual)  # mismas funciones que el dataset
    X = df[paquete["feature_cols"]]
    df["probabilidad"] = paquete["model"].predict_proba(X)[:, 1]
    ...
```

Salida típica:

```
Predicción de protesta · próximos 30 días · semana 2026-07-07 (Áncash, por UGT):

  Huarmey            99.9%   ALTO
  Valle Fortaleza    78.6%   MEDIO
  Mina San Marcos    78.1%   MEDIO
  Huallanca          57.2%   MEDIO
```

---

## 8. Dashboard

### 8.1 Dashboard técnico: `dashboard_ancash.py`

Genera `data/processed/dashboard_ancash.html` autocontenido con Plotly via CDN.

Combina:
- **Predicción del modelo**: 4 tarjetas por UGT con probabilidad y badge PR-AUC.
- **Actividad observada**: mapa de distritos, serie mensual, composición por
  tipo de conflicto.
- **KPI "Compromiso de las partes"**: muestra datos reales de `rep_compromiso_4w`
  por UGT (menciones de acuerdos/mesas activas en las últimas 4 semanas,
  desde los reportes Word/PowerPoint de Antamina). Si no hay modelo activo,
  muestra "PENDIENTE DE FUENTE" como fallback.
- Filtros jerárquicos UGT → Provincia → Distrito implementados en JavaScript
  puro, sin backend.

### 8.2 Dashboard de presentación: `dashboard_ancash_predictivo.py`

Usa una plantilla visual diseñada (`src/scoring/assets/ancash_dashboard/`)
con los mismos datos reales inyectados via `window.__DATA__` en el HTML.
**Es el dashboard recomendado para presentar al cliente.**

```bash
python -m src.scoring.dashboard_ancash_predictivo
# → data/processed/dashboard_ancash_predictivo.html
```

---

## 9. Cómo reproducir todo desde cero

Requiere que los archivos de datos fuente estén en `src/data/` (ver sección 2.2).

**Paso 1 — Generar insumos intermedios** (una sola vez, o cuando cambien los datos fuente):

```bash
python -m src.data.loader_calendario     # → data/interim/calendario.parquet
python -m src.data.loader_inei           # → data/interim/inei_pobreza.parquet
python -m src.data.scraper_defensoria    # → data/raw/defensoria/ (requiere conexión)
python -m src.data.loader_oefa           # → data/interim/oefa_denuncias_ancash.parquet (requiere conexión)
python -m src.data.loader_reportes       # → data/interim/reportes_antamina_parrafos.parquet
python -m src.data.loader_historica      # → data/interim/historica_protestas_ugt.parquet
```

**Paso 2 — Pipeline de Áncash** (ejecutar en este orden):

```bash
python -m src.dataset.build_ancash       # → data/processed/master_ancash_ugt.parquet (444 filas)
python -m src.models.train_ancash        # → models/modelo_v1_track_A_ancash.pkl (si pasa go/no-go)
python -m src.models.recalibrar_umbral   # → data/processed/calibracion_umbral_ancash.csv
python -m src.scoring.score_ancash       # predicción de la semana actual en consola
python -m src.scoring.dashboard_ancash_predictivo  # dashboard HTML
```

**Operación semanal** (cada viernes):

```bash
python -m src.scoring.score_ancash
python -m src.scoring.dashboard_ancash_predictivo
```

Anotar manualmente: ¿hubo conflicto/protesta real esta semana en cada UGT?
Con varias semanas acumuladas se puede calcular el recall operativo real.

---

## 10. Cómo agregar una feature nueva

Este es el flujo estándar. Seguirlo evita errores de fuga de datos.

### Paso 1 — Crear el loader (si la fuente es nueva)

Crear `src/data/loader_<nombre>.py` con este patrón:

```python
SALIDA = Path("data/interim") / "<nombre>.parquet"

def cargar() -> pd.DataFrame:
    # leer fuente, limpiar, devolver DataFrame con columnas conocidas
    ...

def main():
    df = cargar()
    Path("data/interim").mkdir(parents=True, exist_ok=True)
    df.to_parquet(SALIDA, index=False)

if __name__ == "__main__":
    main()
```

### Paso 2 — Agregar `_join_<nombre>()` en `build_ancash.py`

Patrón con anti-fuga usando `np.searchsorted` (ver `_join_oefa` o
`_join_reportes` como referencia):

```python
def _join_nueva_feature(master: pd.DataFrame) -> pd.DataFrame:
    ruta = INTERIM_DIR / "<nombre>.parquet"
    if not ruta.exists():
        warnings.warn("No se encontró <nombre>.parquet. Corre loader_<nombre> primero.")
        master["nombre_feature"] = np.nan
        return master

    df = pd.read_parquet(ruta)
    # IMPORTANTE: solo usar datos con fecha < semana_inicio (anti-fuga)
    master = master.sort_values(["ugt", "semana_inicio"]).reset_index(drop=True)
    valores = np.zeros(len(master), dtype=float)
    for ugt in UGTS:
        ev = np.sort(df[df["ugt"] == ugt]["fecha"].values.astype("datetime64[ns]"))
        idx = master[master["ugt"] == ugt].index
        sv = master.loc[idx, "semana_inicio"].values.astype("datetime64[ns]")
        ventana_ini = sv - np.timedelta64(N_DIAS, "D")
        valores[idx] = np.searchsorted(ev, sv, "right") - np.searchsorted(ev, ventana_ini, "left")
    master["nombre_feature"] = valores
    return master
```

### Paso 3 — Llamar a la función en `construir()` y en `calcular_features_actuales()`

En `build_ancash.py`, función `construir()`:
```python
master = _join_nueva_feature(master)
```

En `score_ancash.py`, función `calcular_features_actuales()` e importar:
```python
from src.dataset.build_ancash import (..., _join_nueva_feature)
# ...
master = _join_nueva_feature(master)
```

### Paso 4 — Agregar a `FEATURES` en `train_ancash.py`

```python
FEATURES = [
    ...,
    "nombre_feature",
]
```

### Paso 5 — Rebuilding y validación

```bash
python -m src.dataset.build_ancash    # verifica que la feature aparece sin NaN inesperados
python -m src.models.train_ancash     # compara PR-AUC con y sin la feature
python -m src.models.recalibrar_umbral
```

Si el PR-AUC **baja o no cambia significativamente** (< +0.01), la feature
probablemente agrega ruido. Sacarla de `FEATURES` y documentarla como
"probada y descartada" en el comentario de `train_ancash.py` (igual que
`prot_impacto_ultimo` e `hist_prot_antamina_acum`).

---

## 11. Fuentes evaluadas y descartadas

### 11.1 OEFA/SINADA — denuncias ambientales

Se evaluó si las denuncias ambientales públicas de OEFA (actividad "minería")
podían usarse como alerta temprana. Dataset público, descarga directa desde
`datosabiertos.oefa.gob.pe`, 95 denuncias en los 18 distritos desde 2019.

**Resultado: inconcluso.** El archivo OEFA tiene un rezago de publicación de
~12+ meses. Solo 9 protestas reales caen dentro de la ventana donde ambas
fuentes se superponen — muestra demasiado pequeña. Con esos 9 casos: a 30/60
días las protestas tuvieron *menos* denuncias previas que el promedio; a 90
días, levemente más (4 casos). Sin evidencia sólida en ningún sentido.

**Decisión:** integrar como feature de contexto acumulado (`oefa_denuncias_mineria_12m`,
ventana móvil de 12 meses), no como alerta temprana. Si en el futuro se
consigue una fuente OEFA más actualizada, vale la pena rehacer el análisis.

### 11.2 `dataHistoricaProtestas.xlsx` — historial de prensa 2001–2025

Registro periodístico (La República, Expreso, etc.) de 481 protestas en
Áncash desde el inicio de operaciones de Antamina (2001). Atributos ricos:
tipo de acción, adversario, reclamo, duración, participantes, heridos.

**Cobertura en las 4 UGTs:** 60 eventos totales, 34 dirigidos explícitamente
a Antamina (adversario = "Empresas Privadas" o mención directa).

**Problema:** `dataIncidentes.xlsx` solo cubre desde agosto 2023. Extender la
ventana de entrenamiento al período 2001–2023 dejaría las features
autoregresivas (`inc_prot_*`, `racha_prot`, etc.) todas en NaN, que el
SimpleImputer rellenaría con cero — ruido, no señal.

**Resultado:** las features `hist_prot_antamina_acum` y `hist_prot_antamina_5y`
derivadas de esta fuente se probaron y descartaron: PR-AUC 0.8004 → 0.7962.
Son casi constantes por UGT en la ventana 2024–2026 (todos los eventos pre-2024
tienen el mismo valor para todas las filas 2024).

**Código disponible:** `src/data/loader_historica.py` y `_join_historica()` en
`build_ancash.py` están implementados. Cuando `dataIncidentes` tenga cobertura
desde 2018–2019, se puede activar `_join_historica()` junto con una ventana de
entrenamiento extendida y estas features volverán a tener sentido.

### 11.3 `dataProtestas.xlsx` — nivel de impacto/severidad

Dataset con nivel de impacto numérico (70–500) por evento de protesta.
Feature `prot_impacto_ultimo` (severidad de la última protesta conocida,
forward-fill por UGT): PR-AUC 0.780 → 0.761. Descartada.

---

## 12. Limitaciones honestas (léase antes de presentar el modelo a alguien)

- **Dataset pequeño:** 444 filas, 195 positivos, solo 3-4 folds de validación
  válidos. El GO es real, pero con margen de certeza estadística bajo. La BD
  de incidentes arrancó en agosto 2023 — el modelo mejorará naturalmente
  conforme pasen más semanas.
- **`dataFormulario.xlsx` es un log de noticias, no de eventos deduplicados:**
  dos protestas distintas el mismo día en el mismo distrito cuentan como una
  sola (deduplicación por fecha+distrito).
- **Hueco de cobertura: Paramonga** (Valle Fortaleza, único distrito en
  provincia "Barranca - Lima") no tiene eventos en ninguna fuente actual.
- **`tasa_pobreza` no es reproducible desde cero:** el Excel fuente original
  se perdió al actualizar los archivos de datos. Funciona via parquet cacheado
  (`data/interim/inei_pobreza.parquet`), pero no se puede regenerar sin el
  Excel original. Además es anual y casi constante: poca señal real.
- **`def_escalamiento_ancash` y `tasa_pobreza` son departamentales:** misma
  feature para las 4 UGTs de cada semana.
- **`oefa_denuncias_mineria_12m` no validada como predictor de corto plazo**
  (ver sección 11.1) — refleja historia de hasta 12+ meses atrás.
- **Features de reportes Antamina usan palabras clave**, no comprensión
  semántica. "Coordina" cuenta como "compromiso" aunque el contexto sea otro.
  Solo tienen cobertura desde enero 2025; 2024 queda en 0.
- **Umbral de alerta específico del modelo actual:** si cambia el feature set
  o hay más datos, hay que recalibrar con `recalibrar_umbral.py`. El umbral
  calibrado (ALTO≥80%, MEDIO≥44%) solo es válido para el modelo actual.
- **No incluye aún:** histórico de alertas propias, precio del cobre
  (commodities, zona minera), intensidad mediática (GDELT).

---

## 13. Trabajo pendiente / roadmap

| Prioridad | Tarea | Bloqueante |
|---|---|---|
| Alta | Extender ventana de entrenamiento hacia atrás | Necesita datos de incidentes desde 2018–2021 |
| Alta | Añadir precio del cobre como feature (zona minera) | Fuente a definir |
| Media | Reactivar `hist_prot_antamina_*` cuando haya datos históricos | Depende del punto anterior |
| Media | Integrar alertas propias del sistema actual como feature | Acceso a BD histórica |
| Media | Validación operativa: anotar semanal si hubo protesta real por UGT | Proceso manual, acumulable |
| Baja | GDELT: intensidad mediática como señal de contexto | API disponible, requiere investigación |
| Baja | Resolver cobertura de Paramonga | Nueva fuente de datos de Lima-Barranca |
| Baja | Recuperar Excel fuente de INEI para reproducibilidad completa | Archivar en carpeta del equipo |

---

## 14. Glosario rápido

| Término | Significado en este proyecto |
|---|---|
| **Feature** | Dato de entrada que el modelo usa para razonar (ej. cuántas protestas hubo la semana pasada) |
| **Label / target** | Respuesta correcta que se le enseña al modelo (`y_30`: ¿hubo protesta?) |
| **Leakage (fuga de datos)** | Cuando el modelo "ve" información del futuro durante el entrenamiento, inflando artificialmente sus métricas |
| **Walk-forward / TimeSeriesSplit** | Validación que respeta el orden cronológico: el conjunto de prueba siempre es posterior al de entrenamiento |
| **PR-AUC** | Métrica que resume la discriminación del modelo, robusta a clases desbalanceadas |
| **OOF (out-of-fold)** | Predicciones generadas en los folds de validación (nunca vistos en entrenamiento). Se usan para calibrar umbrales sin tocar datos de prueba |
| **Baseline trivial** | Predictor "tonto" (ej. siempre la tasa histórica) contra el que se compara el modelo para verificar que aprende algo real |
| **class_weight="balanced"** | Ajuste que evita que el modelo aprenda a "siempre decir no" ignorando la clase minoritaria |
| **UGT** | Unidad de Gestión Territorial de Antamina — la unidad espacial de predicción del modelo |
| **Anti-fuga** | Cualquier medida de diseño que garantiza que las features de la semana `t` usen solo datos disponibles hasta el fin de `t` |
