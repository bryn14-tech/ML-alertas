 # Proyecto: Modelo Predictivo de Conflictos Sociales — Áncash (ANTAMINA)

Sistema de alerta temprana que estima, por Unidad de Gestión Territorial (UGT)
de ANTAMINA en Áncash y con anticipación, la probabilidad de un conflicto
social o protesta. Complementa (no reemplaza) el sistema de alertas reactivo
actual.

**Alcance: exclusivamente la región de Áncash y el cliente ANTAMINA.** El
proyecto se reenfocó a este alcance único; no se mantiene código, datos ni
configuración de otras zonas o clientes.

## Objetivo

Clasificación binaria supervisada con componente temporal (pronóstico): para
una UGT y una ventana futura, predecir si habrá conflicto/protesta (sí/no). El
entregable es un puntaje de riesgo semanal por UGT, consumido vía el dashboard
predictivo y el script de scoring.

## Decisiones clave (NO cambiar sin justificar)

- **Etiqueta:** binaria. Definida como "protesta nueva en la UGT" en la
  ventana futura — NO la mera existencia de conflicto crónico (eso haría el
  target trivialmente positivo).
- **Unidad de predicción:** una fila = `UGT × semana` (granularidad semanal).
  NO distrito (datos demasiado escasos: 9 de 18 distritos de interés tuvieron
  alguna vez una protesta) NI departamento completo (label se vuelve trivial
  por protestas urbanas de Chimbote/Huaraz no relacionadas con la operación).
- **UGTs de ANTAMINA (4):** Mina San Marcos, Huallanca, Valle Fortaleza,
  Huarmey. Cada una agrupa varios distritos (ver `src/scoring/ancash_datos.py`,
  constante `JERARQUIA`, 18 distritos en total).
- **Horizontes:** evaluar multi-horizonte `y_14`, `y_30`, `y_60` días. El
  modelo desplegado usa `y_30`.
- **Tipo de conflicto:** protestas y violencia que afecten directa o
  indirectamente a la operación de ANTAMINA en sus UGTs.
- **Label y features autoregresivas vienen exclusivamente de eventos dentro
  de los 18 distritos de las 4 UGTs** (BD de incidentes + formulario de
  monitoreo Antamina). Tres features de contexto (`tasa_pobreza`,
  `def_escalamiento_ancash`, `oefa_denuncias_mineria_12m`) son departamentales
  o de actividad genérica, usadas como señal ambiente, no como fuente del
  label.
- **Hueco de cobertura conocido:** Paramonga (distrito de Valle Fortaleza,
  único en provincia "Barranca - Lima") no tiene eventos en ninguna de las
  fuentes actuales — quedó sin cobertura de incidentes tras el cambio de
  fuente de datos (2026-06-30). Valle Fortaleza predice con los otros 9
  distritos.

## Fuentes de datos (todas filtradas o acotadas a Áncash)

- **BD de incidentes interna** (`src/data/dataIncidentes.xlsx`): eventos
  geolocalizados con distrito y fecha. Fuente principal del label y de las
  features autoregresivas (resolución semanal).
- **Formulario de monitoreo Antamina** (`src/data/dataFormulario.xlsx`): log
  de noticias específico de Antamina (~8,400 filas), categoría "Protestas,
  Paros y Bloqueos". Se suma a la BD de incidentes (deduplicado a 1 evento
  por distrito×fecha) — aporta ~3x más densidad de eventos de protesta dentro
  de las 4 UGTs que la BD de incidentes sola. Ver `src/scoring/ancash_datos.py`,
  `_cargar_formulario_protestas_raw()`.
- **Reportes situacionales de Antamina** (`src/data/2025/`, `src/data/2026/`,
  ~53 reportes Word/PowerPoint semanales/mensuales; `src/data/loader_reportes.py`):
  texto narrativo del equipo de relaciones de Antamina. ÚNICA fuente del
  proyecto que distingue explícitamente menciones A Antamina (no solo
  conflicto genérico de la zona) y que registra compromiso de las partes
  (mesas de diálogo, acuerdos, cronogramas) — antes marcado "PENDIENTE DE
  FUENTE" en el dashboard. Features: `rep_antamina_neg_4w` (tensión dirigida
  a Antamina), `rep_compromiso_4w` (mesas de diálogo/acuerdos activos).
- **Defensoría del Pueblo** (`src/data/scraper_defensoria.py`): reporte
  mensual de escalamiento de conflicto, acotado a Áncash. Feature de
  contexto departamental (rezago de 2 meses, anti-fuga).
- **Calendario** (feriados, elecciones, fechas críticas): features
  estructurales, no específicas de zona.
- **INEI** (`src/data/loader_inei.py`): tasa de pobreza de Áncash. Feature
  lenta (anual), acotada a Áncash.
- **OEFA/SINADA** (`src/data/loader_oefa.py`): denuncias ambientales públicas
  (portal de datos abiertos), filtradas a actividad minera dentro de las 4
  UGTs. Feature de **contexto acumulado** (ventana móvil de 12 meses),
  NO un predictor de corto plazo validado — se intentó comprobar si las
  denuncias anteceden a las protestas reales y el resultado fue inconcluso
  por falta de superposición temporal entre fuentes (el archivo de OEFA
  tiene ~12+ meses de rezago de publicación). Ver memoria del proyecto.
- **Historial de prensa** (`src/data/dataHistoricaProtestas.xlsx`,
  `src/data/loader_historica.py`): 481 protestas en Áncash desde 2001 (inicio
  de operaciones de ANTAMINA), fuente periodística. 60 eventos en las 4 UGTs,
  34 dirigidos explícitamente a Antamina. Se intentó usar como feature de
  tensión histórica (`hist_prot_antamina_acum`, `hist_prot_antamina_5y`) pero
  se descartó: los eventos son casi constantes por UGT en la ventana 2024–2026
  (todos pre-2024), por lo que actúan como ruido. Código disponible en
  `_join_historica()` — se podrá activar cuando `dataIncidentes` tenga
  cobertura histórica desde 2018–2021.
- **Pendientes de integrar** (mejoras identificadas, no implementadas):
  histórico de alertas propias, precio del cobre (commodities, zona minera),
  intensidad mediática (GDELT).

## Regla crítica: anti-fuga temporal (leakage)

- Toda feature de la semana `t` usa SOLO datos disponibles hasta el fin de la
  semana `t`. Nada posterior.
- El label `y_h` mira la ventana futura `[t+1, t+h]`. Features miran atrás.
- Defensoría se publica con rezago (reporte del mes M sale en M+1): usar
  SIEMPRE el último reporte efectivamente publicado a la fecha de corte.
- INEI: valor "as-of" (último publicado a la fecha, clip a 2025).

## Stack

- Python + venv. pandas, scikit-learn, joblib.
- Trabajo 100% LOCAL. Fuentes y datasets intermedios en `data/` (gitignored).
  Sin Neon, sin Postgres, sin Railway — no hay despliegue en la nube.

## Metodología (no negociable)

- **Validación temporal** (walk-forward / TimeSeriesSplit), NUNCA k-fold
  aleatorio.
- Métricas: **PR-AUC** y **recall a precisión fija**, NO accuracy (clases
  desbalanceadas).
- **Punto de control go/no-go:** el modelo debe superar un baseline trivial
  (tasa histórica de la UGT) por al menos +0.05 de PR-AUC. Si no, detener.

## Estructura de repositorio

```
ML alertas/
├── requirements.txt
├── CLAUDE.md
├── README_ANCASH.md       # documentación técnica detallada del modelo
├── notebooks/             # exploración
├── src/
│   ├── data/               # loaders: incidentes, calendario, INEI, Defensoría
│   ├── dataset/
│   │   └── build_ancash.py        # tabla maestra UGT × semana
│   ├── models/
│   │   └── train_ancash.py        # entrenamiento + punto de control go/no-go
│   └── scoring/
│       ├── ancash_datos.py            # índice de actividad observada (no es predicción)
│       ├── score_ancash.py            # scoring semanal del modelo
│       ├── dashboard_ancash.py        # dashboard Plotly (interno)
│       ├── dashboard_ancash_predictivo.py  # dashboard con plantilla de diseño + datos reales
│       └── assets/ancash_dashboard/    # plantilla visual, runtime, logo
├── models/                # artefactos entrenados (.pkl, gitignored)
└── data/                  # datasets intermedios (gitignored)
```

## Convenciones

- Código y comentarios pueden ir en español.
- Las fechas de corte y ventanas SIEMPRE explícitas en el código (evitar fuga).
- Antes de proponer un cambio al modelo, confirmar que el baseline existe y
  está medido.

## Estado actual (julio 2026)

### Modelo activo (Track A — Áncash, UGT × semana)
- Archivo: `models/modelo_v1_track_A_ancash.pkl`
- Algoritmo: regresión logística (`class_weight="balanced"`), horizonte `y_30`
- PR-AUC modelo (logistic_regression): **0.8004** (3 folds efectivos; Fold 1
  skip porque Jan-Jun 2024 tiene 0 positivos en entrenamiento — propio de los
  datos, no un bug). PR-AUC baseline: 0.554 → GO (+0.2464). Walk-forward RF:
  0.7922. Se guarda LR por probabilidades bien calibradas (Huarmey 90.3%, spread
  amplio), vs RF que comprime probabilidades ~63% para todas las UGTs.
- Dataset: 444 observaciones (4 UGTs × ~111 semanas), 195 con protesta (43.9%)
- **Umbral de alerta calibrado** (`src/models/recalibrar_umbral.py`,
  2026-06-30): ALTO ≥ 80% (precisión 75%, recall 69%, tasa alerta 59%),
  MEDIO 44–79%, BAJO < 44%.

### Extensión de ventana histórica (infraestructura lista, bloqueada)
- Se descargaron 62 PDFs de la Defensoría (2018–2023, 86% cobertura).
- `src/data/loader_defensoria_historico.py` parsea PDFs y extrae 65 eventos
  de protesta (mes×UGT) — ver `data/interim/defensoria_hist_conflictos.parquet`.
- `build_ancash._incidentes_defensoria_hist()` convierte esos eventos a
  formato compatible (fecha=día 15 del mes) y los añade a `_incidentes_por_ugt`.
- **Para activar:** cambiar `FECHA_INICIO = "2018-01-01"` en `build_ancash.py`.
- **Intentos de calibración fallidos (julio 2026):**
  1. RF sin calibrar: probabilidades comprimidas ~63% para todas las UGTs
  2. `CalibratedClassifierCV(isotonic, cv=TimeSeriesSplit(3))`: calibración se
     ancla a la tasa histórica (27.2%) en vez de la de deployment (43.9%) →
     todo BAJO, 22–26%, sin spread útil. También baja PR-AUC promedio de 0.80
     a 0.51 porque folds históricos (2019–2022) tienen labels Defensoría muy
     escasos (~5% positivos) y hunden la métrica.
  3. LR tampoco mejora: baja de 0.8004 a 0.4929 porque `rep_antamina_neg_4w`
     y `rep_compromiso_4w` son 0 en 2018-2023 y diluyen sus coeficientes.
- **Para desbloquear:** calibrar isotónica SOLO sobre predicciones OOF del
  período 2024-2026 (distribución de deployment): entrenar RF en 2018-2023,
  predecir 2024-2026, ajustar isotónica ahí, y re-entrenar RF en todos los
  datos con esa calibración aplicada post-hoc. Requiere refactorizar
  `train_ancash._preparar()` para exponer fechas y separar el paso de
  calibración.

### Operación semanal
```bash
python -m src.scoring.score_ancash               # predicción por UGT (consola)
python -m src.scoring.dashboard_ancash_predictivo # dashboard interactivo
```

### Reconstruir el pipeline desde cero
```bash
python -m src.data.loader_calendario
python -m src.data.loader_inei
python -m src.data.scraper_defensoria   # requiere conexión, opcional si ya existe el CSV
python -m src.data.loader_oefa          # requiere conexión, opcional si ya existe el parquet
python -m src.data.loader_reportes      # parsea src/data/2025/ y 2026/ (Word/PowerPoint)
python -m src.data.loader_historica     # historial de prensa 2001–2025 (dataHistoricaProtestas.xlsx)
# Extensión histórica (opcional, requiere conexión):
python -m src.data.loader_defensoria_historico --desde 2018-01 --hasta 2023-12
python -m src.dataset.build_ancash
python -m src.models.train_ancash
python -m src.models.recalibrar_umbral  # recalibrar el umbral si el modelo cambió
```

### Fase de validación operativa
Cada viernes anotar: ¿hubo conflicto/protesta real esta semana en cada UGT?
(sí/no). Con varias semanas de datos se puede calcular el recall operativo
real y contrastarlo contra el PR-AUC de validación.

### Limitaciones honestas (ver README_ANCASH.md, sección 12)
Dataset todavía pequeño (444 filas, 195 positivos/43.9%). Umbral calibrado
(ALTO≥80%, MEDIO≥44%) con predicciones OOF — válido solo para el modelo
actual; si cambia el feature set hay que recalibrar. Features de contexto
departamental limitadas. Ver sección 12 del README para lista completa.
