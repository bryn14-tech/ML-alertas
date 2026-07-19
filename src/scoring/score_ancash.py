"""Scoring predictivo Track A — Áncash (UGT × semana).

Calcula la probabilidad de protesta en los próximos 30 días para cada UGT de
ANTAMINA, con el modelo guardado. Reutiliza las funciones de construcción de
features de build_ancash (no duplica lógica) y respeta el anti-fuga: las
features de la semana actual usan solo incidentes hasta hoy.
"""
import sys
from itertools import product
from pathlib import Path

import joblib
import pandas as pd

from src.dataset.build_ancash import (
    FECHA_INICIO,
    UGTS,
    _features_incidentes,
    _floor_to_monday,
    _incidentes_por_ugt,
    _join_calendario,
    _join_cobre,
    _join_defensoria,
    _join_historica,
    _join_impacto,
    _join_inei,
    _join_oefa,
    _join_reportes,
)

MODELO_PATH   = Path("models") / "modelo_v1_track_A_ancash.pkl"
LOG_PRED_PATH = Path("data/processed") / "predicciones_log.csv"
LOG_REAL_PATH = Path("data/processed") / "resultados_reales.csv"

# Umbral calibrado (2026-06-30, src/models/recalibrar_umbral.py): precisión
# 74%, recall 65%, tasa de alerta 56% sobre predicciones out-of-fold
# walk-forward. MEDIO usa la tasa histórica de protesta (43.9% ≈ 44%) como
# referencia de "más riesgo que el promedio".
UMBRAL_ALTO = 80
UMBRAL_MEDIO = 44


def _semana_actual() -> pd.Timestamp:
    return _floor_to_monday(pd.Series([pd.Timestamp.today()])).iloc[0]


def calcular_features_actuales(lunes_actual: pd.Timestamp) -> pd.DataFrame:
    semanas = pd.date_range(FECHA_INICIO, lunes_actual, freq="W-MON")
    skeleton = pd.DataFrame(list(product(UGTS, semanas)), columns=["ugt", "semana_inicio"])

    inc = _incidentes_por_ugt()
    inc = inc[inc["fecha"] <= pd.Timestamp.today().normalize()]  # anti-fuga defensivo

    master = _features_incidentes(skeleton, inc)
    master = _join_calendario(master)
    master = _join_inei(master)
    master = _join_defensoria(master)
    master = _join_oefa(master)
    master = _join_cobre(master)
    master = _join_impacto(master)
    master["prot_impacto_ultimo"] = master["prot_impacto_ultimo"].fillna(0)
    master = _join_reportes(master)
    master = _join_historica(master)
    master["hist_prot_antamina_acum"] = master["hist_prot_antamina_acum"].fillna(0)
    master["hist_prot_antamina_5y"] = master["hist_prot_antamina_5y"].fillna(0)

    actual = master[master["semana_inicio"] == lunes_actual].copy()
    if len(actual) != len(UGTS):
        faltan = set(UGTS) - set(actual["ugt"])
        raise RuntimeError(f"Features incompletas para la semana {lunes_actual.date()}. Faltan UGTs: {faltan}")
    return actual


def predecir(lunes_actual: pd.Timestamp | None = None) -> pd.DataFrame:
    if not MODELO_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {MODELO_PATH}. Corre src.models.train_ancash primero.")
    paquete = joblib.load(MODELO_PATH)
    lunes_actual = lunes_actual or _semana_actual()

    df = calcular_features_actuales(lunes_actual)
    X = df[paquete["feature_cols"]]
    df["probabilidad"] = paquete["model"].predict_proba(X)[:, 1]
    df["semana_scoring"] = lunes_actual
    cols_extra = [c for c in ("rep_compromiso_4w", "rep_antamina_neg_4w") if c in df.columns]
    return df[["ugt", "semana_scoring", "probabilidad"] + cols_extra].sort_values("probabilidad", ascending=False).reset_index(drop=True)


def _guardar_log(pred: pd.DataFrame) -> None:
    """Appends predictions to predicciones_log.csv (one row per UGT per run)."""
    fecha_ej = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    filas = []
    for _, r in pred.iterrows():
        pct = r["probabilidad"] * 100
        nivel = "ALTO" if pct >= UMBRAL_ALTO else ("MEDIO" if pct >= UMBRAL_MEDIO else "BAJO")
        filas.append({
            "semana_scoring": r["semana_scoring"].date(),
            "ugt": r["ugt"],
            "probabilidad": round(r["probabilidad"], 4),
            "nivel": nivel,
            "fecha_ejecucion": fecha_ej,
        })
    nuevo = pd.DataFrame(filas)
    LOG_PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOG_PRED_PATH.exists():
        existente = pd.read_csv(LOG_PRED_PATH)
        semana_str = str(nuevo["semana_scoring"].iloc[0])
        # Evitar duplicar la misma semana si se corre dos veces el mismo día
        existente = existente[existente["semana_scoring"] != semana_str]
        pd.concat([existente, nuevo], ignore_index=True).to_csv(LOG_PRED_PATH, index=False)
    else:
        nuevo.to_csv(LOG_PRED_PATH, index=False)
    print(f"\n✓ Log guardado en {LOG_PRED_PATH} ({len(filas)} filas)")


def _inicializar_log_real() -> None:
    """Crea resultados_reales.csv si no existe (plantilla para anotación manual)."""
    if LOG_REAL_PATH.exists():
        return
    LOG_REAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["semana", "ugt", "hubo_protesta_real", "notas"]).to_csv(
        LOG_REAL_PATH, index=False
    )
    print(f"✓ Plantilla creada en {LOG_REAL_PATH} — completar manualmente cada viernes")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    pred = predecir()
    lunes = pred["semana_scoring"].iloc[0]
    print(f"Predicción de protesta · próximos 30 días · semana {lunes.date()} (Áncash, por UGT):\n")
    for _, r in pred.iterrows():
        pct = r["probabilidad"] * 100
        nivel = "ALTO" if pct >= UMBRAL_ALTO else ("MEDIO" if pct >= UMBRAL_MEDIO else "BAJO")
        print(f"  {r['ugt']:<18} {pct:5.1f}%   {nivel}")
    _guardar_log(pred)
    _inicializar_log_real()


if __name__ == "__main__":
    main()
