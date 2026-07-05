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
    _join_defensoria,
    _join_historica,
    _join_impacto,
    _join_inei,
    _join_oefa,
    _join_reportes,
)

MODELO_PATH = Path("models") / "modelo_v1_track_A_ancash.pkl"

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


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    pred = predecir()
    lunes = pred["semana_scoring"].iloc[0]
    print(f"Predicción de protesta · próximos 30 días · semana {lunes.date()} (Áncash, por UGT):\n")
    for _, r in pred.iterrows():
        pct = r["probabilidad"] * 100
        nivel = "ALTO" if pct >= UMBRAL_ALTO else ("MEDIO" if pct >= UMBRAL_MEDIO else "BAJO")
        print(f"  {r['ugt']:<18} {pct:5.1f}%   {nivel}")


if __name__ == "__main__":
    main()
