"""Calibración del umbral de alerta — Track A Áncash (UGT × semana).

Hoy el dashboard usa 55%/35% como referencia VISUAL para ALTO/MEDIO/BAJO,
sin respaldo estadístico. Este script calcula un umbral real a partir de las
predicciones out-of-fold de la validación walk-forward (cada semana se
predice solo con un modelo entrenado con datos estrictamente anteriores,
igual que en producción — no hay fuga).

Regla de decisión: el umbral elegido debe cumplir recall>=RECALL_MIN Y
precision>=PRECISION_MIN simultáneamente (mismo criterio que se usó para
Track B). Si ningún umbral cumple ambos pisos a la vez, se cae a un
fallback de máximo F1.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

from src.models.train_ancash import FEATURES, MASTER, MODELOS, N_SPLITS, TARGET, _preparar

SALIDA = Path("data/processed") / "calibracion_umbral_ancash.csv"
MODELO_FINAL = "logistic_regression"  # debe coincidir con el que guarda train_ancash.py

RECALL_MIN = 0.40
PRECISION_MIN = 0.35
CANDIDATOS = np.round(np.arange(0.05, 0.96, 0.01), 2)


def _predicciones_oof(X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Probabilidad predicha para cada fila SOLO cuando cae en el conjunto de
    prueba de algún fold walk-forward (out-of-fold) — nunca se evalúa una
    fila con un modelo que la vio en entrenamiento."""
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    probs = np.full(len(y), np.nan)
    for tr, te in tscv.split(X):
        if y.iloc[tr].nunique() < 2 or y.iloc[te].nunique() < 2:
            continue
        modelo = clone(MODELOS[MODELO_FINAL]).fit(X.iloc[tr], y.iloc[tr])
        probs[te] = modelo.predict_proba(X.iloc[te])[:, 1]
    mask = ~np.isnan(probs)
    return probs[mask], y.values[mask]


def _barrer_umbrales(probs: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    filas = []
    for t in CANDIDATOS:
        pred = (probs >= t).astype(int)
        filas.append({
            "umbral": t,
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "tasa_alerta": pred.mean(),
        })
    return pd.DataFrame(filas)


def elegir_umbral(tabla: pd.DataFrame) -> tuple[pd.Series, str]:
    cumplen = tabla[(tabla["recall"] >= RECALL_MIN) & (tabla["precision"] >= PRECISION_MIN)]
    if not cumplen.empty:
        return cumplen.sort_values("f1", ascending=False).iloc[0], (
            f"cumple recall>={RECALL_MIN} y precision>={PRECISION_MIN} (de los que cumplen, máximo F1)"
        )
    return tabla.sort_values("f1", ascending=False).iloc[0], (
        f"NINGÚN umbral cumplió recall>={RECALL_MIN} y precision>={PRECISION_MIN} a la vez — fallback a máximo F1"
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    master = pd.read_parquet(MASTER)
    X, y = _preparar(master)

    probs, y_eval = _predicciones_oof(X, y)
    print(f"Predicciones out-of-fold: {len(probs)} de {len(y)} filas totales "
          f"({len(y) - len(probs)} quedaron fuera por caer en el primer fold de entrenamiento)\n")

    tabla = _barrer_umbrales(probs, y_eval)
    tabla.to_csv(SALIDA, index=False)

    elegido, motivo = elegir_umbral(tabla)
    print(f"Criterio: {motivo}\n")
    print(f"{'Umbral':<8} | {'Precisión':<10} | {'Recall':<8} | {'F1':<8} | {'Tasa alerta':<12}")
    print("-" * 58)
    for t in [0.35, 0.45, 0.50, 0.55, 0.65, round(float(elegido['umbral']), 2)]:
        fila = tabla.iloc[(tabla["umbral"] - t).abs().idxmin()]
        marca = " <- elegido" if abs(fila["umbral"] - elegido["umbral"]) < 1e-9 else ""
        print(f"{fila['umbral']:<8.2f} | {fila['precision']:<10.3f} | {fila['recall']:<8.3f} | "
              f"{fila['f1']:<8.3f} | {fila['tasa_alerta']*100:<11.1f}%{marca}")

    print(f"""
╔════════════════════════════════════════════════════════╗
║   UMBRAL CALIBRADO — Track A Áncash                      ║
╠════════════════════════════════════════════════════════╣
  Umbral:        {elegido['umbral']:.2f}
  Precisión:     {elegido['precision']:.3f}
  Recall:        {elegido['recall']:.3f}
  F1:            {elegido['f1']:.3f}
  Tasa de alerta: {elegido['tasa_alerta']*100:.1f}% de las semana-UGT
╚════════════════════════════════════════════════════════╝
""")
    print(f"✓ Tabla completa guardada en {SALIDA}")
    print(f"\nSiguiente paso manual: actualizar el umbral ALTO en score_ancash.py / "
          f"dashboard_ancash_predictivo.py de 0.55 a {elegido['umbral']:.2f} si se decide adoptar este valor.")


if __name__ == "__main__":
    main()
