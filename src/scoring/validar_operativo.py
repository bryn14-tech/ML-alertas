"""Validación operativa semanal — Áncash Track A.

Compara las predicciones guardadas en predicciones_log.csv con los resultados
reales anotados manualmente en resultados_reales.csv y reporta métricas
operativas: precisión y recall por nivel de alerta, tendencia temporal.

Uso:
    python -m src.scoring.validar_operativo
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LOG_PRED = Path("data/processed/predicciones_log.csv")
LOG_REAL = Path("data/processed/resultados_reales.csv")

UMBRAL_ALTO = 80
UMBRAL_MEDIO = 44


def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not LOG_PRED.exists():
        raise FileNotFoundError(
            f"No se encontró {LOG_PRED}. Corre src.scoring.score_ancash al menos una vez."
        )
    if not LOG_REAL.exists():
        raise FileNotFoundError(
            f"No se encontró {LOG_REAL}. Corre src.scoring.score_ancash para crear la plantilla."
        )
    pred = pd.read_csv(LOG_PRED, parse_dates=["semana_scoring"])
    real = pd.read_csv(LOG_REAL)
    real["semana"] = pd.to_datetime(real["semana"], errors="coerce")
    return pred, real


def _metricas_nivel(sub: pd.DataFrame, nivel: str) -> dict:
    """Precisión y recall para un nivel de alerta dado."""
    # Verdaderos positivos: alerta disparada Y hubo protesta real
    tp = ((sub["nivel"] == nivel) & (sub["hubo_protesta_real"] == 1)).sum()
    fp = ((sub["nivel"] == nivel) & (sub["hubo_protesta_real"] == 0)).sum()
    fn = ((sub["nivel"] != nivel) & (sub["hubo_protesta_real"] == 1)).sum()
    alertas = tp + fp
    prec = tp / alertas if alertas > 0 else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return {"alertas": int(alertas), "tp": int(tp), "fp": int(fp),
            "precision": prec, "recall": rec}


def _reporte_por_ugt(merged: pd.DataFrame) -> None:
    print("\n── Por UGT ─────────────────────────────────────────────────────")
    for ugt, g in merged.groupby("ugt"):
        n_semanas = len(g)
        n_real = int(g["hubo_protesta_real"].sum())
        n_pred_alto = int((g["nivel"] == "ALTO").sum())
        tp_alto = int(((g["nivel"] == "ALTO") & (g["hubo_protesta_real"] == 1)).sum())
        print(f"  {ugt:<22}  semanas: {n_semanas:3d}  "
              f"reales: {n_real:3d}  alertas ALTO: {n_pred_alto:3d}  "
              f"TP-ALTO: {tp_alto:3d}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    pred, real = cargar_datos()

    # Semanas con datos reales anotados
    real_ok = real[real["hubo_protesta_real"].notna()].copy()
    real_ok["hubo_protesta_real"] = real_ok["hubo_protesta_real"].astype(int)

    if len(real_ok) == 0:
        print("⚠  Sin semanas anotadas todavía en resultados_reales.csv.")
        print(f"   Edita {LOG_REAL} y completa la columna hubo_protesta_real (1 o 0) para cada UGT.")
        print(f"\n   Predicciones acumuladas: {len(pred)} filas ({pred['semana_scoring'].nunique()} semanas)")
        _mostrar_historial(pred)
        return

    # Join predicciones × resultados reales
    merged = pred.merge(
        real_ok[["semana", "ugt", "hubo_protesta_real"]],
        left_on=["semana_scoring", "ugt"],
        right_on=["semana", "ugt"],
        how="inner",
    )
    if len(merged) == 0:
        print("⚠  No hay cruce entre predicciones y resultados reales.")
        print("   Verifica que las fechas en resultados_reales.csv coincidan con semanas_scoring del log.")
        return

    n_semanas = merged["semana_scoring"].nunique()
    n_obs = len(merged)
    tasa_real = merged["hubo_protesta_real"].mean()
    print(f"Validación operativa · {n_semanas} semanas anotadas · {n_obs} observaciones (UGT×semana)")
    print(f"Tasa real de protesta: {tasa_real*100:.1f}%\n")

    # Métricas globales por nivel
    print("── Métricas por nivel de alerta ────────────────────────────────")
    print(f"{'Nivel':<8} {'Alertas':>8} {'TP':>6} {'FP':>6} {'Precisión':>10} {'Recall':>8}")
    print("-" * 55)
    for nivel in ["ALTO", "MEDIO", "BAJO"]:
        m = _metricas_nivel(merged, nivel)
        prec_str = f"{m['precision']*100:.1f}%" if not np.isnan(m["precision"]) else "  —  "
        rec_str  = f"{m['recall']*100:.1f}%"   if not np.isnan(m["recall"])    else "  —  "
        print(f"{nivel:<8} {m['alertas']:>8} {m['tp']:>6} {m['fp']:>6} {prec_str:>10} {rec_str:>8}")

    # Precisión binaria: ¿Cualquier nivel de alerta (no BAJO) anticipó protesta?
    merged["alerto"] = merged["nivel"].isin(["ALTO", "MEDIO"]).astype(int)
    tp_any = ((merged["alerto"] == 1) & (merged["hubo_protesta_real"] == 1)).sum()
    fp_any = ((merged["alerto"] == 1) & (merged["hubo_protesta_real"] == 0)).sum()
    fn_any = ((merged["alerto"] == 0) & (merged["hubo_protesta_real"] == 1)).sum()
    alertas_any = tp_any + fp_any
    prec_any = tp_any / alertas_any if alertas_any > 0 else float("nan")
    rec_any  = tp_any / (tp_any + fn_any) if (tp_any + fn_any) > 0 else float("nan")
    f1_any   = 2 * prec_any * rec_any / (prec_any + rec_any) if (prec_any + rec_any) > 0 else float("nan")
    print("-" * 55)
    print(f"{'ALTO+MED':<8} {int(alertas_any):>8} {int(tp_any):>6} {int(fp_any):>6} "
          f"{prec_any*100:>9.1f}% {rec_any*100:>7.1f}%")
    print(f"\n  F1 (ALTO+MEDIO): {f1_any:.3f}" if not np.isnan(f1_any) else "")

    _reporte_por_ugt(merged)

    # Tendencia temporal (precisión ALTO+MEDIO por mes)
    merged["mes"] = merged["semana_scoring"].dt.to_period("M")
    meses = merged.groupby("mes").apply(
        lambda g: pd.Series({
            "n_obs": len(g),
            "tasa_real": g["hubo_protesta_real"].mean(),
            "precision": (
                ((g["alerto"] == 1) & (g["hubo_protesta_real"] == 1)).sum() /
                max((g["alerto"] == 1).sum(), 1)
            ),
            "recall": (
                ((g["alerto"] == 1) & (g["hubo_protesta_real"] == 1)).sum() /
                max(g["hubo_protesta_real"].sum(), 1)
            ),
        }),
        include_groups=False
    ).reset_index()

    if len(meses) >= 2:
        print("\n── Tendencia mensual (precisión de alerta ALTO+MEDIO) ──────────")
        print(f"{'Mes':<10} {'Obs':>5} {'Tasa real':>10} {'Precisión':>10} {'Recall':>8}")
        print("-" * 48)
        for _, row in meses.iterrows():
            print(f"{str(row['mes']):<10} {int(row['n_obs']):>5} "
                  f"{row['tasa_real']*100:>9.1f}% "
                  f"{row['precision']*100:>9.1f}% "
                  f"{row['recall']*100:>7.1f}%")

    print(f"\n✓ Predicciones disponibles: {len(pred)} filas ({pred['semana_scoring'].nunique()} semanas en log)")
    pendientes = pred["semana_scoring"].nunique() - n_semanas
    if pendientes > 0:
        print(f"⚠  {pendientes} semana(s) en el log sin anotar todavía en {LOG_REAL}")
        _mostrar_historial(pred[~pred["semana_scoring"].isin(merged["semana_scoring"])])


def _mostrar_historial(pred: pd.DataFrame) -> None:
    """Muestra el historial de predicciones sin anotar como referencia."""
    print("\n── Predicciones pendientes de anotar ───────────────────────────")
    print(f"{'Semana':<12} {'UGT':<22} {'%':>6}  {'Nivel'}")
    print("-" * 55)
    for _, r in pred.sort_values(["semana_scoring", "ugt"]).iterrows():
        pct = r["probabilidad"] * 100
        print(f"{str(r['semana_scoring'].date()):<12} {r['ugt']:<22} {pct:5.1f}%  {r['nivel']}")


if __name__ == "__main__":
    main()
