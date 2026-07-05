"""Loader de denuncias ambientales OEFA (SINADA) — contexto de tensión minera.

Descarga el dataset público de denuncias ambientales de OEFA (portal de datos
abiertos, sin API key) y lo acota a Áncash. Se usa como feature de CONTEXTO
acumulado (tensión ambiental ligada a actividad minera en la zona), análogo a
`def_escalamiento_ancash` — NO como predictor de corto plazo: el archivo de
OEFA tiene un rezago de publicación de varios meses (la última actualización
observada solo llega hasta dic-2024 pese a republicarse en abr-2025), así que
no refleja eventos recientes en tiempo real.

IMPORTANTE (ver memoria del proyecto): se intentó validar si las denuncias
anteceden a las protestas reales (hipótesis de alerta temprana) y el análisis
fue inconcluso por falta de superposición temporal entre ambas fuentes (solo
9 de 51 protestas caían dentro del rango con datos de OEFA). Por eso esta
fuente se trata como contexto, no como señal validada de corto plazo.
"""
import sys
from pathlib import Path

import pandas as pd
import requests

URL_DESCARGA = "https://datosabiertos.oefa.gob.pe/datasets/197779-denuncias-sinada-2019-2025.download/"
RAW_DIR = Path("data/raw/oefa")
ARCHIVO_RAW = RAW_DIR / "Denuncias_SINADA.xlsx"

INTERIM_DIR = Path("data/interim")
SALIDA = INTERIM_DIR / "oefa_denuncias_ancash.parquet"

DEPARTAMENTO = "ANCASH"


# ── Descarga ──────────────────────────────────────────────────────────────────
def descargar() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    r = requests.get(URL_DESCARGA, timeout=60, allow_redirects=True)
    r.raise_for_status()
    ARCHIVO_RAW.write_bytes(r.content)
    return ARCHIVO_RAW


# ── Carga y limpieza ──────────────────────────────────────────────────────────
def cargar_oefa(usar_cache: bool = True) -> pd.DataFrame:
    if not usar_cache or not ARCHIVO_RAW.exists():
        descargar()

    df = pd.read_excel(ARCHIVO_RAW, sheet_name="BASE DE DATOS")
    df["fecha"] = pd.to_datetime(df["FECHA DE REGISTRO"], errors="coerce")
    df["departamento"] = df["DEPARTAMENTO"].astype(str).str.strip().str.upper()
    df["distrito"] = df["DISTRITO"].astype(str).str.strip()
    df["actividad_economica"] = df["ACTIVIDAD ECONOMICA"].astype(str).str.strip()
    df["estado"] = df["ESTADO"].astype(str).str.strip()

    df = df[df["departamento"] == DEPARTAMENTO].copy()
    df = df[df["fecha"].notna()]

    return df[["fecha", "distrito", "actividad_economica", "estado"]].reset_index(drop=True)


# ── Guardar ───────────────────────────────────────────────────────────────────
def guardar(df: pd.DataFrame) -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SALIDA, index=False)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    df = cargar_oefa(usar_cache=False)
    guardar(df)

    es_mineria = df["actividad_economica"].str.upper().str.contains("MINER", na=False)
    print(f"✓ OEFA/SINADA procesado: {len(df)} denuncias en Áncash "
          f"({df['fecha'].min().date()} → {df['fecha'].max().date()})")
    print(f"  De actividad minera: {int(es_mineria.sum())} ({es_mineria.mean()*100:.0f}%)")
    print(f"  Guardado en {SALIDA}")


if __name__ == "__main__":
    main()
