"""Loader de protestas históricas en Áncash (prensa escrita, 2001–2025).

Fuente: dataHistoricaProtestas.xlsx — registro periodístico que cubre desde
el inicio de operaciones de ANTAMINA (2001). Cada fila es una protesta con
atributos: tipo de acción, adversario, reclamo, duración, participantes.

Se extraen solo los eventos en los distritos de las 4 UGTs y se marca si el
evento fue dirigido explícitamente a una empresa privada (proxy Antamina).
El parquet resultante es consumido por build_ancash._join_historica() para
construir features de tensión histórica acumulada por UGT.

Nota: NO se usa para extender la ventana de entrenamiento — dataIncidentes
solo cubre desde 2023-08, por lo que las features autoregresivas (inc_prot_*)
serían NaN para períodos pre-2023. Se usa exclusivamente como feature de
contexto histórico de largo plazo.
"""
import sys
import unicodedata
from pathlib import Path

import pandas as pd

from src.scoring.ancash_datos import JERARQUIA

ARCHIVO = Path("src/data") / "dataHistoricaProtestas.xlsx"
INTERIM_DIR = Path("data/interim")
SALIDA = INTERIM_DIR / "historica_protestas_ugt.parquet"


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()


def cargar_historica() -> pd.DataFrame:
    df = pd.read_excel(ARCHIVO, sheet_name="Hoja1")
    df["Fecha_Protesta"] = pd.to_datetime(df["Fecha_Protesta"], errors="coerce")
    df = df[df["Fecha_Protesta"].notna()].copy()

    dist2ugt = {_norm(d): ugt for ugt, _, d in JERARQUIA}
    df["distrito_n"] = df["Distrito"].apply(_norm)
    df["ugt"] = df["distrito_n"].map(dist2ugt)
    df = df[df["ugt"].notna()].copy()

    # Evento Antamina-dirigido: adversario empresa privada O mención directa
    es_empresa = df["Adversario"].str.contains("Empresa", case=False, na=False)
    menciona_ant = df.apply(
        lambda r: r.astype(str).str.contains("ntamina", case=False).any(), axis=1
    )
    df["es_antamina"] = es_empresa | menciona_ant

    accion_col = [c for c in df.columns if "cci" in c and "_1" in c and "id" not in c.lower()][0]
    adversario_col = [c for c in df.columns if "dvers" in c and "id" not in c.lower()][0]
    reclamo_col = [c for c in df.columns if "eclam" in c and "_t" not in c
                   and "ub" not in c.lower() and "id" not in c.lower()][0]

    return df[["Fecha_Protesta", "ugt", "Distrito", accion_col, adversario_col,
               reclamo_col, "es_antamina"]].rename(columns={
        "Fecha_Protesta": "fecha",
        accion_col: "accion",
        adversario_col: "adversario",
        reclamo_col: "reclamo",
    }).copy()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    df = cargar_historica()
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SALIDA, index=False)
    print(f"✓ Histórico de protestas (prensa): {len(df)} eventos en las 4 UGTs")
    print(f"  Rango: {df['fecha'].min().date()} → {df['fecha'].max().date()}")
    print(f"  Antamina-dirigidos: {df['es_antamina'].sum()}")
    print()
    print(df.groupby("ugt")[["es_antamina"]].agg(["sum", "count"]).rename(
        columns={"sum": "antamina", "count": "total"}).droplevel(0, axis=1).to_string())
    print(f"\n  Guardado en {SALIDA}")


if __name__ == "__main__":
    main()
