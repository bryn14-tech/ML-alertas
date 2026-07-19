"""Loader de precio del cobre — futuros COMEX (HG=F) via Yahoo Finance.

Descarga la serie semanal y guarda un parquet con el cierre de cada semana.
No requiere API key.

Hipótesis de relevancia para zona minera:
  - Precio ALTO → Antamina más rentable → comunidades exigen más regalías/
    beneficios → tensión por distribución de ingresos mineros.
  - Caída BRUSCA → recortes de inversión social → protestas por promesas
    incumplidas.
  Ambas direcciones pueden disparar conflictos, por eso se usan tanto el
  nivel como el momentum (retorno a 4 semanas).

Anti-fuga: loader_cobre solo descarga datos; _join_cobre en build_ancash
garantiza que para la semana t solo se use el cierre de la semana t-1.
"""
import sys
from pathlib import Path

import pandas as pd

INTERIM_DIR = Path("data/interim")
SALIDA = INTERIM_DIR / "cobre_semanal.parquet"
TICKER = "HG=F"
FECHA_INICIO_DESCARGA = "2020-01-01"  # margen para calcular retornos históricos


def descargar_cobre() -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Instala yfinance: pip install yfinance>=0.2")

    raw = yf.download(
        TICKER,
        start=FECHA_INICIO_DESCARGA,
        interval="1wk",
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError("No se obtuvieron datos de yfinance. Verifica la conexión.")

    # Aplanar MultiIndex de columnas (yfinance ≥ 0.2 lo genera)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = pd.DataFrame({
        "fecha": pd.to_datetime(raw.index).tz_localize(None),
        "cobre_usd_lb": raw["Close"].values.astype(float),
    })
    return df.dropna(subset=["cobre_usd_lb"]).reset_index(drop=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    df = descargar_cobre()
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SALIDA, index=False)
    print(f"✓ Precio del cobre (HG=F, semanal): {len(df)} semanas")
    print(f"  Rango: {df['fecha'].min().date()} → {df['fecha'].max().date()}")
    print(f"  Último cierre: USD {df['cobre_usd_lb'].iloc[-1]:.4f}/lb")
    print(f"  Min / Max período: {df['cobre_usd_lb'].min():.4f} / {df['cobre_usd_lb'].max():.4f}")
    print(f"  Guardado en {SALIDA}")


if __name__ == "__main__":
    main()
