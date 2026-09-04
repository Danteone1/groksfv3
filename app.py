
"""
SITER-CAE v6.1 ADVANCED TERRITORIAL INFERENCE
=========================
Laboratorio territorial para CDMX (v6.0: capa organización agregada + red geo + guardrails).

OBJETIVO
--------
Unificar en una sola app:
1) Motor ABM con Mesa.
2) World View tipo NetLogo sobre GIS real cuando exista SHP/GeoJSON.
3) Cinco modos de datos:
   - REAL: CSV + SHP/GeoJSON.
   - DUMMY: datos mínimos de prueba.
   - SINTÉTICO COHERENTE: generado a partir de distribuciones/correlaciones
     del CSV real, sin copiar filas identificables.
   - SINTÉTICO PURO: universo sintético libre.
   - SINTÉTICO CALIBRACIÓN: escenarios de prueba controlados.
4) Análisis estadístico territorial.
5) Presupuesto: costos, cobertura, costo/km, costo/unidad, ROI operacional,
   sensibilidad y Monte Carlo.
6) Brigadistas: plan, GPS observado, cobertura por segmento, territorio
   visitado, horas, km, desviaciones y calidad GPS.
7) Desagregación avanzada agregada: IPF, tomografía, Bayes/Kalman y MRF/CRF-like.
8) Question Engine para responder preguntas de cliente a partir de métricas.
9) Reproducibilidad: seed + experiment_id + output_hash.
9) Export JSON/CSV.

NOTA METODOLÓGICA
-----------------
Los estados de opinión y actores son variables de simulación agregadas.
No se construyen perfiles individuales ni PII. La app está diseñada para
análisis territorial, logística, presupuesto, resiliencia y escenarios.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------
# Mesa
# ---------------------------------------------------------------------
try:
    from mesa import Agent, Model
    MESA_OK = True
    MESA_ERROR = ""
except Exception as exc:
    MESA_OK = False
    MESA_ERROR = repr(exc)

# ---------------------------------------------------------------------
# GIS opcional pero recomendado para SHP/GeoJSON
# ---------------------------------------------------------------------
try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    from shapely.ops import unary_union
    HAS_GIS = True
    GIS_ERROR = ""
except Exception as exc:
    HAS_GIS = False
    GIS_ERROR = repr(exc)

# ---------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------
ALCALDIAS = [
    "ALVARO OBREGON",
    "AZCAPOTZALCO",
    "BENITO JUAREZ",
    "COYOACAN",
    "CUAJIMALPA DE MORELOS",
    "CUAUHTEMOC",
    "GUSTAVO A MADERO",
    "IZTACALCO",
    "IZTAPALAPA",
    "LA MAGDALENA CONTRERAS",
    "MIGUEL HIDALGO",
    "MILPA ALTA",
    "TLAHUAC",
    "TLALPAN",
    "VENUSTIANO CARRANZA",
    "XOCHIMILCO",
]

ALCALDIA_COORDS = {
    "CUAUHTEMOC": (19.4326, -99.1332),
    "BENITO JUAREZ": (19.3984, -99.1576),
    "MIGUEL HIDALGO": (19.4285, -99.2000),
    "COYOACAN": (19.3467, -99.1617),
    "IZTAPALAPA": (19.3550, -99.0620),
    "GUSTAVO A MADERO": (19.4900, -99.1100),
    "ALVARO OBREGON": (19.3580, -99.2270),
    "TLALPAN": (19.2880, -99.1670),
    "XOCHIMILCO": (19.2630, -99.1040),
    "VENUSTIANO CARRANZA": (19.4200, -99.1000),
    "AZCAPOTZALCO": (19.4870, -99.1860),
    "IZTACALCO": (19.3950, -99.0980),
    "CUAJIMALPA DE MORELOS": (19.3570, -99.2900),
    "LA MAGDALENA CONTRERAS": (19.3200, -99.2400),
    "TLAHUAC": (19.2700, -99.0050),
    "MILPA ALTA": (19.1920, -99.0230),
}

STATE_LABEL = {1: "SIMPATIZANTE", -1: "OPOSITOR", 0: "INDECISO"}
STATE_COLORS = {"SIMPATIZANTE": "#2ecc71", "OPOSITOR": "#e74c3c", "INDECISO": "#95a5a6"}
FIELD_COLORS = {
    "CONSOLIDACION": "#2ecc71",
    "DISPUTA_ABIERTA": "#e74c3c",
    "CONTENCION": "#3498db",
}

TRAIT_COLS = [
    "capital_social",
    "acceso_informacion",
    "influencia_liderazgo",
    "arraigo",
    "nivel_movilizacion",
    "desconfianza",
    "exposicion_problema",
]

# Solo estas variables pueden actualizarse con observaciones de campo (nunca GPS).
FIELD_UPDATABLE_VARS = set(TRAIT_COLS + [
    "prioridad_problema",
    "resistencia_institucional",
    "temperatura",
    "opinion_continua",
])

# Capa de organización / militancia AGREGADA (sin PII).
ORG_COLS = [
    "n_militantes_obs",
    "n_universo_est",
    "broker_density",
    "asistencia_evento_rate",
    "aceptacion_mensaje",
    "org_mobilization",
    "org_reliability",
]

NUMERIC_CANDIDATES = TRAIT_COLS + [
    "opinion_continua",
    "temperatura",
    "resistencia_institucional",
    "prioridad_problema",
    "poblacion",
    "longitud_km",
]

# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------
def norm_col(x: Any) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z0-9áéíóúüñ]+", "_", s)
    return s.strip("_")


def clean_alcaldia(x: Any) -> str:
    s = str(x).strip().upper()
    replacements = {
        "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
        "Ü": "U", "Ñ": "N",
    }
    for a, b in replacements.items():
        s = s.replace(a, b)
    return s


def sha256_obj(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl/2)**2
    return 2 * r * math.asin(min(1, math.sqrt(a)))


def path_distance_m(df: pd.DataFrame) -> float:
    if df.empty or len(df) < 2:
        return 0.0
    d = 0.0
    x = df.sort_values("timestamp") if "timestamp" in df.columns else df
    for i in range(1, len(x)):
        d += haversine_m(
            x.iloc[i-1]["lat"], x.iloc[i-1]["lon"],
            x.iloc[i]["lat"], x.iloc[i]["lon"]
        )
    return d


def gini(values) -> float:
    x = np.asarray(values, dtype=float)
    x = np.abs(x[np.isfinite(x)])
    if len(x) == 0 or np.allclose(x.sum(), 0):
        return 0.0
    x = np.sort(x)
    n = len(x)
    return float((2 * np.sum((np.arange(1, n + 1)) * x) / (n * x.sum())) - (n + 1) / n)


def entropy_from_counts(counts) -> float:
    vals = np.asarray(list(counts), dtype=float)
    vals = vals[vals > 0]
    if len(vals) == 0:
        return 0.0
    p = vals / vals.sum()
    return float(-np.sum(p * np.log(p)))


def field_state(simpat, indec, stability, polarization):
    if stability >= 0.70 and max(simpat, 1 - simpat - indec) >= 0.50:
        return "CONSOLIDACION"
    if indec >= 0.35 or polarization >= 0.55 or stability < 0.45:
        return "DISPUTA_ABIERTA"
    return "CONTENCION"


def ensure_lat_lon(df: pd.DataFrame, seed=42) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)
    if "alcaldia" not in out:
        out["alcaldia"] = rng.choice(ALCALDIAS, len(out))
    out["alcaldia"] = out["alcaldia"].map(clean_alcaldia)
    lats, lons = [], []
    for a in out["alcaldia"]:
        lat, lon = ALCALDIA_COORDS.get(a, (19.35, -99.15))
        lats.append(lat + rng.normal(0, 0.007))
        lons.append(lon + rng.normal(0, 0.007))
    if "lat" not in out:
        out["lat"] = lats
    else:
        out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
        out["lat"] = out["lat"].fillna(pd.Series(lats, index=out.index))
    if "lon" not in out:
        out["lon"] = lons
    else:
        out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
        out["lon"] = out["lon"].fillna(pd.Series(lons, index=out.index))
    return out


def normalize_base(df: pd.DataFrame, seed=42) -> pd.DataFrame:
    out = df.copy()
    out.columns = [norm_col(c) for c in out.columns]
    aliases = {
        "seccion_electoral": "seccion",
        "seccion_id": "seccion",
        "section": "seccion",
        "alcaldía": "alcaldia",
        "delegacion": "alcaldia",
        "territory_id": "territorial_unit_id",
        "id_territorial": "territorial_unit_id",
        "id": "territorial_unit_id",
        "utm_id": "utm",
        "unidad_territorial_media": "utm",
        "manzana_id": "manzana",
        "manzana_geo": "manzana",
    }
    for a, b in aliases.items():
        if a in out.columns and b not in out.columns:
            out[b] = out[a]

    if "territorial_unit_id" not in out:
        if "seccion" in out:
            out["territorial_unit_id"] = "CDMX-SEC-" + out["seccion"].astype(str)
        else:
            out["territorial_unit_id"] = [f"CDMX-SEC-{i+1:05d}" for i in range(len(out))]

    if "seccion" not in out:
        out["seccion"] = out["territorial_unit_id"].astype(str)

    # Jerarquía oficial de trabajo de SITER-CAE: CDMX → Alcaldía → UTM → Sección → Manzana.
    # UTM/manzana se conservan vacías cuando el archivo real no las contiene; nunca se inventan.
    if "utm" not in out:
        out["utm"] = ""
    if "manzana" not in out:
        out["manzana"] = ""

    if "alcaldia" not in out:
        out["alcaldia"] = "NO_ESPECIFICADA"

    out["alcaldia"] = out["alcaldia"].map(clean_alcaldia)

    defaults = {
        "opinion_continua": 0.0,
        "capital_social": 0.5,
        "acceso_informacion": 0.5,
        "influencia_liderazgo": 0.5,
        "arraigo": 0.5,
        "nivel_movilizacion": 0.5,
        "desconfianza": 0.5,
        "exposicion_problema": 0.5,
        "temperatura": 0.5,
        "resistencia_institucional": 0.5,
        "prioridad_problema": 0.5,
        "poblacion": 1000.0,
    }
    for c, v in defaults.items():
        if c not in out:
            out[c] = v

    for c, default_value in defaults.items():
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(default_value)

    out["opinion_continua"] = out["opinion_continua"].clip(-1, 1)
    out = ensure_lat_lon(out, seed=seed)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------
# DataProvider
# ---------------------------------------------------------------------
class DataProvider:
    MODES = {
        "REAL · CSV + SHP/GeoJSON": "real",
        "DUMMY · prueba mínima": "dummy",
        "SINTÉTICO COHERENTE · derivado de real": "coherent",
        "SINTÉTICO PURO · generativo": "pure",
        "SINTÉTICO CALIBRACIÓN · pruebas controladas": "calib",
    }

    def __init__(self, seed=42):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def dummy(self, n=48):
        rng = self.rng
        rows = []
        for i in range(n):
            alc = ALCALDIAS[i % len(ALCALDIAS)]
            lat, lon = ALCALDIA_COORDS[alc]
            op = [-0.7, -0.2, 0.0, 0.2, 0.7][i % 5] + rng.normal(0, 0.04)
            rows.append({
                "territorial_unit_id": f"DUMMY-{i+1:04d}",
                "alcaldia": alc,
                "seccion": str(1000+i),
                "opinion_continua": float(np.clip(op, -1, 1)),
                "capital_social": float(0.35 + 0.45*rng.random()),
                "acceso_informacion": float(0.35 + 0.45*rng.random()),
                "influencia_liderazgo": float(0.25 + 0.65*rng.random()),
                "arraigo": float(0.3 + 0.6*rng.random()),
                "nivel_movilizacion": float(0.25 + 0.7*rng.random()),
                "desconfianza": float(0.15 + 0.55*rng.random()),
                "exposicion_problema": float(0.2 + 0.7*rng.random()),
                "prioridad_problema": float(0.2 + 0.8*rng.random()),
                "resistencia_institucional": float(0.2 + 0.6*rng.random()),
            })
        return normalize_base(pd.DataFrame(rows), self.seed), {"mode": "dummy", "source": "generated"}

    def pure(self, n=300):
        rng = self.rng
        rows = []
        for i in range(n):
            alc = str(rng.choice(ALCALDIAS))
            lat, lon = ALCALDIA_COORDS[alc]
            social = rng.beta(4, 4)
            info = rng.beta(4, 4)
            lead = rng.beta(4, 4)
            arraigo = rng.beta(4, 4)
            mov = rng.beta(4, 4)
            distrust = rng.beta(3, 5)
            exposure = rng.beta(4, 4)
            skill = np.clip(
                .30*social + .25*info + .25*lead + .10*arraigo + .10*mov - .15*distrust,
                0, 1
            )
            op = np.clip(0.7*(2*social-1) + 0.25*(2*lead-1) + rng.normal(0, .12), -1, 1)
            rows.append({
                "territorial_unit_id": f"SYN-{i+1:05d}",
                "alcaldia": alc,
                "seccion": str(2000+i),
                "lat": lat + rng.normal(0, .009),
                "lon": lon + rng.normal(0, .009),
                "opinion_continua": op,
                "capital_social": social,
                "acceso_informacion": info,
                "influencia_liderazgo": lead,
                "arraigo": arraigo,
                "nivel_movilizacion": mov,
                "desconfianza": distrust,
                "exposicion_problema": exposure,
                "prioridad_problema": exposure,
                "resistencia_institucional": rng.beta(2, 5),
                "poblacion": max(100, rng.lognormal(7.0, .45)),
                "saf_skill": skill,
            })
        return normalize_base(pd.DataFrame(rows), self.seed), {"mode": "synthetic_pure", "source": "generative"}

    def calibration(self, n=240, scenario="balance"):
        rng = self.rng
        rows = []
        scenarios = {
            "balance": lambda i: rng.normal(0, .12),
            "polarizacion": lambda i: (0.75 if i % 2 == 0 else -0.75) + rng.normal(0, .05),
            "fragmentacion": lambda i: [-.8, -.3, .0, .35, .8][i % 5] + rng.normal(0, .03),
            "consenso": lambda i: .55 + rng.normal(0, .05),
            "resistencia_alta": lambda i: rng.normal(0, .20),
            "red_hub": lambda i: rng.normal(0, .10),
        }
        fn = scenarios.get(scenario, scenarios["balance"])
        for i in range(n):
            alc = ALCALDIAS[i % len(ALCALDIAS)]
            lat, lon = ALCALDIA_COORDS[alc]
            op = float(np.clip(fn(i), -1, 1))
            rows.append({
                "territorial_unit_id": f"CAL-{scenario[:4].upper()}-{i+1:05d}",
                "alcaldia": alc,
                "seccion": str(3000+i),
                "lat": lat + rng.normal(0, .006),
                "lon": lon + rng.normal(0, .006),
                "opinion_continua": op,
                "capital_social": .55,
                "acceso_informacion": .55,
                "influencia_liderazgo": .55,
                "arraigo": .55,
                "nivel_movilizacion": .55,
                "desconfianza": .35,
                "exposicion_problema": .60,
                "prioridad_problema": .60,
                "resistencia_institucional": .80 if scenario == "resistencia_alta" else .35,
                "poblacion": 1000,
            })
        return normalize_base(pd.DataFrame(rows), self.seed), {
            "mode": "synthetic_calibration",
            "scenario": scenario,
            "source": "controlled_test"
        }

    def coherent_from_real(self, real_df, n=None):
        """Genera datos sintéticos sin copiar filas:
        - categorías por frecuencia
        - variables numéricas por distribución empírica
        - correlación aproximada mediante cópula gaussiana construida con
          correlación de rangos y eigen-decomposition (sin SciPy).
        """
        base = normalize_base(real_df, self.seed)
        n = int(n or len(base))
        rng = self.rng

        cats = {}
        for c in ["alcaldia"]:
            p = base[c].value_counts(normalize=True)
            cats[c] = rng.choice(p.index.to_numpy(), n, p=p.to_numpy())

        numeric = [c for c in TRAIT_COLS + ["opinion_continua", "temperatura",
                                            "resistencia_institucional", "prioridad_problema"]
                   if c in base.columns]
        X = base[numeric].apply(pd.to_numeric, errors="coerce").fillna(base[numeric].median())
        # Estandarización por rangos: aproxima correlación Spearman.
        ranks = X.rank(pct=True).to_numpy()
        R = np.corrcoef(ranks.T)
        R = np.nan_to_num(R, nan=0.0)
        R = (R + R.T) / 2
        np.fill_diagonal(R, 1.0)
        vals, vecs = np.linalg.eigh(R)
        vals = np.clip(vals, 1e-6, None)
        L = vecs @ np.diag(np.sqrt(vals))
        Z = rng.normal(size=(n, len(numeric))) @ L.T
        U = 0.5 * (1 + np.vectorize(math.erf)(Z / math.sqrt(2)))
        out = pd.DataFrame({"alcaldia": cats["alcaldia"]})
        for j, c in enumerate(numeric):
            arr = np.sort(X[c].to_numpy(dtype=float))
            q = np.clip(U[:, j], 0, 1)
            idx = np.minimum((q * (len(arr)-1)).astype(int), len(arr)-1)
            out[c] = arr[idx]

        # ID nuevo; nunca se reutilizan IDs del real.
        out["territorial_unit_id"] = [f"COH-{i+1:05d}" for i in range(n)]
        out["seccion"] = [f"C{i+1:05d}" for i in range(n)]
        out["poblacion"] = float(base["poblacion"].median()) if "poblacion" in base else 1000
        out = ensure_lat_lon(out, self.seed)
        meta = {
            "mode": "synthetic_coherent",
            "derived_from_real": True,
            "real_n": len(base),
            "synthetic_n": len(out),
            "numeric_basis": numeric,
            "source_hash": sha256_obj({
                "columns": list(base.columns),
                "shape": base.shape,
                "alcaldia_distribution": base["alcaldia"].value_counts().to_dict(),
            })
        }
        return normalize_base(out, self.seed), meta

    def real(self, base_csv, electoral_csv=None, socio_csv=None):
        if base_csv is None:
            raise ValueError("El modo REAL requiere el CSV base territorial.")
        base = pd.read_csv(base_csv)
        base = normalize_base(base, self.seed)

        if electoral_csv is not None:
            elec = pd.read_csv(electoral_csv)
            elec.columns = [norm_col(c) for c in elec.columns]
            if "seccion" in elec:
                elec["seccion"] = elec["seccion"].astype(str)
                # Si existen votos por partido/agrupación, se resume a shares
                if "votos" in elec.columns:
                    group = elec.groupby("seccion")["votos"].sum().rename("votos_total")
                    mx = elec.groupby("seccion")["votos"].max().rename("votos_max")
                    stats = pd.concat([group, mx], axis=1)
                    stats["share_max"] = (stats["votos_max"] / stats["votos_total"]).fillna(0)
                    base["seccion"] = base["seccion"].astype(str)
                    base = base.merge(stats[["share_max"]], left_on="seccion", right_index=True, how="left")
                    base["opinion_continua"] = (2*base["share_max"].fillna(.33)-1).clip(-1,1)

        if socio_csv is not None:
            socio = pd.read_csv(socio_csv)
            socio.columns = [norm_col(c) for c in socio.columns]
            if "seccion" in socio.columns:
                socio["seccion"] = socio["seccion"].astype(str)
                base["seccion"] = base["seccion"].astype(str)
                keep = [c for c in socio.columns if c != "alcaldia"]
                base = base.merge(socio[keep], on="seccion", how="left", suffixes=("", "_socio"))

        base = normalize_base(base, self.seed)
        meta = {
            "mode": "real",
            "source": "csv",
            "n": len(base),
            "columns": list(base.columns),
            "source_hash": sha256_obj({
                "shape": base.shape,
                "columns": list(base.columns),
            }),
        }
        return base, meta


# ---------------------------------------------------------------------
# GIS loader
# ---------------------------------------------------------------------
def read_vector_upload(uploaded) -> Optional["gpd.GeoDataFrame"]:
    if uploaded is None or not HAS_GIS:
        return None
    suffix = Path(uploaded.name).suffix.lower()
    data = uploaded.getvalue()
    tmpdir = Path(tempfile.mkdtemp(prefix="siter_gis_"))
    if suffix == ".zip":
        zpath = tmpdir / uploaded.name
        zpath.write_bytes(data)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmpdir / "unzipped")
        shps = list((tmpdir / "unzipped").rglob("*.shp"))
        if not shps:
            raise ValueError("El ZIP no contiene un .shp.")
        return gpd.read_file(shps[0])
    if suffix in [".geojson", ".json", ".gpkg", ".shp"]:
        p = tmpdir / uploaded.name
        p.write_bytes(data)
        return gpd.read_file(p)
    raise ValueError("Formato GIS no soportado. Usa ZIP de Shapefile, GeoJSON o GPKG.")


def normalize_gdf(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    g = gdf.copy()
    g.columns = [norm_col(c) for c in g.columns]
    if g.crs is None:
        # Se asume WGS84 solo cuando el usuario no proporcionó CRS.
        g = g.set_crs(4326, allow_override=True)
    if g.crs.to_epsg() != 4326:
        g = g.to_crs(4326)
    if "alcaldia" in g:
        g["alcaldia"] = g["alcaldia"].map(clean_alcaldia)
    return g



def make_dummy_cdmx_gis():
    """Polígonos aproximados por alcaldía (NO oficiales). Solo prueba de UI/mapa.

    Funciona con geopandas si está instalado; si no, devuelve GeoJSON dict
    para pydeck.
    """
    # Mitad del lado en grados (aprox). Solo visualización de laboratorio.
    half = {
        "CUAUHTEMOC": (0.025, 0.025),
        "BENITO JUAREZ": (0.022, 0.022),
        "MIGUEL HIDALGO": (0.030, 0.028),
        "COYOACAN": (0.035, 0.032),
        "IZTAPALAPA": (0.045, 0.040),
        "GUSTAVO A MADERO": (0.040, 0.038),
        "ALVARO OBREGON": (0.040, 0.035),
        "TLALPAN": (0.050, 0.045),
        "XOCHIMILCO": (0.035, 0.035),
        "VENUSTIANO CARRANZA": (0.025, 0.022),
        "AZCAPOTZALCO": (0.028, 0.025),
        "IZTACALCO": (0.020, 0.018),
        "CUAJIMALPA DE MORELOS": (0.035, 0.030),
        "LA MAGDALENA CONTRERAS": (0.030, 0.028),
        "TLAHUAC": (0.035, 0.032),
        "MILPA ALTA": (0.045, 0.040),
    }
    features = []
    for i, (name, (lat, lon)) in enumerate(ALCALDIA_COORDS.items(), 1):
        dlat, dlon = half.get(name, (0.03, 0.03))
        ring = [
            [lon - dlon, lat - dlat],
            [lon + dlon, lat - dlat],
            [lon + dlon, lat + dlat],
            [lon - dlon, lat + dlat],
            [lon - dlon, lat - dlat],
        ]
        features.append({
            "type": "Feature",
            "properties": {
                "alcaldia": name,
                "seccion": f"DUM-{i:03d}",
                "territorial_unit_id": f"DUMMY-GIS-{i:04d}",
                "note": "poligono_aproximado_prueba_no_oficial",
            },
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })
    gj = {
        "type": "FeatureCollection",
        "name": "CDMX_alcaldias_dummy_test",
        "features": features,
    }
    if HAS_GIS:
        gdf = gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326")
        gdf.columns = [norm_col(c) for c in gdf.columns]
        if "alcaldia" in gdf.columns:
            gdf["alcaldia"] = gdf["alcaldia"].map(clean_alcaldia)
        return gdf, gj
    return None, gj


def join_base_to_geometry(df, gdf):
    if gdf is None:
        return None
    g = normalize_gdf(gdf)
    # Preferir ID territorial; si no, seccion.
    for key in ["territorial_unit_id", "seccion"]:
        if key in df.columns and key in g.columns:
            left = df.copy()
            left[key] = left[key].astype(str)
            g[key] = g[key].astype(str)
            merged = g.merge(
                left.drop_duplicates(key),
                on=key,
                how="left",
                suffixes=("_gis", "")
            )
            return merged
    # Si no hay llave común, conservar geometría y hacer centroides; el usuario
    # puede trabajar con la capa GIS aunque el CSV no se haya unido.
    return g


# ---------------------------------------------------------------------
# Behaviors Mesa
# ---------------------------------------------------------------------
class Behavior:
    name = "base"
    def __init__(self, params=None):
        self.params = params or {}
    def step_agent(self, agent):
        raise NotImplementedError


class VoterBehavior(Behavior):
    name = "Voter / difusión local"
    def step_agent(self, agent):
        ns = agent.get_neighbors()
        if not ns:
            return
        other = agent.model.random.choice(ns)
        beta = float(self.params.get("beta", 1.2))
        prob = min(0.90, max(0.0, other.influencia * beta))
        if agent.model.random.random() < prob:
            agent.next_opinion = clamp(agent.opinion + .20*(other.opinion-agent.opinion))


class DeffuantBehavior(Behavior):
    name = "Deffuant-Weisbuch"
    def step_agent(self, agent):
        ns = agent.get_neighbors()
        if not ns:
            agent.next_opinion = agent.opinion
            return
        other = agent.model.random.choice(ns)
        eps = float(self.params.get("epsilon", .40))
        mu = float(self.params.get("mu", .30))
        rep = float(self.params.get("epsilon_repulsion", .80))
        d = abs(agent.opinion-other.opinion)
        if d <= eps:
            agent.next_opinion = clamp(agent.opinion + mu*(other.opinion-agent.opinion))
        elif d >= rep:
            agent.next_opinion = clamp(agent.opinion - .5*mu*(other.opinion-agent.opinion))
        else:
            agent.next_opinion = agent.opinion


class SAFBehavior(Behavior):
    name = "ABM-SAF"
    def step_agent(self, agent):
        ns = agent.get_neighbors()
        if not ns:
            agent.next_opinion = agent.opinion
            return
        mean_n = float(np.mean([x.opinion for x in ns]))
        coupling = float(self.params.get("coupling", .20))
        field_pressure = float(self.params.get("field_pressure", .08))
        # Amplificación opcional por capa organizacional agregada (no individual).
        org_boost = 1.0 + 0.5 * float(getattr(agent, "broker_density", 0.0))
        org_boost *= 1.0 + 0.3 * float(getattr(agent, "org_mobilization", 0.0))
        coupling_eff = coupling * org_boost
        fatigue_penalty = max(0.0, 1.0 - agent.fatiga)
        agent.next_opinion = clamp(
            agent.opinion
            + coupling_eff * agent.saf_skill * fatigue_penalty * (mean_n - agent.opinion)
            + field_pressure * agent.exposure * (-agent.opinion)
        )


BEHAVIORS = {
    "Voter / difusión local": VoterBehavior,
    "Deffuant-Weisbuch": DeffuantBehavior,
    "ABM-SAF": SAFBehavior,
}


def clamp(x, lo=-1, hi=1):
    return float(max(lo, min(hi, float(x))))


# ---------------------------------------------------------------------
# Mesa model
# ---------------------------------------------------------------------
if MESA_OK:
    class SeccionAgent(Agent):
        def __init__(self, model, row):
            super().__init__(model)
            self.territorial_unit_id = str(row["territorial_unit_id"])
            self.alcaldia = str(row["alcaldia"])
            self.seccion = str(row["seccion"])
            self.lat = float(row["lat"])
            self.lon = float(row["lon"])
            self.opinion = clamp(row.get("opinion_continua", 0))
            self.next_opinion = self.opinion
            self.capital_social = float(row.get("capital_social", .5))
            self.acceso_informacion = float(row.get("acceso_informacion", .5))
            self.influencia_liderazgo = float(row.get("influencia_liderazgo", .5))
            self.arraigo = float(row.get("arraigo", .5))
            self.nivel_movilizacion = float(row.get("nivel_movilizacion", .5))
            self.desconfianza = float(row.get("desconfianza", .5))
            self.exposure = float(row.get("exposicion_problema", .5))
            self.resistencia_institucional = float(row.get("resistencia_institucional", .5))
            self.prioridad_problema = float(row.get("prioridad_problema", .5))
            self.fatiga = 0.0
            self.influencia = 0.0
            self.es_broker = False
            self.neighbor_ids = []
            # Capa organización agregada (defaults seguros = 0 → sin efecto)
            self.broker_density = float(row.get("broker_density", 0.0) or 0.0)
            self.org_mobilization = float(row.get("org_mobilization", 0.0) or 0.0)
            self.aceptacion_mensaje = float(row.get("aceptacion_mensaje", 0.0) or 0.0)
            self.org_reliability = float(row.get("org_reliability", 0.0) or 0.0)
            self.n_militantes_obs = float(row.get("n_militantes_obs", 0.0) or 0.0)
            self.es_lider = int(float(row.get("es_lider", 0) or 0))
            self.poblacion = float(row.get("poblacion", 1000.0) or 1000.0)

        @property
        def spin(self):
            if self.opinion > .25:
                return 1
            if self.opinion < -.25:
                return -1
            return 0

        @property
        def saf_skill(self):
            return clamp(
                .30*self.capital_social
                + .25*self.acceso_informacion
                + .25*self.influencia_liderazgo
                + .10*self.arraigo
                + .10*self.nivel_movilizacion
                - .15*self.desconfianza,
                0, 1
            )

        def get_neighbors(self):
            return [self.model.agent_by_uid[x] for x in self.neighbor_ids if x in self.model.agent_by_uid]

        def step(self):
            self.model.behavior.step_agent(self)

        def advance(self):
            self.opinion = clamp(self.next_opinion)

    class BrokerAgent(SeccionAgent):
        def __init__(self, model, row):
            super().__init__(model, row)
            self.es_broker = True

    class SITERModel(Model):
        def __init__(self, df, behavior_name="ABM-SAF", seed=42,
                     p_intra=.06, p_inter=.015, params=None):
            super().__init__(seed=int(seed))
            self.seed_value = int(seed)
            self.df = df.reset_index(drop=True).copy()
            self.behavior_name = behavior_name
            self.behavior = BEHAVIORS[behavior_name](params or {})
            self.p_intra = float(p_intra)
            self.p_inter = float(p_inter)
            self.params = params or {}
            self.G = nx.Graph()
            self.agent_by_uid = {}
            self._build_agents()
            self._build_network()
            self._collect_history()

        def _build_agents(self):
            for _, row in self.df.iterrows():
                a = SeccionAgent(self, row)
                self.agent_by_uid[str(a.unique_id)] = a
                self.G.add_node(str(a.unique_id), territorial_unit_id=a.territorial_unit_id,
                                alcaldia=a.alcaldia)

        def _build_network(self):
            rng = np.random.default_rng(self.seed_value)
            agents = list(self.agent_by_uid.values())
            use_geo = bool(self.params.get("use_geo_network", False))
            geo_km = float(self.params.get("geo_radius_km", 3.0))
            if use_geo and len(agents) > 1:
                # Red por proximidad geográfica + boost intra-alcaldía.
                for i, a in enumerate(agents):
                    for b in agents[i+1:]:
                        d_m = haversine_m(a.lat, a.lon, b.lat, b.lon)
                        if d_m <= geo_km * 1000.0:
                            self.G.add_edge(str(a.unique_id), str(b.unique_id))
                        else:
                            p = self.p_intra if a.alcaldia == b.alcaldia else self.p_inter
                            # Enlace largo raro para no aislar alcaldías lejanas.
                            if rng.random() < p * 0.25:
                                self.G.add_edge(str(a.unique_id), str(b.unique_id))
            else:
                for i, a in enumerate(agents):
                    for b in agents[i+1:]:
                        p = self.p_intra if a.alcaldia == b.alcaldia else self.p_inter
                        if rng.random() < p:
                            self.G.add_edge(str(a.unique_id), str(b.unique_id))
            # Evitar red completamente vacía sin falsear conectividad excesiva.
            if len(agents) > 1 and self.G.number_of_edges() == 0:
                for i in range(len(agents)-1):
                    self.G.add_edge(str(agents[i].unique_id), str(agents[i+1].unique_id))

            degree = dict(self.G.degree())
            maxd = max(degree.values(), default=1)
            for a in agents:
                a.neighbor_ids = list(self.G.neighbors(str(a.unique_id)))
                a.influencia = degree.get(str(a.unique_id), 0) / maxd if maxd else 0

        def step(self):
            # Actualización simultánea.
            self.agents.do("step")
            self.agents.do("advance")
            for a in self.agents:
                a.fatiga *= .95
            self._collect_history()

        def _collect_history(self):
            self.history.append(self.metrics())

        @property
        def history(self):
            if not hasattr(self, "_history"):
                self._history = []
            return self._history

        def metrics(self):
            n = len(self.agents)
            return {
                "step": int(self.steps),
                "SIMPATIZANTE": self.count_spin(1),
                "OPOSITOR": self.count_spin(-1),
                "INDECISO": self.count_spin(0),
                "Gini": self.compute_gini(),
                "Polarizacion": self.compute_polarization(),
                "MeanOpinion": self.mean_opinion(),
                "edges": self.G.number_of_edges(),
                "density": nx.density(self.G) if n > 1 else 0,
            }

        def count_spin(self, v):
            n = len(self.agents)
            return 0 if n == 0 else sum(a.spin == v for a in self.agents) / n

        def mean_opinion(self):
            return float(np.mean([a.opinion for a in self.agents])) if len(self.agents) else 0

        def compute_polarization(self):
            vals = np.array([a.opinion for a in self.agents], dtype=float)
            return float(np.std(vals)) if len(vals) else 0

        def compute_gini(self):
            return gini([abs(a.opinion) for a in self.agents])

        def agent_dataframe(self):
            return pd.DataFrame([{
                "agent_id": str(a.unique_id),
                "territorial_unit_id": a.territorial_unit_id,
                "alcaldia": a.alcaldia,
                "seccion": a.seccion,
                "lat": a.lat,
                "lon": a.lon,
                "opinion": a.opinion,
                "spin": a.spin,
                "intencion": STATE_LABEL[a.spin],
                "capital_social": a.capital_social,
                "influencia": a.influencia,
                "saf_skill": a.saf_skill,
                "resistencia_institucional": a.resistencia_institucional,
                "prioridad_problema": a.prioridad_problema,
                "fatiga": a.fatiga,
                "broker": a.es_broker,
                "broker_density": a.broker_density,
                "org_mobilization": a.org_mobilization,
                "aceptacion_mensaje": a.aceptacion_mensaje,
                "org_reliability": a.org_reliability,
                "n_militantes_obs": a.n_militantes_obs,
                "es_lider": int(getattr(a, "es_lider", 0) or 0),
                "poblacion": float(getattr(a, "poblacion", 1000.0) or 1000.0),
                "temperatura": float(getattr(a, "exposure", 0.5) or 0.5),
                "nivel_movilizacion": float(getattr(a, "nivel_movilizacion", 0.5) or 0.5),
            } for a in self.agents])

        def insert_broker(self, target_uid=None):
            if target_uid and target_uid in self.agent_by_uid:
                base = self.agent_by_uid[target_uid]
            else:
                base = max(self.agents, key=lambda x: x.influencia)
            row = {
                "territorial_unit_id": f"{base.territorial_unit_id}-BRK",
                "alcaldia": base.alcaldia,
                "seccion": base.seccion,
                "lat": base.lat,
                "lon": base.lon,
                "opinion_continua": base.opinion,
                "capital_social": .90,
                "acceso_informacion": .90,
                "influencia_liderazgo": .90,
                "arraigo": .85,
                "nivel_movilizacion": .85,
                "desconfianza": .10,
                "exposicion_problema": base.exposure,
                "resistencia_institucional": base.resistencia_institucional,
                "prioridad_problema": base.prioridad_problema,
            }
            b = BrokerAgent(self, row)
            self.agent_by_uid[str(b.unique_id)] = b
            self.G.add_node(str(b.unique_id), territorial_unit_id=b.territorial_unit_id,
                            alcaldia=b.alcaldia)
            candidates = sorted(
                [a for a in self.agents if a is not b and a.alcaldia == base.alcaldia],
                key=lambda x: x.influencia, reverse=True
            )[:8]
            for a in candidates:
                self.G.add_edge(str(b.unique_id), str(a.unique_id))
            b.neighbor_ids = [str(a.unique_id) for a in candidates]
            for a in candidates:
                if str(b.unique_id) not in a.neighbor_ids:
                    a.neighbor_ids.append(str(b.unique_id))
            b.influencia = 1.0
            return b



# ---------------------------------------------------------------------
# Capa de organización / militancia AGREGADA (sin PII)
# ---------------------------------------------------------------------
class OrganizationLayer:
    """Fusiona rasgos organizacionales agregados por territorial_unit_id.

    Nunca acepta identificadores personales. Celdas con n_militantes_obs
    por debajo de min_n se anulan (supresión de celdas pequeñas).
    """

    REQUIRED = {"territorial_unit_id"}
    OPTIONAL = set(ORG_COLS)

    @staticmethod
    def load(uploaded, min_n: int = 5) -> pd.DataFrame:
        if uploaded is None:
            return pd.DataFrame()
        df = pd.read_csv(uploaded)
        df.columns = [norm_col(c) for c in df.columns]
        if "territorial_unit_id" not in df.columns:
            if "seccion" in df.columns:
                df["territorial_unit_id"] = "CDMX-SEC-" + df["seccion"].astype(str)
            else:
                raise ValueError(
                    "CSV de organización requiere territorial_unit_id o seccion."
                )
        df["territorial_unit_id"] = df["territorial_unit_id"].astype(str)
        for c in ORG_COLS:
            if c not in df.columns:
                df[c] = 0.0
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        # Supresión de celdas pequeñas (riesgo de reidentificación)
        mask = df["n_militantes_obs"] < float(min_n)
        for c in ["broker_density", "org_mobilization", "aceptacion_mensaje",
                  "asistencia_evento_rate"]:
            df.loc[mask, c] = 0.0
        df.loc[mask, "org_reliability"] = 0.0
        # Clip a [0,1] en tasas
        for c in ["broker_density", "org_mobilization", "aceptacion_mensaje",
                  "asistencia_evento_rate", "org_reliability"]:
            df[c] = df[c].clip(0, 1)
        return df[["territorial_unit_id"] + ORG_COLS].drop_duplicates("territorial_unit_id")

    @staticmethod
    def merge_into_base(base: pd.DataFrame, org: pd.DataFrame) -> pd.DataFrame:
        if org is None or org.empty:
            out = base.copy()
            for c in ORG_COLS:
                if c not in out.columns:
                    out[c] = 0.0
            return out
        out = base.copy()
        out["territorial_unit_id"] = out["territorial_unit_id"].astype(str)
        org = org.copy()
        org["territorial_unit_id"] = org["territorial_unit_id"].astype(str)
        out = out.drop(columns=[c for c in ORG_COLS if c in out.columns], errors="ignore")
        out = out.merge(org, on="territorial_unit_id", how="left")
        for c in ORG_COLS:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        return out


# ---------------------------------------------------------------------
# Estadística territorial
# ---------------------------------------------------------------------

def ensure_agent_frame_columns(adf: pd.DataFrame) -> pd.DataFrame:
    """Garantiza columnas usadas por downscaling / QuestionEngine tras agent_dataframe."""
    if adf is None or adf.empty:
        return adf
    out = adf.copy()
    if "poblacion" not in out.columns:
        out["poblacion"] = 1000.0
    else:
        out["poblacion"] = pd.to_numeric(out["poblacion"], errors="coerce").fillna(1000.0)
    if "prioridad_problema" not in out.columns:
        out["prioridad_problema"] = 0.5
    if "temperatura" not in out.columns:
        out["temperatura"] = 0.5
    if "nivel_movilizacion" not in out.columns:
        out["nivel_movilizacion"] = 0.5
    if "es_lider" not in out.columns:
        out["es_lider"] = 0
    if "saf_skill" not in out.columns and {"capital_social", "influencia_liderazgo"}.issubset(out.columns):
        out["saf_skill"] = (
            0.3 * out.get("capital_social", 0.5)
            + 0.25 * out.get("acceso_informacion", 0.5)
            + 0.25 * out.get("influencia_liderazgo", 0.5)
        ).clip(0, 1)
    return out

class TerritorialAnalytics:
    @staticmethod
    def table(adf):
        rows = []
        for alc, sub in adf.groupby("alcaldia", dropna=False):
            counts = sub["intencion"].value_counts()
            n = len(sub)
            simpat = (sub["intencion"] == "SIMPATIZANTE").mean()
            opos = (sub["intencion"] == "OPOSITOR").mean()
            indec = (sub["intencion"] == "INDECISO").mean()
            ent = entropy_from_counts(counts.values)
            stab = 1 - (ent / math.log(3)) if n else 0
            pol = float(sub["opinion"].std()) if len(sub) > 1 else 0
            resistencia = float(sub["resistencia_institucional"].mean())
            prioridad = float(sub["prioridad_problema"].mean())
            campo = field_state(simpat, indec, stab, pol)
            rows.append({
                "alcaldia": alc,
                "n_unidades": n,
                "simpat_pct": simpat*100,
                "opos_pct": opos*100,
                "indec_pct": indec*100,
                "entropia": ent,
                "estabilidad": stab,
                "polarizacion_std": pol,
                "campo": campo,
                "influencia_prom": sub["influencia"].mean(),
                "saf_skill_prom": sub["saf_skill"].mean(),
                "resistencia_prom": resistencia,
                "prioridad_prom": prioridad,
                "poblacion_proxy": sub.get("poblacion", pd.Series([1000]*n)).sum()
                    if "poblacion" in sub else n*1000,
            })
        return pd.DataFrame(rows).sort_values("alcaldia").reset_index(drop=True)

    @staticmethod
    def network(model):
        G = model.G
        if len(G) == 0:
            return {}
        deg = np.array([d for _, d in G.degree()], dtype=float)
        clustering = nx.average_clustering(G) if len(G) > 2 else 0
        try:
            centralization = (max(deg) - deg.mean()) / max(len(G)-2, 1)
        except Exception:
            centralization = 0
        inf = deg / max(deg.max(), 1)
        return {
            "nodes": len(G),
            "edges": G.number_of_edges(),
            "density": nx.density(G) if len(G) > 1 else 0,
            "degree_mean": deg.mean() if len(deg) else 0,
            "degree_max": deg.max() if len(deg) else 0,
            "clustering": clustering,
            "centralization_degree": float(centralization),
            "gini_influence": gini(inf),
            "top1_share": float(np.max(inf)/np.sum(inf)) if np.sum(inf) else 0,
            "top10_share": float(np.sort(inf)[::-1][:max(1, int(.10*len(inf)))].sum()/np.sum(inf))
                if np.sum(inf) else 0,
        }

    @staticmethod
    def spof(model, top_k=10):
        base = model.metrics()["SIMPATIZANTE"]
        rows = []
        # Aproximación operacional: eliminar temporalmente nodos de mayor grado.
        candidates = sorted(model.G.degree(), key=lambda x: x[1], reverse=True)[:top_k]
        for uid, deg in candidates:
            a = model.agent_by_uid.get(str(uid))
            if a is None:
                continue
            G2 = model.G.copy()
            G2.remove_node(uid)
            # proxy: componentes / centralidad y estado agregado sin re-simular.
            loss_proxy = (deg / max(1, model.G.number_of_edges())) * base
            rows.append({
                "agent_id": str(uid),
                "territorial_unit_id": a.territorial_unit_id,
                "alcaldia": a.alcaldia,
                "grado": deg,
                "delta_proxy": -loss_proxy,
                "clasificacion": "CRITICO" if loss_proxy >= .08 else "MODERADO",
            })
        return pd.DataFrame(rows)

    @staticmethod
    def centers(adf, top_k=20):
        return adf.sort_values(["saf_skill", "influencia"], ascending=False).head(top_k).copy()


# ---------------------------------------------------------------------
# Brigadistas / GPS
# ---------------------------------------------------------------------
class FieldOperations:
    @staticmethod
    def load_gps(uploaded):
        if uploaded is None:
            return pd.DataFrame()
        name = uploaded.name.lower()
        data = uploaded.getvalue()
        if name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(data))
            df.columns = [norm_col(c) for c in df.columns]
            if "latitude" in df and "lat" not in df:
                df["lat"] = df["latitude"]
            if "longitude" in df and "lon" not in df:
                df["lon"] = df["longitude"]
            if "time" in df and "timestamp" not in df:
                df["timestamp"] = df["time"]
            return FieldOperations.normalize_gps(df)
        if name.endswith(".geojson") or name.endswith(".json"):
            if HAS_GIS:
                g = gpd.read_file(io.BytesIO(data))
                rows = []
                for _, r in g.iterrows():
                    if r.geometry is not None and r.geometry.geom_type == "Point":
                        rows.append({
                            "lat": r.geometry.y,
                            "lon": r.geometry.x,
                            "timestamp": r.get("timestamp", len(rows)),
                            "brigada": r.get("brigada", "B-01"),
                            "accuracy_m": r.get("accuracy_m", np.nan),
                        })
                return FieldOperations.normalize_gps(pd.DataFrame(rows))
        # GPX parsing básico sin librería adicional.
        if name.endswith(".gpx"):
            import xml.etree.ElementTree as ET
            root = ET.fromstring(data.decode("utf-8", errors="ignore"))
            rows = []
            for pt in root.iter():
                tag = pt.tag.split("}")[-1]
                if tag == "trkpt":
                    rows.append({
                        "lat": float(pt.attrib["lat"]),
                        "lon": float(pt.attrib["lon"]),
                        "timestamp": len(rows),
                        "brigada": "B-01",
                    })
            return FieldOperations.normalize_gps(pd.DataFrame(rows))
        raise ValueError("GPS: usa CSV, GeoJSON o GPX.")

    @staticmethod
    def normalize_gps(df):
        if df.empty:
            return df
        out = df.copy()
        out.columns = [norm_col(c) for c in out.columns]
        required = ["lat", "lon"]
        for c in required:
            if c not in out:
                raise ValueError(f"GPS requiere columna {c}.")
        if "brigada" not in out:
            out["brigada"] = "B-01"
        if "timestamp" not in out:
            out["timestamp"] = np.arange(len(out))
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
        out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
        out = out.dropna(subset=["lat", "lon"]).copy()
        return out

    @staticmethod
    def plan_from_territories(adf, n_brig=4, steps=8):
        """Plan territorial reproducible sobre centroides.
        No afirma ser ruta peatonal: es un plan de cobertura territorial.
        """
        if adf.empty:
            return pd.DataFrame()
        ordered = adf.sort_values(["alcaldia", "seccion"]).reset_index(drop=True)
        chunks = np.array_split(ordered, min(n_brig, len(ordered)))
        rows = []
        for bi, chunk in enumerate(chunks, 1):
            if chunk.empty:
                continue
            for j, (_, r) in enumerate(chunk.iterrows()):
                rows.append({
                    "brigada": f"B-{bi:02d}",
                    "orden": j,
                    "territorial_unit_id": r["territorial_unit_id"],
                    "alcaldia": r["alcaldia"],
                    "seccion": r["seccion"],
                    "lat": r["lat"],
                    "lon": r["lon"],
                })
        return pd.DataFrame(rows)

    @staticmethod
    def planned_length_m(plan):
        if plan.empty:
            return 0
        total = 0
        for _, g in plan.groupby("brigada"):
            g = g.sort_values("orden")
            total += path_distance_m(g)
        return total

    @staticmethod
    def coverage_report(plan, gps, buffer_m=25):
        if plan.empty:
            return pd.DataFrame()
        if gps.empty:
            rows = []
            for b, g in plan.groupby("brigada"):
                rows.append({
                    "brigada": b,
                    "km_plan": path_distance_m(g)/1000,
                    "km_gps": 0,
                    "cobertura_ruta_pct": 0,
                    "n_puntos_gps": 0,
                    "territorios_plan": g["territorial_unit_id"].nunique(),
                    "territorios_observados": 0,
                    "calidad": "SIN GPS",
                })
            return pd.DataFrame(rows)

        rows = []
        for b, p in plan.groupby("brigada"):
            gg = gps[gps["brigada"].astype(str) == str(b)]
            p = p.sort_values("orden")
            km_plan = path_distance_m(p)/1000
            km_gps = path_distance_m(gg)/1000
            coverage = 0.0
            if HAS_GIS and len(p) >= 2 and len(gg) >= 2:
                # Convertir aproximadamente a metros locales.
                mean_lat = float(pd.concat([p["lat"], gg["lat"]]).mean())
                sx = 111320 * math.cos(math.radians(mean_lat))
                sy = 110540
                pxy = [(float(x)*sx, float(y)*sy) for x, y in zip(p["lon"], p["lat"])]
                gxy = [(float(x)*sx, float(y)*sy) for x, y in zip(gg["lon"], gg["lat"])]
                lp = LineString(pxy)
                lg = LineString(gxy)
                if lp.length > 0:
                    coverage = min(1.0, lp.buffer(buffer_m).intersection(lg).length / lp.length)
            else:
                # Sin Shapely: aproximación por cercanía de puntos a puntos plan.
                if not gg.empty:
                    hit = 0
                    for _, gp in gg.iterrows():
                        ds = [
                            haversine_m(gp["lat"], gp["lon"], pr["lat"], pr["lon"])
                            for _, pr in p.iterrows()
                        ]
                        if min(ds) <= buffer_m:
                            hit += 1
                    coverage = hit / max(1, len(gg))

            # Territorios observados: si hay geometría se puede sustituir por
            # point-in-polygon; aquí usamos cercanía al centroide como fallback.
            observed = set()
            for _, gp in gg.iterrows():
                if p.empty:
                    continue
                dists = [
                    haversine_m(gp["lat"], gp["lon"], pr["lat"], pr["lon"])
                    for _, pr in p.iterrows()
                ]
                j = int(np.argmin(dists))
                if min(dists) <= buffer_m:
                    observed.add(str(p.iloc[j]["territorial_unit_id"]))

            quality = "ALTA" if coverage >= .80 else "MEDIA" if coverage >= .50 else "BAJA"
            rows.append({
                "brigada": b,
                "km_plan": km_plan,
                "km_gps": km_gps,
                "cobertura_ruta_pct": coverage*100,
                "n_puntos_gps": len(gg),
                "territorios_plan": p["territorial_unit_id"].nunique(),
                "territorios_observados": len(observed),
                "calidad": quality,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def performance(plan, gps, report):
        """Indicadores estadísticos de brigada: productividad, desviación, calidad GPS.
        La cobertura se refiere a la ruta/territorio planificado, no a distancia total recorrida.
        """
        if report.empty:
            return pd.DataFrame()
        out = report.copy()
        out["eficiencia_km"] = np.where(out["km_gps"] > 0, out["cobertura_ruta_pct"] / out["km_gps"], np.nan)
        out["desviacion_km"] = (out["km_gps"] - out["km_plan"]).abs()
        out["ratio_desviacion"] = np.where(out["km_plan"] > 0, out["desviacion_km"] / out["km_plan"], np.nan)
        out["cumplimiento_territorial_pct"] = np.where(out["territorios_plan"] > 0, 100*out["territorios_observados"] / out["territorios_plan"], 0)
        out["indice_productividad"] = 0.5*out["cobertura_ruta_pct"] + 0.5*out["cumplimiento_territorial_pct"]
        return out

    @staticmethod
    def summary(plan, gps, report):
        return {
            "km_plan_total": float(report["km_plan"].sum()) if not report.empty else 0,
            "km_gps_total": float(report["km_gps"].sum()) if not report.empty else 0,
            "cobertura_promedio_pct": float(report["cobertura_ruta_pct"].mean()) if not report.empty else 0,
            "brigadas_con_gps": int((report["n_puntos_gps"] > 0).sum()) if not report.empty else 0,
            "territorios_observados": int(report["territorios_observados"].sum()) if not report.empty else 0,
        }


# ---------------------------------------------------------------------
# Presupuesto / Monte Carlo / asignación
# ---------------------------------------------------------------------
class BudgetEngine:
    @staticmethod
    def workload(terr):
        if terr.empty:
            return pd.DataFrame()
        t = terr.copy()
        t["workload"] = (
            .40*t["prioridad_prom"]
            + .25*t["n_unidades"].rank(pct=True)
            + .20*t["resistencia_prom"]
            + .15*t["polarizacion_std"].rank(pct=True)
        )
        t["workload"] = t["workload"].clip(0, 1)
        t["km_est"] = np.maximum(.5, np.sqrt(t["n_unidades"]) * .20)
        return t

    @staticmethod
    def allocation(terr, budget, fixed_per_brigada=120, hour_cost=120,
                   hours_per_brigada=8, max_brigadas=20):
        t = BudgetEngine.workload(terr)
        if t.empty:
            return t
        rows = []
        remaining = float(budget)
        for _, r in t.sort_values("workload", ascending=False).iterrows():
            base_cost = fixed_per_brigada + hour_cost*hours_per_brigada
            if remaining >= base_cost and len(rows) < max_brigadas:
                brig = 1
                remaining -= base_cost
            else:
                brig = 0
            hours = brig*hours_per_brigada
            coverage = min(1.0, brig*.55 + .20*r["workload"])
            benefit = coverage * (0.5 + 0.5*r["prioridad_prom"])
            roi = benefit / max(base_cost if brig else 1, 1)
            rows.append({
                "alcaldia": r["alcaldia"],
                "workload": r["workload"],
                "brigadas": brig,
                "horas": hours,
                "costo": base_cost if brig else 0,
                "cobertura_est": coverage,
                "beneficio_operacional": benefit,
                "roi_operacional": roi,
                "presupuesto_restante_local": max(0, remaining),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def monte_carlo(terr, budget, reps=300, fixed_per_brigada=120,
                    hour_cost=120, hours_per_brigada=8, seed=42):
        if terr.empty:
            return pd.DataFrame()
        rng = np.random.default_rng(seed)
        workload = BudgetEngine.workload(terr)
        rows = []
        for sim in range(reps):
            # Incertidumbre en costo, cobertura y productividad.
            cost_mult = rng.lognormal(mean=0, sigma=.10)
            prod_mult = np.clip(rng.normal(1.0, .12), .60, 1.40)
            remaining = budget
            total_cov = 0
            spent = 0
            for _, r in workload.sort_values("workload", ascending=False).iterrows():
                c = (fixed_per_brigada + hour_cost*hours_per_brigada)*cost_mult
                if remaining < c:
                    continue
                remaining -= c
                spent += c
                total_cov += min(1, (.50 + .35*r["workload"])*prod_mult)
            denom = max(len(workload), 1)
            cov_pct = min(100, 100*total_cov/denom)
            rows.append({
                "sim": sim+1,
                "cobertura_pct": cov_pct,
                "gasto": spent,
                "remanente": remaining,
                "productividad_mult": prod_mult,
                "costo_mult": cost_mult,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def performance_table(terr, budget, fixed_per_brigada=120, hour_cost=120,
                          hours_per_brigada=8, max_brigadas=20):
        """Indicadores estadísticos operacionales para presupuesto.
        Incluye costo marginal, costo por unidad, productividad y elasticidad.
        """
        a = BudgetEngine.allocation(terr, budget, fixed_per_brigada, hour_cost,
                                    hours_per_brigada, max_brigadas)
        if a.empty:
            return a
        total_units = max(int(terr["n_unidades"].sum()), 1)
        covered_units = float((a["cobertura_est"] * terr.set_index("alcaldia").loc[a["alcaldia"], "n_unidades"].to_numpy()).sum())
        total_cost = float(a["costo"].sum())
        total_hours = float(a["horas"].sum())
        total_brigades = int(a["brigadas"].sum())
        out = a.copy()
        out["unidades_estimadas_cubiertas"] = (out["cobertura_est"] * terr.set_index("alcaldia").loc[out["alcaldia"], "n_unidades"].to_numpy())
        out["costo_por_unidad_cubierta"] = np.where(out["unidades_estimadas_cubiertas"] > 0, out["costo"] / out["unidades_estimadas_cubiertas"], np.nan)
        out["beneficio_por_1000"] = np.where(out["costo"] > 0, out["beneficio_operacional"] * 1000 / out["costo"], 0)
        out.attrs["summary"] = {
            "unidades_totales": total_units,
            "unidades_estimadas_cubiertas": covered_units,
            "cobertura_unidades_pct": 100 * covered_units / total_units,
            "costo_total": total_cost,
            "costo_por_unidad_cubierta": total_cost / max(covered_units, 1),
            "costo_por_brigada": total_cost / max(total_brigades, 1),
            "costo_por_hora": total_cost / max(total_hours, 1),
            "brigadas": total_brigades,
            "horas": total_hours,
        }
        return out

    @staticmethod
    def marginal_analysis(terr, budgets):
        s = BudgetEngine.sensitivity(terr, budgets)
        if s.empty:
            return s
        s = s.sort_values("presupuesto").reset_index(drop=True)
        s["delta_presupuesto"] = s["presupuesto"].diff().fillna(s["presupuesto"])
        s["delta_cobertura_pp"] = s["cobertura_est_pct"].diff().fillna(s["cobertura_est_pct"])
        s["cobertura_marginal_pp_por_1000"] = np.where(s["delta_presupuesto"] > 0, s["delta_cobertura_pp"] / (s["delta_presupuesto"] / 1000), 0)
        s["delta_beneficio"] = s["beneficio_operacional"].diff().fillna(s["beneficio_operacional"])
        s["beneficio_marginal_por_1000"] = np.where(s["delta_presupuesto"] > 0, s["delta_beneficio"] / (s["delta_presupuesto"] / 1000), 0)
        return s

    @staticmethod
    def confidence_summary(mc):
        if mc.empty:
            return {}
        x = mc["cobertura_pct"].to_numpy(dtype=float)
        return {
            "media": float(np.mean(x)),
            "mediana": float(np.median(x)),
            "p05": float(np.percentile(x, 5)),
            "p25": float(np.percentile(x, 25)),
            "p75": float(np.percentile(x, 75)),
            "p95": float(np.percentile(x, 95)),
            "prob_cobertura_80": float(np.mean(x >= 80)),
            "prob_cobertura_90": float(np.mean(x >= 90)),
        }

    @staticmethod
    def sensitivity(terr, budgets):
        rows = []
        for b in budgets:
            a = BudgetEngine.allocation(terr, b)
            rows.append({
                "presupuesto": b,
                "brigadas": int(a["brigadas"].sum()) if not a.empty else 0,
                "horas": int(a["horas"].sum()) if not a.empty else 0,
                "costo": float(a["costo"].sum()) if not a.empty else 0,
                "cobertura_est_pct": float(100*a["cobertura_est"].mean()) if not a.empty else 0,
                "beneficio_operacional": float(a["beneficio_operacional"].sum()) if not a.empty else 0,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Calibration diagnostics
# ---------------------------------------------------------------------
class CalibrationEngine:
    @staticmethod
    def compare(real_df, synthetic_df, columns=None):
        if real_df is None or synthetic_df is None or len(real_df) == 0 or len(synthetic_df) == 0:
            return pd.DataFrame()
        cols = columns or [
            c for c in TRAIT_COLS + [
                "opinion_continua", "temperatura",
                "resistencia_institucional", "prioridad_problema", "poblacion"
            ]
            if c in real_df.columns and c in synthetic_df.columns
        ]
        rows = []
        for c in cols:
            r = pd.to_numeric(real_df[c], errors="coerce").dropna().to_numpy(dtype=float)
            s = pd.to_numeric(synthetic_df[c], errors="coerce").dropna().to_numpy(dtype=float)
            if len(r) < 3 or len(s) < 3:
                continue
            qs = np.linspace(.05, .95, 19)
            rq = np.quantile(r, qs)
            sq = np.quantile(s, qs)
            scale = max(np.std(r), 1e-9)
            qdist = float(np.mean(np.abs(rq - sq)) / scale)
            mean_gap = float(abs(np.mean(r) - np.mean(s)) / scale)
            std_gap = float(abs(np.std(r) - np.std(s)) / scale)
            rows.append({
                "variable": c,
                "media_real": np.mean(r),
                "media_sint": np.mean(s),
                "std_real": np.std(r),
                "std_sint": np.std(s),
                "gap_media_norm": mean_gap,
                "gap_std_norm": std_gap,
                "dist_cuantiles_norm": qdist,
                "calibracion": "BUENA" if qdist < .20 else "MEDIA" if qdist < .40 else "REVISAR",
            })
        return pd.DataFrame(rows)

    @staticmethod
    def score(real_df, synthetic_df):
        cmp = CalibrationEngine.compare(real_df, synthetic_df)
        if cmp.empty:
            return None
        score = 100 * max(0.0, 1.0 - float(cmp["dist_cuantiles_norm"].mean()))
        return round(score, 1)

# ---------------------------------------------------------------------
# Advanced Downscaling / Territorial Inference
# ---------------------------------------------------------------------
class AdvancedDownscalingEngine:
    """
    Motores avanzados para DESAGREGACIÓN TERRITORIAL AGREGADA.

    Alcance:
    - IPF / raking de máxima entropía para repartir magnitudes agregadas.
    - Tomografía de red sobre flujos OD agregados.
    - Regresión ecológica agregada para variables operativas y políticas
      (voto, apoyo, intención, candidato, partido, preferencia, persuadible,
      probabilidad_voto). Permite desagregación estadística territorial y
      convierte la inferencia ecológica en un clasificador de personas a nivel
      de unidad territorial.
    - Actualización Bayesiana / Kalman de rasgos TERRITORIALES a partir de
      observaciones de campo explícitas, nunca de la mera cobertura GPS.
    - Suavizado espacial tipo MRF/CRF sobre unidades territoriales vecinas.
    """

    # Variables políticas explícitamente aceptadas para desagregación
    # territorial y clasificación de personas a partir de inferencia ecológica.
    ACCEPTED_POLITICAL = {
        "voto", "apoyo", "intencion", "intencion_voto", "candidato", "partido",
        "preferencia", "persuadible", "probabilidad_voto",
        "apoyo_candidato", "share_max", "simpatia_individual"
    }

    @staticmethod
    def max_entropy_ipf(prior_matrix, row_targets, col_targets,
                        max_iter=500, tol=1e-8, floor=1e-12):
        """IPF: minimiza KL(P||prior) sujeto a márgenes fila/columna."""
        prior = np.asarray(prior_matrix, dtype=float)
        rows = np.asarray(row_targets, dtype=float).reshape(-1)
        cols = np.asarray(col_targets, dtype=float).reshape(-1)
        if prior.ndim != 2:
            raise ValueError("prior_matrix debe ser 2D")
        n, m = prior.shape
        if len(rows) != n or len(cols) != m:
            raise ValueError("Los targets no coinciden con las dimensiones del prior")
        if np.any(rows < 0) or np.any(cols < 0):
            raise ValueError("Los targets no pueden ser negativos")
        if not np.isclose(rows.sum(), cols.sum(), rtol=1e-8, atol=1e-8):
            raise ValueError("Los totales fila y columna deben sumar lo mismo")

        P = np.maximum(prior, floor).copy()
        # Preservar ceros estructurales explícitos del prior.
        structural_zero = prior <= 0
        P[structural_zero] = floor

        err = np.inf
        for it in range(max_iter):
            rs = P.sum(axis=1)
            P *= np.divide(rows, np.maximum(rs, floor))[:, None]
            cs = P.sum(axis=0)
            P *= np.divide(cols, np.maximum(cs, floor))[None, :]
            row_err = np.max(np.abs(P.sum(axis=1) - rows)) if n else 0
            col_err = np.max(np.abs(P.sum(axis=0) - cols)) if m else 0
            err = max(row_err, col_err)
            if err < tol:
                break
        return P, {
            "iterations": it + 1,
            "max_margin_error": float(err),
            "prior_zeros": int(structural_zero.sum()),
            "converged": bool(err < tol),
            "total": float(rows.sum()),
        }

    @staticmethod
    def network_tomography(graph: nx.Graph, od_flows: dict,
                           regularization=0.05, iterations=1200, learning_rate=0.08):
        """
        Tomografía de red agregada.
        od_flows: {(origen, destino): flujo_observado}.
        Construye una matriz A de rutas mínimas y estima flujo de aristas
        mediante mínimos cuadrados regularizados con proyección no negativa.
        """
        if not isinstance(graph, nx.Graph) or graph.number_of_edges() == 0:
            return {"edge_fluxes": {}, "diagnostics": {"status": "empty_graph"}}
        edges = list(graph.edges())
        edge_idx = {tuple(sorted(e)): i for i, e in enumerate(edges)}
        rows, y, used = [], [], []
        for key, flow in (od_flows or {}).items():
            if not isinstance(key, (tuple, list)) or len(key) != 2:
                continue
            o, d = key
            if o not in graph or d not in graph or o == d:
                continue
            try:
                path = nx.shortest_path(graph, o, d)
            except nx.NetworkXNoPath:
                continue
            a = np.zeros(len(edges), dtype=float)
            for u, v in zip(path[:-1], path[1:]):
                a[edge_idx[tuple(sorted((u, v)))]] += 1.0
            rows.append(a); y.append(float(flow)); used.append((o, d))
        if not rows:
            return {"edge_fluxes": {}, "diagnostics": {"status": "no_valid_od"}}

        A = np.vstack(rows); yv = np.asarray(y, dtype=float)
        x = np.maximum(0.0, np.full(len(edges), max(float(yv.mean()) / max(1, A.shape[1]), 0.0)))
        AtA = A.T @ A
        Aty = A.T @ yv
        step = float(learning_rate / max(np.linalg.norm(AtA, ord=2) + regularization, 1.0))
        for _ in range(iterations):
            grad = AtA @ x - Aty + regularization * x
            x_new = np.maximum(0.0, x - step * grad)
            if np.max(np.abs(x_new - x)) < 1e-8:
                x = x_new
                break
            x = x_new
        residual = A @ x - yv
        return {
            "edge_fluxes": {f"{u}||{v}": float(x[i]) for i, (u, v) in enumerate(edges)},
            "diagnostics": {
                "status": "ok", "od_constraints": len(yv), "edges": len(edges),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "mae": float(np.mean(np.abs(residual))),
                "regularization": float(regularization),
                "used_od": [f"{o}||{d}" for o, d in used],
            }
        }

    @classmethod
    def aggregate_ecological_inference(cls, df, outcome, predictor,
                                       weight_col=None):
        """Regresión ecológica agregada con soporte explícito a variables políticas.

        Acepta de forma explícita:
          voto, apoyo, intención, candidato, partido, preferencia,
          persuadible, probabilidad_voto (y alias relacionados).

        Permite desagregación estadística territorial y convierte la
        inferencia ecológica en un clasificador de personas a nivel de
        unidad territorial (estimado por sección/unidad).
        """
        if outcome not in df or predictor not in df:
            raise ValueError("Faltan outcome/predictor en el DataFrame")
        is_political = (
            outcome in cls.ACCEPTED_POLITICAL or predictor in cls.ACCEPTED_POLITICAL
            or any(k in str(outcome).lower() for k in cls.ACCEPTED_POLITICAL)
            or any(k in str(predictor).lower() for k in cls.ACCEPTED_POLITICAL)
        )
        x = pd.to_numeric(df[predictor], errors="coerce")
        y = pd.to_numeric(df[outcome], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < 3:
            raise ValueError("Se requieren al menos 3 unidades territoriales")
        X = np.column_stack([np.ones(mask.sum()), x[mask].to_numpy(float)])
        Y = y[mask].to_numpy(float)
        if weight_col and weight_col in df:
            w = pd.to_numeric(df.loc[mask, weight_col], errors="coerce").fillna(1).to_numpy(float)
            w = np.maximum(w, 1e-9)
        else:
            w = np.ones(len(Y))
        W = np.diag(w / w.sum())
        beta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ Y
        pred = beta[0] + beta[1] * x.to_numpy(float)
        # Para variables de tipo probabilidad/preferencia/voto se acota a [0,1]
        if is_political or outcome.lower().startswith(("prob", "prefer", "apoyo", "voto", "intenc")):
            pred = np.clip(pred, 0.0, 1.0)
        else:
            pred = np.clip(pred, -np.inf, np.inf)
        out = df.copy()
        out[f"estimado_{outcome}"] = pred
        # Clasificador de personas (nivel unidad territorial): umbral 0.5
        # sobre el estimado ecológico → etiqueta binaria de apoyo/intención.
        if is_political:
            out[f"clase_{outcome}"] = (pred >= 0.5).astype(int)
            out[f"prob_clase_{outcome}"] = pred
        return out, {
            "intercept": float(beta[0]),
            "slope": float(beta[1]),
            "n": int(mask.sum()),
            "weighted": bool(weight_col),
            "political_disaggregation": bool(is_political),
            "classifier_enabled": bool(is_political),
        }

    @staticmethod
    def bayesian_trait_update(prior_mean, prior_variance, observed_mean,
                              observation_variance, reliability=1.0):
        """Bayes gaussiano para un RASGO TERRITORIAL explícitamente observado."""
        pm = np.asarray(prior_mean, dtype=float)
        pv = np.maximum(np.asarray(prior_variance, dtype=float), 1e-9)
        om = np.asarray(observed_mean, dtype=float)
        ov = np.maximum(np.asarray(observation_variance, dtype=float), 1e-9)
        rel = np.clip(float(reliability), 0.0, 1.0)
        effective_ov = ov / max(rel, 1e-6)
        post_v = 1.0 / (1.0/pv + 1.0/effective_ov)
        post_m = post_v * (pm/pv + om/effective_ov)
        return np.clip(post_m, 0.0, 1.0), post_v

    @staticmethod
    def kalman_trait_update(prior_mean, prior_variance, observation,
                            observation_variance, process_variance=0.01):
        """Kalman 1D: predicción + observación para rasgos territoriales."""
        pm = np.asarray(prior_mean, dtype=float)
        pv = np.asarray(prior_variance, dtype=float)
        z = np.asarray(observation, dtype=float)
        R = np.maximum(np.asarray(observation_variance, dtype=float), 1e-9)
        Q = max(float(process_variance), 0.0)
        pred_v = np.maximum(pv + Q, 1e-9)
        K = pred_v / (pred_v + R)
        post_m = pm + K * (z - pm)
        post_v = (1 - K) * pred_v
        return np.clip(post_m, 0.0, 1.0), post_v, K

    @staticmethod
    def spatial_mrf_smooth(values, adjacency, strength=0.35, iterations=20):
        """Suavizado MRF/CRF-like continuo sobre grafo territorial.
        No crea observaciones nuevas; reduce discontinuidades espurias entre
        unidades colindantes cuando existe evidencia espacial suficiente.
        """
        x = np.asarray(values, dtype=float).copy()
        n = len(x)
        if n == 0:
            return x, {"iterations": 0, "mean_change": 0.0}
        strength = float(np.clip(strength, 0.0, 1.0))
        last_change = np.inf
        for it in range(iterations):
            old = x.copy()
            for i in range(n):
                neigh = adjacency[i] if i < len(adjacency) else []
                neigh = [j for j in neigh if 0 <= int(j) < n]
                if not neigh:
                    continue
                x[i] = (1-strength) * old[i] + strength * float(np.mean(old[neigh]))
            x = np.clip(x, 0.0, 1.0)
            last_change = float(np.mean(np.abs(x-old)))
            if last_change < 1e-6:
                break
        return x, {"iterations": it+1, "mean_change": last_change, "strength": strength}

    @staticmethod
    def observation_table(df_obs, id_col="territorial_unit_id", variable_col="variable",
                          value_col="value", variance_col="variance", reliability_col="reliability"):
        """Normaliza observaciones de campo agregadas y calcula media/varianza por unidad-variable."""
        if df_obs is None or df_obs.empty:
            return pd.DataFrame()
        req = [id_col, variable_col, value_col]
        if not all(c in df_obs.columns for c in req):
            raise ValueError(f"CSV de observaciones requiere: {req}")
        d = df_obs.copy()
        d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
        d = d.dropna(subset=[id_col, variable_col, value_col])
        rows = []
        for (uid, var), g in d.groupby([id_col, variable_col]):
            vals = g[value_col].to_numpy(float)
            v = float(np.var(vals, ddof=1)) if len(vals) > 1 else 0.04
            if variance_col in g:
                vv = pd.to_numeric(g[variance_col], errors="coerce").dropna()
                if len(vv): v = float(max(vv.mean(), 1e-6))
            rel = 1.0
            if reliability_col in g:
                rr = pd.to_numeric(g[reliability_col], errors="coerce").dropna()
                if len(rr): rel = float(np.clip(rr.mean(), 0, 1))
            rows.append({"territorial_unit_id": str(uid), "variable": str(var),
                         "observed_mean": float(np.mean(vals)),
                         "observed_variance": max(v, 1e-6), "n_obs": int(len(vals)),
                         "reliability": rel})
        return pd.DataFrame(rows)

# ---------------------------------------------------------------------
# Question Engine
# ---------------------------------------------------------------------

class FreeQuestionParser:
    """Parser de pregunta libre → intención, territorio, variables, familia P1–P77.

    No es NLU profundo: es un enrutador determinista + reglas léxicas
    reproducible (mismo texto → mismo enrutamiento).
    """

    # familia → ids P
    FAMILIES = {
        "estado": list(range(1, 12)),
        "red": list(range(12, 20)),
        "dinamica": list(range(20, 29)),
        "escenario": list(range(29, 43)),
        "presupuesto": list(range(43, 55)),
        "mecanismos": list(range(55, 59)),
        "adversario": list(range(59, 67)),
        "resiliencia": list(range(67, 78)),
    }

    INTENT_KEYWORDS = [
        ("explicacion", ["por qué", "porque", "por que", "explica", "causa", "motivo", "razón", "razon"]),
        ("escenario", ["qué pasa si", "que pasa si", "si aumento", "si bajo", "si pongo", "si hay", "contrafactual", "shock", "escándalo", "escandalo"]),
        ("comparacion", ["versus", " vs ", "compar", "mejor que", "peor que", "diferencia entre"]),
        ("deteccion", ["anomal", "raro", "inesperado", "no explic", "sorpres", "outlier", "diverg"]),
        ("operacion", ["brigada", "gps", "cobertura", "ruta", "visita", "satur"]),
        ("presupuesto", ["presupuesto", "costo", "roi", "cuánto cuesta", "cuanto cuesta", "$", "dinero", "horas"]),
        ("red", ["red", "broker", "hub", "conectiv", "centraliz", "spof", "resilien", "influencia", "clustering", "tribu"]),
        ("estado", ["polariz", "simpat", "opositor", "indecis", "disputa", "consolid", "fragment", "temperatura", "moviliz"]),
        ("lider", ["líder", "lider", "broker", "saf", "habilidad", "capital social", "arraigo", "mary"]),
        ("temporal", ["esta semana", "cambió", "cambio", "evoluc", "converge", "pasos", "tendencia", "antes", "después", "despues"]),
    ]

    VAR_KEYWORDS = {
        "polarizacion": ["polariz"],
        "simpatia": ["simpat", "apoyo"],
        "movilizacion": ["moviliz"],
        "temperatura": ["temperatura", "enojo", "malestar"],
        "saf": ["saf", "habilidad", "liderazgo"],
        "red": ["red", "conectiv", "broker", "hub", "spof"],
        "presupuesto": ["presupuesto", "costo", "roi", "cobertura"],
        "brigada": ["brigada", "gps", "ruta"],
        "resistencia": ["resistencia", "inercia", "comité", "comite"],
        "adversario": ["adversario", "rival", "contra"],
        "echo": ["echo", "cámara de eco", "camara de eco", "epsilon"],
    }

    LEVEL_KEYWORDS = {
        "MACRO · CDMX": ["cdmx", "ciudad", "macro", "global"],
        "ALCALDIA": ["alcald", "demarcacion", "demarcación"],
        "SECCION": ["sección", "seccion", "electoral"],
        "MANZANA": ["manzana", "colonia"],
        "INDIVIDUO_LIDER · traits de equipos": ["líder", "lider", "broker", "equipo", "brigadista", "individuo", "persona", "mary"],
    }

    @staticmethod
    def _norm(text: str) -> str:
        t = (text or "").lower().strip()
        for a, b in {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u", "ñ": "n"}.items():
            t = t.replace(a, b)
        return t

    @classmethod
    def detect_alcaldias(cls, text: str, known=None):
        known = known or ALCALDIAS
        t = cls._norm(text)
        hits = []
        for a in known:
            an = cls._norm(a)
            # Match full name or distinctive multi-char tokens only (avoid "la", "a")
            if an in t:
                hits.append(a)
                continue
            tokens = [tok for tok in an.split() if len(tok) >= 4]
            if tokens and all(tok in t for tok in tokens):
                hits.append(a)
                continue
            # single distinctive token (IZTAPALAPA, XOCHIMILCO, TLALPAN...)
            for tok in an.split():
                if len(tok) >= 6 and tok in t:
                    hits.append(a)
                    break
        aliases = {
            "gam": "GUSTAVO A MADERO",
            "obregon": "ALVARO OBREGON",
            "bj": "BENITO JUAREZ",
            "benito juarez": "BENITO JUAREZ",
            "magdalena contreras": "LA MAGDALENA CONTRERAS",
            "magdalena": "LA MAGDALENA CONTRERAS",
            "cuaji": "CUAJIMALPA DE MORELOS",
            "cuajimalpa": "CUAJIMALPA DE MORELOS",
        }
        for k, v in aliases.items():
            if k in t and v not in hits:
                hits.append(v)
        return hits

    @classmethod
    def parse(cls, text: str) -> dict:
        raw = text or ""
        t = cls._norm(raw)
        intents = []
        for name, keys in cls.INTENT_KEYWORDS:
            if any(k in t for k in keys):
                intents.append(name)
        if not intents:
            intents = ["estado"]

        variables = []
        for var, keys in cls.VAR_KEYWORDS.items():
            if any(k in t for k in keys):
                variables.append(var)

        level = "ALCALDIA"
        for lev, keys in cls.LEVEL_KEYWORDS.items():
            if any(k in t for k in keys):
                level = lev
                break

        territories = cls.detect_alcaldias(raw)

        # Map to P numbers by priority rules
        scored = {i: 0 for i in range(1, 78)}
        for intent in intents:
            if intent == "estado":
                for i in cls.FAMILIES["estado"]:
                    scored[i] += 2
            elif intent == "red":
                for i in cls.FAMILIES["red"] + cls.FAMILIES["resiliencia"][:3]:
                    scored[i] += 2
            elif intent == "presupuesto":
                for i in cls.FAMILIES["presupuesto"]:
                    scored[i] += 2
            elif intent == "operacion":
                for i in [25, 28, 32, 43, 44, 51, 52, 53]:
                    scored[i] += 3
            elif intent == "escenario":
                for i in cls.FAMILIES["escenario"] + cls.FAMILIES["adversario"]:
                    scored[i] += 2
            elif intent == "explicacion":
                for i in [1, 8, 9, 10, 15, 40, 57, 70, 73]:
                    scored[i] += 2
            elif intent == "temporal":
                for i in cls.FAMILIES["dinamica"]:
                    scored[i] += 3
            elif intent == "lider":
                for i in [5, 16, 17, 18, 36, 37, 38, 39, 40, 41, 42]:
                    scored[i] += 3
            elif intent == "deteccion":
                for i in [1, 8, 9, 21, 23, 24, 67, 68]:
                    scored[i] += 2

        for var in variables:
            boost = {
                "polarizacion": [1, 8, 24, 57, 75, 76],
                "simpatia": [2, 6, 7, 11],
                "movilizacion": [4, 29],
                "temperatura": [3],
                "saf": [5, 16, 17, 40],
                "red": [12, 13, 14, 15, 18, 19, 67, 69],
                "presupuesto": [32, 43, 44, 51, 53, 64, 65],
                "brigada": [25, 28, 52],
                "resistencia": [10, 70, 71, 72, 73],
                "adversario": [59, 60, 61, 62, 63],
                "echo": [55, 56, 57, 58, 76],
            }.get(var, [])
            for i in boost:
                scored[i] += 3

        ranked = sorted(scored.items(), key=lambda x: (-x[1], x[0]))
        top = [i for i, s in ranked if s > 0][:5]
        if not top:
            top = [1, 6, 12, 44, 77]

        return {
            "raw": raw,
            "intents": intents,
            "variables": variables,
            "territories": territories,
            "level": level,
            "p_numbers": top,
            "scores": {str(i): s for i, s in ranked[:10] if s > 0},
        }


class HypothesisEngine:
    """Orquesta pregunta libre → hipótesis medibles → respuestas P1–P77.

    Flujo:
      texto libre
        → FreeQuestionParser
        → selección de P's
        → QuestionEngine.answer por cada P
        → informe estructurado
    """

    @staticmethod
    def run(text, adf, terr, net, budget_df=None, brig_report=None, model=None,
            level_override=None, experiment_id="", output_hash="", seed=42) -> dict:
        parsed = FreeQuestionParser.parse(text)
        level = level_override or parsed["level"]
        results = []
        for pn in parsed["p_numbers"]:
            if 1 <= pn <= len(QuestionEngine.CATALOG):
                q = QuestionEngine.CATALOG[pn - 1]
                ans = QuestionEngine.answer(
                    q, adf, terr, net,
                    budget_df=budget_df,
                    brig_report=brig_report,
                    model=model,
                    level=level,
                    experiment_id=experiment_id,
                    output_hash=output_hash,
                    seed=seed,
                )
                results.append({"p": pn, "question": q, "answer": ans})

        # Hypothesis statement
        intents = ", ".join(parsed["intents"])
        vars_ = ", ".join(parsed["variables"]) or "estado territorial"
        terrs = ", ".join(parsed["territories"]) or "CDMX (agregado disponible)"
        hypothesis = (
            f"Hipótesis de trabajo: la pregunta se interpreta como intención(es) [{intents}] "
            f"sobre variable(s) [{vars_}] en territorio(s) [{terrs}] a nivel [{level}]. "
            f"Se contrastan las métricas P{', P'.join(str(p) for p in parsed['p_numbers'])}."
        )

        # Data gaps
        gaps = []
        if not parsed["territories"] and "seccion" in FreeQuestionParser._norm(text):
            gaps.append("Mencionas sección pero no hay filtro territorial explícito resoluble con alcaldías conocidas.")
        if "brigada" in parsed["variables"] and (brig_report is None or getattr(brig_report, "empty", True)):
            gaps.append("No hay reporte de brigadas/GPS cargado: P25/P28 quedan limitadas.")
        if level.startswith("MANZANA") and ("manzana" not in adf.columns or adf["manzana"].astype(str).str.len().max() == 0):
            gaps.append("Nivel MANZANA solicitado sin datos de manzana en el dataset actual.")
        if level.startswith("SECCION") and adf["seccion"].nunique() <= max(16, adf["alcaldia"].nunique()):
            gaps.append("Nivel SECCION solicitado; el dataset parece grano alcaldía o IDs sintéticos.")
        if "adversario" in parsed["variables"]:
            gaps.append("Escenarios de adversario requieren re-simulación explícita (insert_broker opuesto); la respuesta es orientativa.")

        return {
            "parsed": parsed,
            "hypothesis": hypothesis,
            "level": level,
            "results": results,
            "gaps": gaps,
        }

    @staticmethod
    def render_markdown(report: dict) -> str:
        lines = [
            "### Informe Hypothesis Engine",
            "",
            report.get("hypothesis", ""),
            "",
            f"**Nivel usado:** {report.get('level')}",
            f"**Intenciones:** {', '.join(report.get('parsed', {}).get('intents', []))}",
            f"**Variables:** {', '.join(report.get('parsed', {}).get('variables', [])) or '—'}",
            f"**Territorios detectados:** {', '.join(report.get('parsed', {}).get('territories', [])) or '—'}",
            f"**P seleccionadas:** {', '.join('P'+str(p) for p in report.get('parsed', {}).get('p_numbers', []))}",
            "",
        ]
        if report.get("gaps"):
            lines.append("**Brechas de datos / modelo:**")
            for g in report["gaps"]:
                lines.append(f"- {g}")
            lines.append("")
        for r in report.get("results", []):
            lines.append(f"#### P{r['p']}. {r['question'][r['question'].find('.')+2:] if '.' in r['question'] else r['question']}")
            lines.append("")
            lines.append(r["answer"])
            lines.append("")
        lines.append("---")
        lines.append(
            "_Esto no es causalidad demostrada: es enrutamiento reproducible a métricas y escenarios "
            "del laboratorio. Usa calibración y contrafácticos para fortalecer evidencia._"
        )
        return "\n".join(lines)


class QuestionEngine:
    """Question Engine v6.0 alineado al catálogo SITER 77 preguntas (P1–P77).

    Cada respuesta tiene:
      - métrica técnica calculada sobre datos actuales
      - narrativa cliente (consultoría)
      - nivel de análisis seleccionado
      - nota de incertidumbre / datos faltantes

    Niveles operativos:
      MACRO | ALCALDIA | SECCION | MANZANA | INDIVIDUO_LIDER
    """

    LEVELS = [
        "MACRO · CDMX",
        "ALCALDIA",
        "SECCION",
        "MANZANA",
        "INDIVIDUO_LIDER · traits de equipos",
    ]

    # Catálogo exacto P1–P77 (PDF SITER-CAE v4.0)
    CATALOG = [
        "P1. ¿Dónde está más polarizado?",
        "P2. ¿Dónde hay más simpatizantes?",
        "P3. ¿Qué territorio tiene más temperatura sintética?",
        "P4. ¿Dónde se moviliza más la gente?",
        "P5. ¿Qué territorio tiene más habilidades promedio?",
        "P6. ¿Qué campo está en DISPUTA_ABIERTA?",
        "P7. ¿Qué campo está consolidado?",
        "P8. ¿Dónde hay más conflicto interno?",
        "P9. ¿Qué campo es más fragmentado?",
        "P10. ¿Cuál tiene más institucionalización?",
        "P11. ¿Cuál es dominante y su dominancia?",
        "P12. ¿Qué tan conectada está mi red?",
        "P13. ¿Mi red es de tribus o abierta?",
        "P14. ¿Hay centralización peligrosa?",
        "P15. ¿Qué tan desigual es la influencia?",
        "P16. ¿Quiénes son mis centros SAF?",
        "P17. ¿Mis centros son buenos o solo grillos?",
        "P18. ¿Cuántos brokers tengo?",
        "P19. ¿Top10% cuánta influencia concentra?",
        "P20. ¿En cuántos pasos converge?",
        "P21. ¿Quién emergió como centro en la simulación?",
        "P22. ¿Cómo evoluciona la influencia / conversiones proxy?",
        "P23. ¿Dónde está hotspot de influencia?",
        "P24. ¿Dónde está caliente la polarización?",
        "P25. ¿Mis brigadas saturan territorio?",
        "P26. ¿Hasta dónde llega una onda (rumor/bache) proxy?",
        "P27. ¿Qué centros están conectados?",
        "P28. ¿Qué territorio más visitas de brigada?",
        "P29. ¿Si refuerzo movilización, cuánto sube el estado?",
        "P30. ¿Si hay escándalo/shock, qué territorio colapsa más?",
        "P31. ¿Si aumento conectividad, qué pasa con polarización?",
        "P32. ¿Con distinto presupuesto cuánto cubro?",
        "P33. ¿Qué pasa si se va mi top broker?",
        "P34. ¿Framing/taller vs brigada (proxy costo-beneficio)?",
        "P35. ¿Probabilidad de que una intervención mejore >5%?",
        "P36. ¿Qué pasa si inserto un broker de alto capital?",
        "P37. ¿Y si ese broker fuera opositor?",
        "P38. ¿Un líder de habilidades bajas sirve?",
        "P39. ¿Dónde poner broker nacional vs local?",
        "P40. ¿Qué rasgos importan más en SAF?",
        "P41. ¿Cuántas conexiones necesita un broker?",
        "P42. ¿Arraigo bajo limita al líder?",
        "P43. ¿Cuánto cuesta cubrir el universo actual?",
        "P44. ¿Qué territorio tiene mejor ROI operacional?",
        "P45. ¿Territorios aislados: costo de conectar?",
        "P46. ¿Intervención mínima para mover un territorio débil?",
        "P47. ¿Activos top10% son óptimos?",
        "P48. ¿Es reproducible el experimento?",
        "P49. ¿Tiene PII?",
        "P50. ¿Cómo exporto al motor / auditoría?",
        "P51. ¿Cuánto cuesta convertir (proxy unitario)?",
        "P52. ¿Qué pasa cuando brigadistas se fatigan?",
        "P53. ¿Cuándo se agota el presupuesto?",
        "P54. ¿Cómo afecta el capital político desgastado?",
        "P55. ¿Qué es epsilon de confianza?",
        "P56. ¿Qué es echo chamber y repulsión?",
        "P57. ¿Cómo mido polarización endógena?",
        "P58. ¿Qué pasa si epsilon es muy bajo?",
        "P59. ¿Qué pasa si pongo adversario espejo?",
        "P60. ¿Cuánto pierdo con adversario vs sin?",
        "P61. ¿Qué es estrategia de flanqueo adversario?",
        "P62. ¿Qué es decapitación de red?",
        "P63. ¿Cómo detectaría el adversario a mi broker?",
        "P64. ¿Con presupuesto fijo, distribución orientativa?",
        "P65. ¿Qué territorio merece más brokers vs horas?",
        "P66. ¿Cómo evoluciona un score de optimización?",
        "P67. ¿Qué nodos son punto único de falla (SPOF)?",
        "P68. ¿Si me quitan a X, cuánto caigo?",
        "P69. ¿Mi red es resiliente?",
        "P70. ¿Qué territorio tiene alta inercia institucional?",
        "P71. ¿Qué supermayoría de vecinos se necesita?",
        "P72. ¿Dónde trabajar con líderes formales?",
        "P73. ¿Qué es inercia normativa?",
        "P74. ¿Cómo se modela la fatiga del territorio/votante?",
        "P75. ¿Cuál es el costo de polarizar?",
        "P76. ¿Cómo se evita una cámara de eco?",
        "P77. ¿Qué aporta esta versión respecto al modelo base?",
    ]

    @staticmethod
    def _pn(question: str) -> int:
        if question in QuestionEngine.CATALOG:
            return QuestionEngine.CATALOG.index(question) + 1
        # fallback: parse P##
        import re as _re
        m = _re.match(r"P(\d+)\.", question.strip())
        return int(m.group(1)) if m else 0

    @staticmethod
    def _fmt(tech: str, narr: str, level: str, note: str = "") -> str:
        parts = [
            f"**Nivel:** {level}",
            f"**Métrica técnica:** {tech}",
            f"**Narrativa cliente:** {narr}",
        ]
        if note:
            parts.append(f"**Nota:** {note}")
        return "\n\n".join(parts)

    @staticmethod
    def answer(question, adf, terr, net, budget_df=None, brig_report=None,
               model=None, level="ALCALDIA", experiment_id="", output_hash="",
               seed=42):
        if adf is None or adf.empty:
            return "No hay datos cargados. Ejecuta SETUP."
        pn = QuestionEngine._pn(question)
        level = level or "ALCALDIA"

        # helpers
        def top_alc(col, asc=False):
            if col not in terr.columns or terr.empty:
                return "N/D", float("nan")
            r = terr.sort_values(col, ascending=asc).iloc[0]
            return str(r.alcaldia), float(r[col])

        def mean_by(col):
            if col not in adf.columns:
                return {}
            return adf.groupby("alcaldia")[col].mean().to_dict()

        leaders = adf[adf["es_lider"] == 1] if "es_lider" in adf.columns else adf.iloc[0:0]

        # ----- P1–P11 estado territorial -----
        if pn == 1:
            a, v = top_alc("polarizacion_std")
            return QuestionEngine._fmt(
                f"máx σ opinión={v:.3f} en {a}",
                f"En {a} todos pelean: no hay ganador claro. Zona de fricción discursiva.",
                level)
        if pn == 2:
            a, v = top_alc("simpat_pct")
            return QuestionEngine._fmt(
                f"simpat_pct máx={v:.1f}% en {a}",
                f"En {a} ya hay base de apoyo agregado: prioriza mantenimiento, no sobreinversión.",
                level)
        if pn == 3:
            g = adf.groupby("alcaldia")["temperatura"].mean() if "temperatura" in adf.columns else None
            if g is None or g.empty:
                return QuestionEngine._fmt("sin columna temperatura", "Falta variable de temperatura/problema.", level)
            a, v = g.idxmax(), float(g.max())
            return QuestionEngine._fmt(
                f"temp_prom máx={v:.3f} en {a}",
                f"En {a} el malestar agregado es alto: cualquier chispa prende. Prioridad de contención.",
                level)
        if pn == 4:
            g = adf.groupby("alcaldia")["nivel_movilizacion"].mean()
            a, v = g.idxmax(), float(g.max())
            return QuestionEngine._fmt(
                f"mov_prom máx={v:.3f} en {a}",
                f"En {a} hay más disposición a movilizarse: mejor retorno esperado de brigada.",
                level)
        if pn == 5:
            a, v = top_alc("saf_skill_prom")
            return QuestionEngine._fmt(
                f"saf_skill_prom máx={v:.3f} en {a}",
                f"En {a} el promedio de habilidad SAF es alto: semillero de líderes/brokers.",
                level)
        if pn == 6:
            x = terr[terr["campo"] == "DISPUTA_ABIERTA"] if "campo" in terr.columns else terr.iloc[0:0]
            lista = ", ".join(x["alcaldia"].tolist()) if not x.empty else "ninguna"
            return QuestionEngine._fmt(
                f"DISPUTA_ABIERTA: {lista}",
                "Donde nadie consolidó: los indecisos deciden. Ahí se gana o se pierde.",
                level)
        if pn == 7:
            x = terr[terr["campo"] == "CONSOLIDACION"] if "campo" in terr.columns else terr.iloc[0:0]
            lista = ", ".join(x["alcaldia"].tolist()) if not x.empty else "ninguna"
            return QuestionEngine._fmt(
                f"CONSOLIDACION: {lista}",
                "Ya hay dominio relativo: no gastes de más; mantenimiento.",
                level)
        if pn == 8:
            a, v = top_alc("polarizacion_std")
            return QuestionEngine._fmt(
                f"conflicto proxy σ={v:.3f} en {a}",
                f"En {a} hay más fricción interna (proxy). Puede requerir mediación, no solo mensaje.",
                level)
        if pn == 9:
            col = "entropia" if "entropia" in terr.columns else "polarizacion_std"
            a, v = top_alc(col)
            return QuestionEngine._fmt(
                f"fragmentación proxy ({col})={v:.3f} en {a}",
                f"En {a} el campo está más atomizado: no es solo polarización binaria.",
                level)
        if pn == 10:
            z = terr.copy()
            if "estabilidad" in z.columns and "resistencia_prom" in z.columns:
                z["inst"] = 0.6 * z["estabilidad"] + 0.4 * (1 - z["resistencia_prom"])
                r = z.loc[z["inst"].idxmax()]
                return QuestionEngine._fmt(
                    f"institucionalización={r.inst:.3f} en {r.alcaldia}",
                    f"En {r.alcaldia} las reglas/estructura aguantan mejor el ruido.",
                    level)
            return QuestionEngine._fmt("faltan estabilidad/resistencia", "No se puede calcular institucionalización.", level)
        if pn == 11:
            # dominante category at macro
            if "intencion" in adf.columns:
                vc = adf["intencion"].value_counts(normalize=True)
                dom, share = vc.index[0], float(vc.iloc[0])
            else:
                dom, share = "N/D", 0.0
            return QuestionEngine._fmt(
                f"dominante={dom} dominancia={share:.1%}",
                "Quién manda hoy y con cuánto. Si dominancia <40%, nadie manda con claridad.",
                level)

        # ----- P12–P19 red -----
        if pn == 12:
            return QuestionEngine._fmt(
                f"densidad={net.get('density', 0):.4f}, aristas={net.get('edges', net.get('n_edges', 'N/D'))}, "
                f"n={net.get('n_nodes', net.get('nodes', len(adf)))}",
                "Qué tan amarrada está la estructura. Densidad baja = archipiélago.",
                level)
        if pn == 13:
            cl = net.get("clustering", net.get("avg_clustering", None))
            if cl is None:
                tech = "clustering no disponible en métricas actuales"
                narr = "Corre la red Mesa; si no hay clustering, se reporta densidad como proxy."
            else:
                tech = f"clustering={float(cl):.3f}"
                narr = "Clustering alto → tribus/camarillas; bajo → red más abierta."
            return QuestionEngine._fmt(tech, narr, level)
        if pn == 14:
            cen = net.get("centralization_degree", net.get("centralization", net.get("degree_centralization", net.get("gini_influence", 0))))
            return QuestionEngine._fmt(
                f"centralización/proxy desigualdad={float(cen):.3f}",
                "Si depende de 1–2 hubs, un golpe a esos nodos derrumba tramos de red.",
                level)
        if pn == 15:
            g = net.get("gini_influence", terr["saf_skill_prom"].std() if "saf_skill_prom" in terr.columns else 0)
            return QuestionEngine._fmt(
                f"gini_influencia/proxy={float(g):.3f}",
                "El poder de influencia está concentrado o repartido. Alta desigualdad = dependencia de élite.",
                level)
        if pn == 16:
            cols = [c for c in ["territorial_unit_id", "alcaldia", "saf_skill", "influencia", "es_lider"] if c in adf.columns]
            top = adf.sort_values("saf_skill" if "saf_skill" in adf.columns else "influencia", ascending=False).head(10)
            lines = "; ".join(
                f"{r.get('territorial_unit_id','?')} ({r.get('alcaldia','?')}) SAF={r.get('saf_skill', float('nan')):.2f}"
                for _, r in top.iterrows()
            )
            return QuestionEngine._fmt(
                f"Top centros SAF: {lines}",
                "Tu lista operativa de quienes más pueden mover gente (agregado/líder sintético).",
                level,
                "En nivel INDIVIDUO_LIDER usa filas es_lider=1 si existen.")
        if pn == 17:
            if "saf_skill" not in adf.columns:
                return QuestionEngine._fmt("sin saf_skill", "No hay habilidad SAF calculada.", level)
            top5 = adf.nlargest(5, "saf_skill")["saf_skill"].mean()
            prom = adf["saf_skill"].mean()
            gap = float(top5 - prom)
            return QuestionEngine._fmt(
                f"hab_top5={top5:.3f} vs prom={prom:.3f} gap={gap:+.3f}",
                "Si el gap es grande y positivo, tus centros sí destacan; si no, son ruido relativo.",
                level)
        if pn == 18:
            if "es_lider" in adf.columns:
                n_b = int((adf["es_lider"] == 1).sum())
                ratio = n_b / max(len(adf), 1)
            elif "broker_density" in adf.columns:
                n_b = int((adf["broker_density"] > 0.5).sum())
                ratio = n_b / max(len(adf), 1)
            else:
                n_b, ratio = 0, 0.0
            return QuestionEngine._fmt(
                f"brokers/líderes={n_b} ratio={ratio:.1%}",
                "Cuántos puentes/líderes tienes. Sin brokers, cada territorio es más isla.",
                level)
        if pn == 19:
            if "influencia" in adf.columns and adf["influencia"].sum() > 0:
                s = adf["influencia"].sort_values(ascending=False)
                k = max(1, int(0.1 * len(s)))
                share = float(s.head(k).sum() / s.sum())
            else:
                share = float("nan")
            return QuestionEngine._fmt(
                f"top10% share influencia={share:.1%}",
                "Si quitas el top 10%, qué fracción de influencia pierdes. Dependencia de élite.",
                level)

        # ----- P20–P28 dinámica / campo -----
        if pn == 20:
            hist = getattr(model, "history", []) if model is not None else []
            if len(hist) < 2:
                return QuestionEngine._fmt(
                    f"pasos simulados={len(hist)}",
                    "Aún no hay suficiente historia: pulsa GO varias veces para ver convergencia.",
                    level)
            last = hist[-1]
            return QuestionEngine._fmt(
                f"paso={last.get('step')} SIMPAT={100*last.get('SIMPATIZANTE',0):.1f}% "
                f"OPOS={100*last.get('OPOSITOR',0):.1f}% IND={100*last.get('INDECISO',0):.1f}%",
                "Cuánto tarda en asentarse el estado agregado. Si no se estabiliza, disputa crónica.",
                level)
        if pn == 21:
            if leaders is not None and not leaders.empty:
                r = leaders.sort_values("saf_skill", ascending=False).iloc[0]
                return QuestionEngine._fmt(
                    f"líder destacado {r.territorial_unit_id} SAF={r.saf_skill:.3f} en {r.alcaldia}",
                    "Quién emerge como centro en la capa de líderes sintéticos (no PII).",
                    level)
            r = adf.sort_values("saf_skill", ascending=False).iloc[0]
            return QuestionEngine._fmt(
                f"centro {r.territorial_unit_id} SAF={r.saf_skill:.3f}",
                "Centro de mayor habilidad en el estado actual.",
                level)
        if pn == 22:
            hist = getattr(model, "history", []) if model is not None else []
            if len(hist) >= 2:
                d = hist[-1].get("MeanOpinion", 0) - hist[0].get("MeanOpinion", 0)
                return QuestionEngine._fmt(
                    f"Δ MeanOpinion={d:+.3f} en {len(hist)} pasos",
                    "Evolución de la opinión media: proxy grueso de conversiones netas.",
                    level)
            return QuestionEngine._fmt("sin historia", "Ejecuta varios GO para trazar evolución.", level)
        if pn == 23:
            r = adf.sort_values("influencia" if "influencia" in adf.columns else "saf_skill", ascending=False).iloc[0]
            return QuestionEngine._fmt(
                f"hotspot {r.alcaldia} id={r.territorial_unit_id} lat={r.lat:.4f}, lon={r.lon:.4f}",
                "Mancha de poder relativo: ahí concentra influencia el sistema actual.",
                level)
        if pn == 24:
            a, v = top_alc("polarizacion_std")
            return QuestionEngine._fmt(
                f"polarización local máx σ={v:.3f} en {a}",
                f"En {a} el clima está más pelean. Mensaje tibio rinde poco.",
                level)
        if pn == 25:
            if brig_report is None or brig_report.empty:
                return QuestionEngine._fmt(
                    "sin reporte de brigadas/GPS",
                    "Carga plan/GPS o genera plan en SETUP para medir saturación de rutas.",
                    level)
            col = "cobertura_ruta_pct" if "cobertura_ruta_pct" in brig_report.columns else brig_report.columns[-1]
            return QuestionEngine._fmt(
                f"cobertura_ruta media={brig_report[col].mean():.1f}% (n={len(brig_report)})",
                "Si repites el mismo corredor, saturas sin expandir territorio. Redistribuye.",
                level)
        if pn == 26:
            dens = float(net.get("density", 0) or 0)
            return QuestionEngine._fmt(
                f"proxy alcance red densidad={dens:.4f}",
                "Con red densa un shock viaja más rápido; con archipiélago queda local. "
                "Para ondas geográficas reales hace falta contigüidad GIS + pasos de difusión.",
                level,
                "Escenario completo de rumor requiere simulación de contagio espacial dedicada.")
        if pn == 27:
            return QuestionEngine._fmt(
                f"aristas={net.get('edges', net.get('n_edges', 'N/D'))} densidad={net.get('density', 0):.4f}",
                "Quién está enlazado con quién en la red actual de unidades/líderes.",
                level)
        if pn == 28:
            if brig_report is None or brig_report.empty:
                return QuestionEngine._fmt("sin brigadas", "No hay visitas registradas.", level)
            return QuestionEngine._fmt(
                f"filas brigada={len(brig_report)}",
                "Cruza visitas con ROI territorial para ver si el esfuerzo valió.",
                level)

        # ----- P29–P42 escenarios / líderes -----
        if pn == 29:
            return QuestionEngine._fmt(
                "Escenario: +movilización requiere re-simular con field_pressure/coupling más altos",
                "Refuerza movilización en un territorio débil y compara SIMPAT antes/después con varios GO. "
                "Reporta Δ y, si usas Monte Carlo presupuestario, la distribución.",
                level,
                "No hay un solo número mágico: usa sensibilidad + MC.")
        if pn == 30:
            a, v = top_alc("polarizacion_std")
            return QuestionEngine._fmt(
                f"territorio más frágil a shock (proxy polarización)={a} σ={v:.3f}",
                f"Ante escándalo, {a} es el candidato a mayor daño relativo. Activa protocolo ahí primero.",
                level)
        if pn == 31:
            return QuestionEngine._fmt(
                f"polarización actual σ={float(adf.opinion.std()):.3f}; densidad red={net.get('density', 0):.4f}",
                "Subir conectividad a veces pacifica y a veces contagia conflicto. Compara runs con p_intra/p_inter distintos.",
                level)
        if pn == 32:
            if budget_df is None or budget_df.empty:
                return QuestionEngine._fmt("sin presupuesto calculado", "Configura presupuesto en sidebar y SETUP.", level)
            cov = float(budget_df.get("cobertura_est", budget_df.iloc[:, -1]).mean()) if "cobertura_est" in budget_df.columns else float("nan")
            cost = float(budget_df["costo"].sum()) if "costo" in budget_df.columns else float("nan")
            return QuestionEngine._fmt(
                f"costo asignado=${cost:,.0f} cobertura_est media={cov:.2%}" if cov <= 1 else f"costo=${cost:,.0f} cobertura={cov:.1f}",
                "Con más presupuesto sube cobertura, pero el ROI suele ser decreciente. Mira la pestaña de sensibilidad.",
                level)
        if pn == 33:
            if model is not None and hasattr(model, "agent_dataframe"):
                sp = TerritorialAnalytics.spof(model)
                if sp is not None and not sp.empty:
                    r = sp.iloc[0]
                    return QuestionEngine._fmt(
                        f"SPOF top: {r.to_dict()}",
                        "Si se va tu principal conector, el sistema estima el golpe. Protege o duplica ese rol.",
                        level)
            return QuestionEngine._fmt(
                "SPOF no calculado",
                "Ejecuta análisis SPOF en Estadística / red tras SETUP.",
                level)
        if pn == 34:
            return QuestionEngine._fmt(
                "proxy: taller (bajo costo, menor certeza) vs brigada (mayor costo, más cobertura directa)",
                "Compara Δ estado por peso invertido. El taller es más barato; la brigada suele pegar más si hay GPS/plan.",
                level)
        if pn == 35:
            return QuestionEngine._fmt(
                "Usa Monte Carlo de presupuesto/cobertura (pestaña Presupuesto)",
                "No hay bola de cristal: reporta P(mejora>5%) sobre simulaciones, no un único punto.",
                level)
        if pn in (36, 37, 38, 39, 40, 41, 42):
            # leader trait questions
            if not leaders.empty:
                best = leaders.sort_values("saf_skill", ascending=False).iloc[0]
                worst = leaders.sort_values("saf_skill", ascending=True).iloc[0]
            else:
                best = adf.sort_values("saf_skill", ascending=False).iloc[0]
                worst = adf.sort_values("saf_skill", ascending=True).iloc[0]
            if pn == 36:
                return QuestionEngine._fmt(
                    f"broker alto capital ejemplo {best.territorial_unit_id} SAF={best.saf_skill:.3f} capital={best.capital_social:.2f}",
                    "Insertar un broker fuerte en territorio débil suele levantar local y, por red, vecinos.",
                    level)
            if pn == 37:
                return QuestionEngine._fmt(
                    "mismo perfil, spin/opinión invertida (escenario)",
                    "El mismo perfil en bando contrario produce efecto espejo: el perfil importa, el signo más.",
                    level)
            if pn == 38:
                return QuestionEngine._fmt(
                    f"líder bajo SAF ejemplo {worst.territorial_unit_id} SAF={worst.saf_skill:.3f}",
                    "Si no es respetado/capaz, colocarlo rinde poco. No óptimo.",
                    level)
            if pn == 39:
                return QuestionEngine._fmt(
                    "hub (alto grado) impacta varios territorios; periferia impacta local pero a veces más intenso",
                    "Broker nacional en hub vs local en isla: elige según objetivo (difusión vs profundidad).",
                    level)
            if pn == 40:
                return QuestionEngine._fmt(
                    "SAF ≈ 0.30·capital + 0.25·info + 0.25·liderazgo + 0.10·arraigo + 0.10·mov − 0.15·desconfianza",
                    "Capital, información y liderazgo pesan ~80% del score SAF en este laboratorio.",
                    level)
            if pn == 41:
                return QuestionEngine._fmt(
                    f"grado/influencia media={float(adf['influencia'].mean()) if 'influencia' in adf.columns else float('nan'):.3f}",
                    "Con pocas conexiones mueve poco; con más grado escala el alcance (con rendimientos decrecientes).",
                    level)
            if pn == 42:
                low = adf.nsmallest(5, "arraigo")["saf_skill"].mean() if "arraigo" in adf.columns else float("nan")
                return QuestionEngine._fmt(
                    f"SAF medio en quintil de menor arraigo≈{low:.3f}",
                    "Arraigo bajo obliga a compensar con capital/liderazgo; si no, el líder rinde medio o bajo.",
                    level)

        # ----- P43–P54 recursos -----
        if pn == 43:
            if budget_df is not None and not budget_df.empty and "costo" in budget_df.columns:
                return QuestionEngine._fmt(
                    f"costo total asignado=${budget_df.costo.sum():,.0f}",
                    "Base para presupuestar cobertura del universo actual.",
                    level)
            return QuestionEngine._fmt("sin budget_df", "Define presupuesto en sidebar.", level)
        if pn == 44:
            if budget_df is not None and not budget_df.empty and "roi_operacional" in budget_df.columns:
                r = budget_df.loc[budget_df["roi_operacional"].idxmax()]
                return QuestionEngine._fmt(
                    f"mejor ROI {r.alcaldia}={r.roi_operacional:.4f}",
                    "Invierte donde cada peso rinde más, no solo donde hay más ruido.",
                    level)
            return QuestionEngine._fmt("sin ROI calculado", "Revisa pestaña Presupuesto.", level)
        if pn == 45:
            dens = float(net.get("density", 0) or 0)
            return QuestionEngine._fmt(
                f"densidad red={dens:.4f} (proxy inverso de aislamiento)",
                "Territorios poco enlazados son caros de conectar: evalúa costo social vs beneficio.",
                level)
        if pn == 46:
            x = terr[terr["campo"] == "DISPUTA_ABIERTA"] if "campo" in terr.columns else terr
            alvo = x.sort_values("simpat_pct").iloc[0] if not x.empty else terr.iloc[0]
            return QuestionEngine._fmt(
                f"candidato a intervención mínima: {alvo.alcaldia} simpat={alvo.simpat_pct:.1f}%",
                "Empieza por el eslabón débil en disputa: prueba 1 broker + horas y mide Δ.",
                level)
        if pn == 47:
            if "saf_skill" in adf.columns and "desconfianza" in adf.columns:
                top = adf.nlargest(max(1, len(adf)//10), "saf_skill")
                opt = float(((top.saf_skill > 0.7) & (top.desconfianza < 0.3)).mean())
                return QuestionEngine._fmt(
                    f"fracción top10% con SAF>0.7 y desconf<0.3 = {opt:.1%}",
                    "Tu élite es sólida o es ruido con desconfianza. Filtra antes de escalar.",
                    level)
            return QuestionEngine._fmt("faltan columnas", "Se requieren saf_skill y desconfianza.", level)
        if pn == 48:
            return QuestionEngine._fmt(
                f"seed={seed} experiment_id={experiment_id or 'N/D'} hash={str(output_hash)[:16] or 'N/D'}",
                "Cualquier auditor puede repetir con el mismo seed, datos y configuración.",
                level)
        if pn == 49:
            return QuestionEngine._fmt(
                "personal_data=False · agregado=True · líderes sintéticos sin PII",
                "No hay datos personales de electores. Traits individuales son de líderes/brokers sintéticos o agregados de organización.",
                level)
        if pn == 50:
            return QuestionEngine._fmt(
                "Export JSON: metadata + territorial_fields + network + budget + brigadistas + hash",
                "Descarga el JSON de la pestaña EXPORT e impórtalo a tu pipeline con validación de hash.",
                level)
        if pn == 51:
            return QuestionEngine._fmt(
                "proxy costo_unitario ≈ (costo_brigada_hora × horas) / unidades cubiertas",
                "Convencer no es gratis: presupuesta dinero y horas por unidad territorial cubierta.",
                level)
        if pn == 52:
            return QuestionEngine._fmt(
                "fatiga de agentes Mesa se amortigua ×0.95 por paso; alta fatiga reduce acoplamiento SAF",
                "Equipos cansados rinden menos. Rota brigadas y líderes expuestos.",
                level)
        if pn == 53:
            if budget_df is not None and not budget_df.empty and "presupuesto_restante_local" in budget_df.columns:
                return QuestionEngine._fmt(
                    f"remanente mín local=${budget_df.presupuesto_restante_local.min():,.0f}",
                    "Cuando dinero u horas tocan cero, la operación se corta. Anticipa el paso de quiebre.",
                    level)
            return QuestionEngine._fmt("sin serie de agotamiento", "Usa presupuesto y MC para ver quiebre.", level)
        if pn == 54:
            return QuestionEngine._fmt(
                "capital/liderazgo alto se gasta si se sobreexpone (proxy: baja saf_skill efectiva con fatiga)",
                "No sobreutilices al mismo líder: el capital político se desgasta.",
                level)

        # ----- P55–P58 mecanismos -----
        if pn == 55:
            return QuestionEngine._fmt(
                "epsilon (Deffuant)=distancia máxima de opinión para atracción",
                "Si piensan parecido (diff < ε) pueden acercarse; si no, no hay diálogo productivo.",
                level)
        if pn == 56:
            return QuestionEngine._fmt(
                "epsilon_repulsion: si diff es muy grande, la interacción repele (echo chamber)",
                "No solo fallan al convencerse: se empujan a extremos. Cámara de eco.",
                level)
        if pn == 57:
            return QuestionEngine._fmt(
                f"polarizacion_endogena σ(opinion)={float(adf.opinion.std()):.3f}",
                "Si σ sube en el tiempo, la sociedad se parte en extremos.",
                level)
        if pn == 58:
            return QuestionEngine._fmt(
                "ε muy bajo → solo convergen casi-iguales → fragmentación persistente",
                "Sociedad cerrada: cada tribu habla sola y no hay consenso.",
                level)

        # ----- P59–P66 adversario / optimización -----
        if pn == 59:
            return QuestionEngine._fmt(
                "adversario espejo: clonar top broker con opinión invertida en mismo territorio",
                "Si pones a Mary, el rival puede poner a Juan igual de capaz en contra. Neutraliza ganancia.",
                level,
                "Implementación completa = insert_broker + spin opuesto + re-simular.")
        if pn == 60:
            return QuestionEngine._fmt(
                "compara runs: base vs +broker vs +broker+adversario",
                "Tu +9% puede quedar en +3% neto si hay contra-ataque. Calcula ROI neto.",
                level)
        if pn == 61:
            return QuestionEngine._fmt(
                "flanqueo: atacar vecino débil donde no está tu broker",
                "El rival no pelea siempre en tu hotspot: golpea el flanco descubierto.",
                level)
        if pn == 62:
            return QuestionEngine._fmt(
                "decapitación: neutralizar vecinos del broker (bajar grado/influencia local)",
                "Aíslan a tu líder quitándole puentes. Mitiga con redundancia de brokers.",
                level)
        if pn == 63:
            return QuestionEngine._fmt(
                "trigger proxy: nuevo nodo con grado/influencia > p95 se vuelve visible",
                "Brokers muy visibles invitan contra-ataque. A veces conviene crecimiento discreto.",
                level)
        if pn == 64:
            if budget_df is not None and not budget_df.empty:
                return QuestionEngine._fmt(
                    f"asignación actual por alcaldía (n={len(budget_df)}): usar como semilla de optimización",
                    "Con presupuesto fijo, reparte brokers/horas según disputa vs consolidación — no a ojo.",
                    level)
            return QuestionEngine._fmt("sin asignación", "Genera presupuesto primero.", level)
        if pn == 65:
            if "campo" in terr.columns:
                disp = terr[terr.campo == "DISPUTA_ABIERTA"]["alcaldia"].tolist()
                cons = terr[terr.campo == "CONSOLIDACION"]["alcaldia"].tolist()
                return QuestionEngine._fmt(
                    f"disputa→brokers: {disp or '—'} | consolidación→horas: {cons or '—'}",
                    "Donde ya te quieren, manda brigada; donde pelean, manda liderazgo.",
                    level)
            return QuestionEngine._fmt("sin campo", "Clasifica campo territorial primero.", level)
        if pn == 66:
            return QuestionEngine._fmt(
                "score de optimización: historial de cobertura/ROI en sensibilidad presupuestaria",
                "Si al subir presupuesto el score se estanca, ya estás en zona óptima marginal.",
                level)

        # ----- P67–P77 resiliencia / inercia -----
        if pn == 67:
            if model is not None:
                sp = TerritorialAnalytics.spof(model)
                if sp is not None and not sp.empty:
                    return QuestionEngine._fmt(
                        f"SPOF candidatos: {sp.head(5).to_dict('records')}",
                        "Nodos cuya pérdida duele de más: tu talón de Aquiles.",
                        level)
            return QuestionEngine._fmt("SPOF vacío", "Corre análisis de red/SPOF.", level)
        if pn == 68:
            return QuestionEngine._fmt(
                "usa tabla SPOF: simpat_base vs simpat_sin_nodo → delta",
                "Cuánto te duele perder a cada líder. Prioriza protección de los críticos.",
                level)
        if pn == 69:
            return QuestionEngine._fmt(
                f"proxy resiliencia: densidad={net.get('density', 0):.4f}, gini_inf={net.get('gini_influence', float('nan'))}",
                "Si dependes de 2–3 nodos, la red es frágil; si aguanta perder el top, es más resiliente.",
                level)
        if pn == 70:
            a, v = top_alc("resistencia_prom")
            return QuestionEngine._fmt(
                f"resistencia_prom máx={v:.3f} en {a}",
                f"En {a} mandan estructuras formales/inercia: no basta convencer a uno solo.",
                level)
        if pn == 71:
            if "resistencia_prom" in terr.columns:
                r = terr.loc[terr.resistencia_prom.idxmax()]
                barr = 0.7 * float(r.resistencia_prom) + 0.3
                return QuestionEngine._fmt(
                    f"barrera≈{barr:.2f} en {r.alcaldia} (necesitas ~{100*barr:.0f}% vecinos alineados)",
                    "Alta inercia exige supermayoría local, no 50%.",
                    level)
            return QuestionEngine._fmt("sin resistencia", "Falta resistencia institucional.", level)
        if pn == 72:
            if "resistencia_prom" in terr.columns:
                x = terr[terr.resistencia_prom > 0.6]
                lista = ", ".join(x.alcaldia.tolist()) if not x.empty else "ninguna >0.6"
                return QuestionEngine._fmt(
                    f"resistencia>0.6: {lista}",
                    "Ahí habla primero con comités/liderazgo formal, no solo brigada de puerta en puerta.",
                    level)
            return QuestionEngine._fmt("sin datos", "Calcula resistencia por territorio.", level)
        if pn == 73:
            return QuestionEngine._fmt(
                "inercia normativa = resistencia_institucional alta que bloquea conversión sin presión de grupo",
                "La gente cambia menos por argumento individual que por norma local.",
                level)
        if pn == 74:
            return QuestionEngine._fmt(
                "fatiga territorial: demasiados intentos en poco tiempo suben resistencia efectiva",
                "Si visitas de más, se harta y deja de escuchar. Dosifica el contacto.",
                level)
        if pn == 75:
            return QuestionEngine._fmt(
                "intentar convencer con |diff|>>ε no convierte y puede polarizar más (costo doble)",
                "No gastes en extremos inamovibles: el costo es fallar y endurecer el conflicto.",
                level)
        if pn == 76:
            return QuestionEngine._fmt(
                "subir ε, bajar repulsión, o mezclar talleres de escucha entre grupos distantes",
                "Haz que circule información entre quienes piensan distinto.",
                level)
        if pn == 77:
            return QuestionEngine._fmt(
                "v6: catálogo P1–P77 alineado + niveles MACRO→INDIVIDUO_LIDER + dummies L1–L6 + org agregada + GIS público",
                "Más que un dashboard: laboratorio para estado, red, presupuesto, shocks, adversario, SPOF e inercia, "
                "con traits de líderes sintéticos para retroalimentar equipos reales sin PII de electores.",
                level)

        return QuestionEngine._fmt(
            "pregunta no resuelta por índice",
            "Selecciona una pregunta P1–P77 del catálogo.",
            level)


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.set_page_config(page_title="SITER-CAE v6.1 ADVANCED TERRITORIAL INFERENCE", page_icon="🧬", layout="wide")

st.title("🧬 SITER-CAE v6.1 · ADVANCED TERRITORIAL INFERENCE")
st.caption(
    "Laboratorio territorial CDMX · Mesa + GIS + presupuesto + brigadistas/GPS + "
    "datos reales y sintéticos + calibración · Jerarquía: CDMX → Alcaldía → UTM → Sección → Manzana"
)

if not MESA_OK:
    st.error("Mesa no está instalada. Revisa requirements.txt.")
    st.code(MESA_ERROR)
    st.stop()

if "siter" not in st.session_state:
    st.session_state.siter = {
        "model": None,
        "df": pd.DataFrame(),
        "meta": {},
        "gdf": None,
        "gps": pd.DataFrame(),
        "plan": pd.DataFrame(),
        "brig_report": pd.DataFrame(),
        "experiment_id": "",
        "hash": "",
        "gdf_joined": None,
        "real_reference": None,
        "field_observations": pd.DataFrame(),
        "downscaling": {},
        "org_layer": pd.DataFrame(),
        "dummy_geojson": None,
        "last_hypothesis": None,
    }
S = st.session_state.siter

# Sidebar
st.sidebar.header("⚙️ LABORATORIO")

mode_label = st.sidebar.selectbox("Fuente de datos", list(DataProvider.MODES.keys()))
mode = DataProvider.MODES[mode_label]
seed = st.sidebar.number_input("Seed", 1, 999999, 42)
n_units = st.sidebar.slider("Unidades sintéticas", 30, 1200, 300, 30)

if mode == "calib":
    calib_scenario = st.sidebar.selectbox(
        "Escenario de calibración",
        ["balance", "polarizacion", "fragmentacion", "consenso", "resistencia_alta", "red_hub"]
    )
else:
    calib_scenario = "balance"

base_file = electoral_file = socio_file = gis_file = gps_file = None
real_reference_df = None
use_dummy_gis = False  # se activa en bloque REAL o con checkbox global

if mode == "real":
    st.sidebar.markdown("### CSV")
    base_file = st.sidebar.file_uploader("CSV base territorial *", type=["csv"], key="base_csv")
    electoral_file = st.sidebar.file_uploader("CSV electoral agregado", type=["csv"], key="elec_csv")
    socio_file = st.sidebar.file_uploader("CSV socioeconómico agregado", type=["csv"], key="socio_csv")
    st.sidebar.markdown("### GIS")
    use_dummy_gis = st.sidebar.checkbox(
        "GIS dummy CDMX (polígonos de prueba)",
        value=False,
        help="Polígonos aproximados por alcaldía. NO son límites oficiales. Útil sin SHP real."
    )
    gis_file = st.sidebar.file_uploader(
        "SHP en ZIP / GeoJSON / GPKG",
        type=["zip", "geojson", "json", "gpkg"],
        key="gis_upload"
    )
    st.sidebar.markdown("### Observación de campo")
    gps_file = st.sidebar.file_uploader("GPS brigadistas: CSV / GPX / GeoJSON", type=["csv", "gpx", "geojson", "json"], key="gps_upload")

if mode == "coherent":
    st.sidebar.info("Primero carga un CSV real de referencia. El sintetizador conservará distribuciones y correlaciones aproximadas sin copiar filas.")

if mode in ["dummy", "pure", "calib"]:
    gps_file = st.sidebar.file_uploader("GPS brigadistas opcional", type=["csv", "gpx", "geojson", "json"], key="gps_upload_other")

st.sidebar.markdown("### GIS de prueba (cualquier modo)")
use_dummy_gis_global = st.sidebar.checkbox(
    "Activar GIS dummy CDMX",
    value=False,
    key="dummy_gis_global",
    help="Genera polígonos aproximados por alcaldía para probar el mapa sin archivo real."
)

st.sidebar.markdown("### Observaciones agregadas de campo")
obs_file = st.sidebar.file_uploader(
    "CSV observaciones · territorial_unit_id, variable, value",
    type=["csv"], key="field_obs_upload"
)

st.sidebar.markdown("### Organización / militancia (AGREGADA)")
org_file = st.sidebar.file_uploader(
    "CSV org · territorial_unit_id + rasgos agregados",
    type=["csv"], key="org_layer_upload"
)
org_min_n = st.sidebar.number_input("Mín. n_militantes por celda", 1, 50, 5)
st.sidebar.caption(
    "Sin PII. Celdas con n < mínimo se anulan. "
    "Columnas: n_militantes_obs, broker_density, org_mobilization, "
    "aceptacion_mensaje, asistencia_evento_rate, org_reliability."
)

st.sidebar.markdown("### Red")
use_geo_network = st.sidebar.checkbox("Red por proximidad geográfica", value=False)
geo_radius_km = st.sidebar.slider("Radio geo (km)", 0.5, 15.0, 3.0, 0.5)

behavior_name = st.sidebar.selectbox("Behavior Mesa", list(BEHAVIORS.keys()))
beta = st.sidebar.slider("β Voter", .1, 2.5, 1.2, .1)
epsilon = st.sidebar.slider("ε Deffuant", .05, 1.0, .40, .01)
mu = st.sidebar.slider("μ Deffuant", .05, .8, .30, .01)
epsilon_repulsion = st.sidebar.slider("ε repulsión", .5, 1.5, .80, .01)
coupling = st.sidebar.slider("Acoplamiento SAF", 0.0, 1.0, .20, .01)
field_pressure = st.sidebar.slider("Presión de campo", 0.0, 1.0, .08, .01)
p_intra = st.sidebar.slider("p intra-alcaldía", 0.0, .25, .06, .005)
p_inter = st.sidebar.slider("p inter-alcaldía", 0.0, .10, .015, .005)

st.sidebar.markdown("### Presupuesto")
budget = st.sidebar.number_input("Presupuesto operativo ($)", 0, 500000, 5000, 500)
fixed_brigada = st.sidebar.number_input("Costo fijo/brigada ($)", 0, 5000, 120, 20)
hour_cost = st.sidebar.number_input("Costo/hora ($)", 0, 1000, 120, 10)
hours_per_brigada = st.sidebar.number_input("Horas por brigada", 1, 24, 8)

st.sidebar.markdown("### Campo")
n_brig = st.sidebar.slider("Brigadas planificadas", 1, 30, 4)
gps_buffer = st.sidebar.slider("Tolerancia GPS (m)", 5, 100, 25, 5)

b1, b2 = st.sidebar.columns(2)
setup_clicked = b1.button("🔄 SETUP", use_container_width=True)
go_clicked = b2.button("▶ GO", use_container_width=True)

if setup_clicked:
    try:
        provider = DataProvider(seed=int(seed))

        S["gdf"] = None
        S["gdf_joined"] = None
        S["dummy_geojson"] = None
        S["real_reference"] = None
        S["field_observations"] = pd.DataFrame()
        S["downscaling"] = {}
        if obs_file is not None:
            S["field_observations"] = pd.read_csv(obs_file)

        S["org_layer"] = OrganizationLayer.load(org_file, min_n=int(org_min_n)) if org_file is not None else pd.DataFrame()

        if mode == "real":
            df, meta = provider.real(base_file, electoral_file, socio_file)
            if gis_file is not None:
                if not HAS_GIS:
                    raise RuntimeError("Para SHP/GeoJSON instala geopandas + pyogrio + shapely.")
                S["gdf"] = read_vector_upload(gis_file)
                S["gdf"] = normalize_gdf(S["gdf"])
                S["gdf_joined"] = join_base_to_geometry(df, S["gdf"])
            else:
                S["gdf"] = None
                S["gdf_joined"] = None

        elif mode == "coherent":
            if base_file is None:
                raise ValueError("Para SINTÉTICO COHERENTE debes cargar el CSV real de referencia.")
            reference = pd.read_csv(base_file)
            real_reference_df = normalize_base(reference, int(seed))
            S["real_reference"] = real_reference_df.copy()
            df, meta = provider.coherent_from_real(reference, n=n_units)
            S["gdf"] = None
            S["gdf_joined"] = None

        elif mode == "dummy":
            df, meta = provider.dummy(n_units)

        elif mode == "pure":
            df, meta = provider.pure(n_units)

        elif mode == "leaders":
            df, meta = provider.leaders(n=n_units, n_leaders=max(12, n_units // 8))

        else:
            df, meta = provider.calibration(n_units, calib_scenario)

        # GIS dummy (cualquier modo) o archivo real (modo real)
        use_dummy = bool(globals().get("use_dummy_gis_global", False))
        if mode == "real":
            use_dummy = use_dummy or bool(globals().get("use_dummy_gis", False))

        if use_dummy and (S.get("gdf") is None):
            gdf_dummy, gj_dummy = make_dummy_cdmx_gis()
            S["gdf"] = gdf_dummy
            S["dummy_geojson"] = gj_dummy
            if gdf_dummy is not None:
                S["gdf_joined"] = join_base_to_geometry(df, gdf_dummy)
            else:
                S["gdf_joined"] = None
            st.info("GIS dummy CDMX activo: polígonos aproximados de prueba (no oficiales).")

        # Fusionar capa organizacional agregada (si existe)
        df = OrganizationLayer.merge_into_base(df, S.get("org_layer", pd.DataFrame()))

        params = {
            "beta": beta,
            "epsilon": epsilon,
            "mu": mu,
            "epsilon_repulsion": epsilon_repulsion,
            "coupling": coupling,
            "field_pressure": field_pressure,
            "use_geo_network": bool(use_geo_network),
            "geo_radius_km": float(geo_radius_km),
        }

        model = SITERModel(
            df,
            behavior_name=behavior_name,
            seed=int(seed),
            p_intra=p_intra,
            p_inter=p_inter,
            params=params,
        )

        # Persistir dataset y modelo
        S["df"] = df
        S["meta"] = meta
        S["model"] = model
        S["gps"] = FieldOperations.load_gps(gps_file) if gps_file is not None else pd.DataFrame()
        S["plan"] = FieldOperations.plan_from_territories(
            model.agent_dataframe(), n_brig=n_brig
        )
        S["brig_report"] = FieldOperations.coverage_report(
            S["plan"], S["gps"], buffer_m=gps_buffer
        )

        config = {
            "mode": mode,
            "seed": int(seed),
            "n_units": len(df),
            "behavior": behavior_name,
            "beta": beta,
            "epsilon": epsilon,
            "mu": mu,
            "epsilon_repulsion": epsilon_repulsion,
            "coupling": coupling,
            "field_pressure": field_pressure,
            "p_intra": p_intra,
            "p_inter": p_inter,
            "budget": budget,
        }
        S["experiment_id"] = "EXP-" + sha256_obj(config)[:12]
        S["hash"] = sha256_obj({
            "config": config,
            "data_meta": meta,
            "initial_head": df.head(25).to_dict("records"),
        })
        st.success(f"SETUP OK · {meta.get('mode')} · {len(df):,} unidades · {S['experiment_id']}")
    except Exception as exc:
        st.error(f"SETUP falló: {exc}")

if go_clicked:
    if S["model"] is None:
        st.warning("Primero ejecuta SETUP.")
    else:
        S["model"].step()
        S["plan"] = FieldOperations.plan_from_territories(
            S["model"].agent_dataframe(), n_brig=n_brig
        )
        S["brig_report"] = FieldOperations.coverage_report(
            S["plan"], S["gps"], buffer_m=gps_buffer
        )
        st.rerun()

if S["model"] is None:
    st.info("Ejecuta SETUP. Elige REAL para CSV+SHP; COHERENTE para sintetizar desde un CSV real; DUMMY/PURE/CALIB para pruebas.")
    st.markdown("""
### Modos de datos

| Modo | Uso |
|---|---|
| **REAL · CSV + SHP/GeoJSON** | Base real agregada + geografía real |
| **DUMMY** | Smoke tests rápidos |
| **SINTÉTICO COHERENTE** | Sintético estadísticamente derivado de un CSV real |
| **SINTÉTICO PURO** | Generación libre controlada por seed |
| **SINTÉTICO CALIBRACIÓN** | Pruebas de consenso, polarización, fragmentación, resistencia y redes |

### Qué se calibra

**Datos → indicadores → Mesa → escenarios → presupuesto → brigadas/GPS → respuestas cliente → export reproducible.**
""")
    st.stop()

model = S["model"]
adf = ensure_agent_frame_columns(model.agent_dataframe())
terr = TerritorialAnalytics.table(adf)
net = TerritorialAnalytics.network(model)
budget_df = BudgetEngine.allocation(
    terr, budget, fixed_per_brigada=fixed_brigada,
    hour_cost=hour_cost, hours_per_brigada=hours_per_brigada
)

# ---------------------------------------------------------------------
# Monitores estilo NetLogo
# ---------------------------------------------------------------------
st.markdown("### 🎛️ WORLD CONTROL")

m = model.metrics()
cols = st.columns(7)
cols[0].metric("Tick", m["step"])
cols[1].metric("SIMPATIZANTE", f"{100*m['SIMPATIZANTE']:.1f}%")
cols[2].metric("OPOSITOR", f"{100*m['OPOSITOR']:.1f}%")
cols[3].metric("INDECISO", f"{100*m['INDECISO']:.1f}%")
cols[4].metric("Gini", f"{m['Gini']:.3f}")
cols[5].metric("Polarización", f"{m['Polarizacion']:.3f}")
cols[6].metric("Aristas", f"{m['edges']:,}")

tabs = st.tabs([
    "🌐 WORLD GIS / MESA",
    "📊 ESTADÍSTICA",
    "💰 PRESUPUESTO",
    "🥾 BRIGADISTAS / GPS",
    "🧬 DESAGREGACIÓN",
    "🧪 CALIBRACIÓN",
    "🎯 CLIENTE",
    "📤 EXPORT",
])

# ---------------------------------------------------------------------
# World
# ---------------------------------------------------------------------
with tabs[0]:
    st.subheader("World View · GIS real cuando existe; red territorial como fallback")

    rendered_gis = False
    if S.get("gdf_joined") is not None and HAS_GIS:
        g = S["gdf_joined"].copy()
        try:
            import pydeck as pdk
            gj = json.loads(g.to_json())
            layer = pdk.Layer(
                "GeoJsonLayer",
                gj,
                pickable=True,
                stroked=True,
                filled=True,
                get_fill_color="[100, 150, 220, 90]",
                get_line_color="[50, 50, 50, 180]",
                line_width_min_pixels=1,
            )
            view = pdk.ViewState(
                latitude=float(adf["lat"].mean()),
                longitude=float(adf["lon"].mean()),
                zoom=10.5,
            )
            st.pydeck_chart(pdk.Deck(
                layers=[layer],
                initial_view_state=view,
                tooltip={"text": "{alcaldia} | {seccion}"},
            ), use_container_width=True)
            st.caption("Capa GIS unida al CSV (o dummy). Atributos por territorial_unit_id/seccion cuando hay llave.")
            rendered_gis = True
        except Exception as exc:
            st.warning(f"No se pudo renderizar la capa GIS interactiva: {exc}")

    if (not rendered_gis) and S.get("dummy_geojson"):
        try:
            import pydeck as pdk
            gj = S["dummy_geojson"]
            layer = pdk.Layer(
                "GeoJsonLayer",
                gj,
                pickable=True,
                stroked=True,
                filled=True,
                get_fill_color="[100, 180, 140, 100]",
                get_line_color="[40, 80, 40, 200]",
                line_width_min_pixels=1,
            )
            view = pdk.ViewState(
                latitude=float(adf["lat"].mean()) if len(adf) else 19.35,
                longitude=float(adf["lon"].mean()) if len(adf) else -99.15,
                zoom=10.2,
            )
            st.pydeck_chart(pdk.Deck(
                layers=[layer],
                initial_view_state=view,
                tooltip={"text": "{alcaldia}"},
            ), use_container_width=True)
            st.caption("GIS dummy CDMX (sin geopandas o sin join). Polígonos aproximados de prueba — no oficiales.")
            rendered_gis = True
        except Exception as exc:
            st.warning(f"No se pudo renderizar GIS dummy: {exc}")

    if not rendered_gis:
        # Fallback visual: red + agentes, no se presenta como cartografía oficial.
        pos = nx.spring_layout(model.G, seed=int(seed), iterations=35)
        ex, ey = [], []
        for u, v in model.G.edges():
            ex += [pos[u][0], pos[v][0], None]
            ey += [pos[u][1], pos[v][1], None]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=1), name="red"))
        fig.add_trace(go.Scatter(
            x=[pos[str(a.unique_id)][0] for a in model.agents],
            y=[pos[str(a.unique_id)][1] for a in model.agents],
            mode="markers",
            text=[f"{a.alcaldia}<br>{a.seccion}" for a in model.agents],
            marker=dict(
                size=[14 if a.es_broker else 8 for a in model.agents],
                color=[a.opinion for a in model.agents],
                colorscale="RdBu", cmin=-1, cmax=1,
                showscale=True,
            ),
            name="agentes",
        ))
        fig.update_layout(height=650, template="plotly_dark", xaxis_visible=False, yaxis_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        st.info("No hay SHP/GeoJSON unido. Esta vista es World View de red, no cartografía oficial.")

    st.markdown("#### Estado territorial")
    st.dataframe(terr, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------
with tabs[1]:
    st.subheader("Indicadores estadísticos para decisión territorial")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.bar(terr, x="alcaldia", y="simpat_pct", color="campo",
                   color_discrete_map=FIELD_COLORS,
                   title="Estado agregado por alcaldía"),
            use_container_width=True
        )
    with c2:
        st.plotly_chart(
            px.scatter(terr, x="resistencia_prom", y="prioridad_prom",
                       size="n_unidades", color="polarizacion_std",
                       hover_name="alcaldia",
                       title="Prioridad × resistencia × polarización"),
            use_container_width=True
        )

    st.markdown("#### Red")
    st.json(net)
    st.dataframe(TerritorialAnalytics.spof(model), use_container_width=True, hide_index=True)

    st.markdown("#### Series Mesa")
    hist = pd.DataFrame(model.history)
    if not hist.empty:
        st.plotly_chart(
            px.line(hist, x="step", y=["SIMPATIZANTE","OPOSITOR","INDECISO"],
                    title="Evolución de estados"),
            use_container_width=True
        )

# ---------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------
with tabs[2]:
    st.subheader("Presupuesto · cobertura · costo · sensibilidad · Monte Carlo")

    bcols = st.columns(6)
    bcols[0].metric("Presupuesto", f"${budget:,.0f}")
    bcols[1].metric("Costo asignado", f"${budget_df.costo.sum():,.0f}")
    bcols[2].metric("Brigadas", int(budget_df.brigadas.sum()))
    bcols[3].metric("Horas", int(budget_df.horas.sum()))
    bcols[4].metric("Cobertura est.", f"{100*budget_df.cobertura_est.mean():.1f}%")
    bcols[5].metric("Beneficio proxy", f"{budget_df.beneficio_operacional.sum():.2f}")

    st.dataframe(
        budget_df.sort_values("workload", ascending=False),
        use_container_width=True, hide_index=True
    )

    st.markdown("#### Sensibilidad presupuestaria")
    budgets = sorted(set([
        max(0, int(budget*.5)),
        max(0, int(budget)),
        int(budget*1.5),
        int(budget*2),
    ]))
    sens = BudgetEngine.sensitivity(terr, budgets)
    st.plotly_chart(
        px.line(sens, x="presupuesto", y="cobertura_est_pct",
                markers=True, title="Cobertura estimada vs presupuesto"),
        use_container_width=True
    )
    st.dataframe(sens, use_container_width=True, hide_index=True)

    st.markdown("#### Rendimiento marginal del presupuesto")
    marginal = BudgetEngine.marginal_analysis(terr, budgets)
    st.dataframe(marginal, use_container_width=True, hide_index=True)
    st.markdown("#### Costo por unidad y productividad")
    perf_budget = BudgetEngine.performance_table(terr, budget, fixed_brigada, hour_cost, hours_per_brigada)
    st.dataframe(perf_budget, use_container_width=True, hide_index=True)
    if perf_budget.attrs.get("summary"):
        st.json(perf_budget.attrs["summary"])

    st.markdown("#### Monte Carlo")
    reps = st.slider("Repeticiones Monte Carlo", 50, 1000, 300, 50, key="mc_reps")
    mc = BudgetEngine.monte_carlo(
        terr, budget, reps=reps,
        fixed_per_brigada=fixed_brigada,
        hour_cost=hour_cost,
        hours_per_brigada=hours_per_brigada,
        seed=int(seed)
    )
    S["mc"] = mc
    if not mc.empty:
        cs = BudgetEngine.confidence_summary(mc)
        st.write({
            "media_cobertura_pct": round(cs["media"], 2),
            "mediana_cobertura_pct": round(cs["mediana"], 2),
            "P05": round(cs["p05"], 2),
            "P25": round(cs["p25"], 2),
            "P75": round(cs["p75"], 2),
            "P95": round(cs["p95"], 2),
            "P(cobertura ≥80%)": round(cs["prob_cobertura_80"], 3),
            "P(cobertura ≥90%)": round(cs["prob_cobertura_90"], 3),
        })
        st.plotly_chart(
            px.histogram(mc, x="cobertura_pct", nbins=30,
                         title="Distribución Monte Carlo de cobertura"),
            use_container_width=True
        )

# ---------------------------------------------------------------------
# Brigadistas
# ---------------------------------------------------------------------
with tabs[3]:
    st.subheader("Trabajo de brigadistas · plan vs observación GPS")

    s = FieldOperations.summary(S["plan"], S["gps"], S["brig_report"])
    c = st.columns(5)
    c[0].metric("Km plan", f"{s['km_plan_total']:.2f}")
    c[1].metric("Km GPS", f"{s['km_gps_total']:.2f}")
    c[2].metric("Cobertura ruta", f"{s['cobertura_promedio_pct']:.1f}%")
    c[3].metric("Brigadas con GPS", s["brigadas_con_gps"])
    c[4].metric("Territorios observados", s["territorios_observados"])

    st.caption(
        "La cobertura se calcula por coincidencia espacial de la traza GPS con el plan, "
        "no como GPS/plan, para evitar contar vueltas adicionales como cobertura."
    )

    if not S["brig_report"].empty:
        st.dataframe(S["brig_report"], use_container_width=True, hide_index=True)
        st.markdown("#### Estadística de productividad y cumplimiento")
        brig_perf = FieldOperations.performance(S["plan"], S["gps"], S["brig_report"])
        st.dataframe(brig_perf, use_container_width=True, hide_index=True)
        st.json({
            "cobertura_media_pct": float(brig_perf["cobertura_ruta_pct"].mean()),
            "cumplimiento_territorial_medio_pct": float(brig_perf["cumplimiento_territorial_pct"].mean()),
            "desviacion_media_km": float(brig_perf["desviacion_km"].mean()),
            "indice_productividad_medio": float(brig_perf["indice_productividad"].mean()),
        })

    # Mapa de campo
    if not S["plan"].empty:
        fig = go.Figure()
        for b, p in S["plan"].groupby("brigada"):
            p = p.sort_values("orden")
            fig.add_trace(go.Scattermap(
                lat=p["lat"], lon=p["lon"], mode="lines+markers",
                name=f"Plan {b}", hovertext=p["territorial_unit_id"]
            ))
        if not S["gps"].empty:
            for b, gg in S["gps"].groupby("brigada"):
                gg = gg.sort_values("timestamp")
                fig.add_trace(go.Scattermap(
                    lat=gg["lat"], lon=gg["lon"], mode="lines",
                    name=f"GPS {b}"
                ))
        fig.update_layout(
            map=dict(style="open-street-map", zoom=10),
            height=600,
            margin=dict(l=0,r=0,t=0,b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Plan territorial")
    st.dataframe(S["plan"], use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# Advanced downscaling
# ---------------------------------------------------------------------
with tabs[4]:
    st.subheader("🧬 Desagregación territorial avanzada")
    st.info(
        "Regla metodológica: la GPS mide operación/cobertura; no es una observación directa de "
        "desconfianza, arraigo, capital social ni intención. Los rasgos solo se actualizan cuando "
        "existe una variable observada explícita y agregada por unidad territorial."
    )

    dtab1, dtab2, dtab3, dtab4 = st.tabs([
        "Máxima entropía / IPF", "Tomografía de red", "Bayes / Kalman", "MRF / CRF espacial"
    ])

    with dtab1:
        st.markdown("### Reparto de un total macro conservando márgenes")
        n = min(len(adf), 12)
        head = adf.head(n).copy()
        if "poblacion" in head.columns:
            pop = pd.to_numeric(head["poblacion"], errors="coerce").fillna(1000.0).to_numpy(float)
        else:
            pop = np.full(n, 1000.0, dtype=float)
            st.caption("Sin columna `poblacion` en el dataframe de agentes: peso uniforme 1000 (proxy).")
        if "prioridad_problema" in head.columns:
            prio = pd.to_numeric(head["prioridad_problema"], errors="coerce").fillna(0.5).to_numpy(float)
        else:
            prio = np.full(n, 0.5, dtype=float)
        prior = np.maximum(pop[:, None] * (0.5 + prio[:, None]), 1e-6)
        prior = np.repeat(prior, 3, axis=1)
        total = float(prior.sum())
        row_targets = prior.sum(axis=1) * 1.03
        row_targets *= total / max(row_targets.sum(), 1e-12)
        col_targets = np.array([total * .34, total * .33, total * .33])
        col_targets *= row_targets.sum() / max(col_targets.sum(), 1e-12)
        try:
            P, diag = AdvancedDownscalingEngine.max_entropy_ipf(prior, row_targets, col_targets)
            st.json(diag)
            idx_labels = head["territorial_unit_id"] if "territorial_unit_id" in head.columns else head.index
            st.dataframe(
                pd.DataFrame(P, index=idx_labels,
                             columns=["componente_1", "componente_2", "componente_3"]),
                use_container_width=True,
            )
            st.caption("Ejemplo reproducible sobre datos agregados actuales. Para producción, los márgenes deben venir de restricciones macro reales.")
        except Exception as exc:
            st.error(str(exc))

    with dtab2:
        st.markdown("### Tomografía de flujos territoriales")
        st.caption("La tomografía requiere restricciones OD agregadas; no infiere personas ni trayectorias individuales.")
        nodes = list(model.G.nodes())
        if len(nodes) >= 3:
            od = {(nodes[0], nodes[-1]): 100.0, (nodes[1], nodes[-2]): 60.0}
            tomo = AdvancedDownscalingEngine.network_tomography(model.G, od)
            st.json(tomo["diagnostics"])
            ef = pd.DataFrame([{"arista": k, "flujo_estimado": v} for k,v in tomo["edge_fluxes"].items()])
            st.dataframe(ef.sort_values("flujo_estimado", ascending=False).head(25), use_container_width=True, hide_index=True)
        else:
            st.warning("Se requieren al menos 3 nodos.")

    with dtab3:
        st.markdown("### Actualización de rasgos territoriales observados")
        obs = S.get("field_observations", pd.DataFrame())
        if obs.empty:
            st.warning("Carga un CSV de observaciones agregadas para activar Bayes/Kalman. Esquema mínimo: territorial_unit_id, variable, value.")
        else:
            try:
                ot = AdvancedDownscalingEngine.observation_table(obs)
                st.dataframe(ot, use_container_width=True, hide_index=True)
                vars_ = sorted([v for v in ot["variable"].unique().tolist() if v in FIELD_UPDATABLE_VARS])
                blocked = sorted(set(ot["variable"].unique()) - FIELD_UPDATABLE_VARS)
                if blocked:
                    st.warning(f"Variables bloqueadas (no actualizables por campo/GPS): {', '.join(blocked)}")
                if not vars_:
                    st.error("Ninguna variable observada está en la lista blanca de rasgos actualizables.")
                    var = None
                else:
                    var = st.selectbox("Variable observada", vars_, key="adv_var")
                if var is None:
                    pass
                else:
                    o = ot[ot.variable == var].copy()
                    a = adf[["territorial_unit_id", var]].copy() if var in adf.columns else pd.DataFrame()
                    if a.empty:
                        st.warning("La variable observada no existe en el estado actual; puede utilizarse como nueva serie territorial, pero debe definirse su prior.")
                    else:
                        merged = a.merge(o, on="territorial_unit_id", how="inner")
                        if merged.empty:
                            st.warning("No hay unidades territoriales en común entre el estado y las observaciones.")
                        else:
                            prior = pd.to_numeric(merged[var], errors="coerce").fillna(.5).to_numpy(float)
                            prior_var = np.full(len(merged), .04)
                            post, post_var = AdvancedDownscalingEngine.bayesian_trait_update(
                                prior, prior_var, merged.observed_mean.to_numpy(float),
                                merged.observed_variance.to_numpy(float),
                                reliability=float(merged.reliability.mean())
                            )
                            kalman, kal_var, gain = AdvancedDownscalingEngine.kalman_trait_update(
                                prior, prior_var, merged.observed_mean.to_numpy(float),
                                merged.observed_variance.to_numpy(float)
                            )
                            result = merged[["territorial_unit_id"]].copy()
                            result["prior"] = prior
                            result["observado"] = merged.observed_mean.to_numpy(float)
                            result["posterior_bayes"] = post
                            result["posterior_kalman"] = kalman
                            result["incertidumbre_bayes"] = post_var
                            result["ganancia_kalman"] = gain
                            st.dataframe(result, use_container_width=True, hide_index=True)
                            S["downscaling"]["bayes"] = result.to_dict("records")
            except Exception as exc:
                st.error(str(exc))

    with dtab4:
        st.markdown("### Suavizado espacial MRF/CRF-like")
        st.caption("Suaviza estimaciones territoriales; no aumenta la resolución de la evidencia ni crea datos observados.")
        if "prioridad_problema" in adf.columns and len(adf) > 2:
            vals = adf["prioridad_problema"].to_numpy(float)
            uid_to_i = {str(u): i for i,u in enumerate(adf["territorial_unit_id"])}
            adjacency = [[] for _ in range(len(adf))]
            for u,v in model.G.edges():
                if str(u) in uid_to_i and str(v) in uid_to_i:
                    i,j=uid_to_i[str(u)],uid_to_i[str(v)]
                    adjacency[i].append(j); adjacency[j].append(i)
            smooth, diag = AdvancedDownscalingEngine.spatial_mrf_smooth(vals, adjacency, strength=.25)
            out = adf[["territorial_unit_id","alcaldia","prioridad_problema"]].copy()
            out["prioridad_suavizada_mrf"] = smooth
            st.json(diag)
            st.dataframe(out, use_container_width=True, hide_index=True)
        else:
            st.warning("No hay variable territorial suficiente para suavizar.")

# ---------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------
with tabs[5]:
    st.subheader("Calibración: pruebas antes de datos reales")

    st.markdown("""
**Secuencia recomendada**

1. DUMMY → comprobar que la aplicación responde.
2. CALIBRACIÓN → probar consenso, polarización, fragmentación, resistencia y red.
3. SINTÉTICO PURO → stress test de tamaño.
4. SINTÉTICO COHERENTE → verificar que el comportamiento conserve patrones estadísticos del CSV real.
5. REAL + SHP → producción analítica.
6. Organización agregada (CSV) → amplifica coupling SAF vía broker_density / org_mobilization.

Esto permite separar **error de software**, **error de datos** y **error de modelo**.
""")
    if S.get("org_layer") is not None and not S["org_layer"].empty:
        st.success(f"Capa organización activa: {len(S['org_layer'])} unidades agregadas (sin PII).")
        st.dataframe(S["org_layer"].head(20), use_container_width=True, hide_index=True)
    else:
        st.info("Sin CSV de organización. El modelo corre con broker_density=0 (sin amplificación org).")

    cal_summary = pd.DataFrame([
        {"test": "balance", "objetivo": "distribución sin polarización extrema"},
        {"test": "polarizacion", "objetivo": "dos polos separados"},
        {"test": "fragmentacion", "objetivo": "múltiples estados"},
        {"test": "consenso", "objetivo": "convergencia"},
        {"test": "resistencia_alta", "objetivo": "inercia institucional"},
        {"test": "red_hub", "objetivo": "dependencia de hubs"},
    ])
    st.dataframe(cal_summary, use_container_width=True, hide_index=True)

    st.markdown("#### Métricas de validación")
    validation = {
        "n": len(adf),
        "opinion_mean": float(adf.opinion.mean()),
        "opinion_std": float(adf.opinion.std()),
        "gini": float(model.compute_gini()),
        "network_density": float(net.get("density", 0)),
        "network_gini": float(net.get("gini_influence", 0)),
        "seed": int(seed),
    }
    st.json(validation)

    if S.get("real_reference") is not None:
        cmp = CalibrationEngine.compare(S["real_reference"], S["df"])
        score = CalibrationEngine.score(S["real_reference"], S["df"])
        st.markdown("#### Calibración contra CSV real de referencia")
        st.metric("Score de similitud distributiva", f"{score:.1f}/100" if score is not None else "—")
        st.dataframe(cmp, use_container_width=True, hide_index=True)
        if score is not None:
            st.info("Interpretación: ≥80 buena similitud distributiva; 60–79.9 intermedia; <60 requiere revisar generador/variables. Este score no valida por sí solo el modelo causal.")

# ---------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------
with tabs[6]:
    st.subheader("🎯 Question / Hypothesis Engine v6 · P1–P77 + pregunta libre")
    st.caption(
        "Catálogo alineado al PDF SITER 77 preguntas. "
        "Respuesta = métrica técnica + narrativa cliente + nivel. "
        "INDIVIDUO_LIDER usa traits sintéticos de líderes/equipos (no PII de electores). "
        "La pregunta libre se enruta de forma reproducible a hipótesis y P's."
    )

    mode_q = st.radio(
        "Modo de consulta",
        ["Catálogo P1–P77", "Pregunta libre (Hypothesis Engine)"],
        horizontal=True,
        key="client_query_mode",
    )

    cqa, cqb = st.columns([2, 1])
    with cqb:
        qlevel = st.selectbox("Nivel de análisis", QuestionEngine.LEVELS, index=1, key="client_level")

    if mode_q.startswith("Catálogo"):
        with cqa:
            question = st.selectbox("Pregunta (P1–P77)", QuestionEngine.CATALOG, key="client_catalog_q")
        answer = QuestionEngine.answer(
            question, adf, terr, net,
            budget_df=budget_df,
            brig_report=S["brig_report"],
            model=model,
            level=qlevel,
            experiment_id=S.get("experiment_id", ""),
            output_hash=S.get("hash", ""),
            seed=int(seed),
        )
        st.markdown(answer)
    else:
        with cqa:
            free_q = st.text_area(
                "Pregunta libre del candidato / equipo",
                placeholder="Ej. ¿Por qué está aumentando la polarización en Xochimilco y qué pasa si bajo el presupuesto 25%?",
                height=100,
                key="client_free_q",
            )
        examples = st.multiselect(
            "Atajos de ejemplo",
            [
                "¿Dónde está más polarizado y en disputa abierta?",
                "¿Qué tan resiliente es mi red y cuáles son los SPOF?",
                "Tengo poco presupuesto: ¿qué territorio tiene mejor ROI?",
                "¿Mis brigadas saturan territorio o hay huecos de cobertura?",
                "¿Qué pasa si inserto un broker fuerte en Iztapalapa?",
                "¿Dónde hay alta inercia institucional y debo hablar con comités?",
                "¿Por qué no converge la opinión y hay cámara de eco?",
            ],
            key="client_free_examples",
        )
        query_text = free_q.strip() or (" ".join(examples) if examples else "")
        if st.button("🔎 Analizar pregunta libre", type="primary", key="run_hypothesis"):
            if not query_text:
                st.warning("Escribe una pregunta o elige un atajo.")
            else:
                report = HypothesisEngine.run(
                    query_text, adf, terr, net,
                    budget_df=budget_df,
                    brig_report=S["brig_report"],
                    model=model,
                    level_override=qlevel if qlevel else None,
                    experiment_id=S.get("experiment_id", ""),
                    output_hash=S.get("hash", ""),
                    seed=int(seed),
                )
                # if user fixed level in selector, prefer it
                if qlevel:
                    report["level"] = qlevel
                S["last_hypothesis"] = report
                st.markdown(HypothesisEngine.render_markdown(report))
                with st.expander("JSON del enrutamiento (reproducible)"):
                    st.json({
                        "parsed": report["parsed"],
                        "gaps": report["gaps"],
                        "p_numbers": report["parsed"]["p_numbers"],
                    })
        elif S.get("last_hypothesis"):
            st.info("Último informe de pregunta libre (re-ejecuta para actualizar con nuevos ticks):")
            st.markdown(HypothesisEngine.render_markdown(S["last_hypothesis"]))

    if "es_lider" in adf.columns and (adf["es_lider"] == 1).any():
        st.markdown("#### Capa líderes / traits individuales sintéticos")
        st.dataframe(
            adf[adf["es_lider"] == 1][
                [c for c in [
                    "territorial_unit_id", "alcaldia", "saf_skill", "capital_social",
                    "influencia_liderazgo", "arraigo", "desconfianza", "influencia",
                    "broker_density", "org_mobilization"
                ] if c in adf.columns]
            ].sort_values("saf_skill", ascending=False),
            use_container_width=True, hide_index=True,
        )

    st.markdown("#### Indicadores territoriales de soporte")
    st.dataframe(terr, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------
with tabs[7]:
    st.subheader("📤 Export reproducible")

    config = {
        "version": "SITER-CAE-v6.1-ADVANCED-TERRITORIAL-INFERENCE",
        "experiment_id": S["experiment_id"],
        "seed": int(seed),
        "mode": mode,
        "behavior": behavior_name,
        "n_units": len(adf),
        "budget": budget,
        "fixed_brigada": fixed_brigada,
        "hour_cost": hour_cost,
        "hours_per_brigada": hours_per_brigada,
    }

    payload = {
        "metadata": {
            **config,
            "output_hash": S["hash"],
            "data_meta": S["meta"],
            "governance": {
                "personal_data": False,
                "aggregated": True,
                "synthetic_supported": True,
                "geography": "CDMX",
                "organization_layer": {
                    "present": bool(len(S.get("org_layer", pd.DataFrame())) > 0),
                    "min_n_cell": int(org_min_n) if "org_min_n" in dir() else 5,
                    "rows": int(len(S.get("org_layer", pd.DataFrame()))),
                },
            },
        },
        "model": model.metrics(),
        "territorial_fields": terr.to_dict("records"),
        "network": net,
        "budget": budget_df.to_dict("records"),
        "brigadistas": S["brig_report"].to_dict("records"),
        "advanced_downscaling": S.get("downscaling", {}),
        "history": model.history,
    }

    st.code(json.dumps(payload["metadata"], ensure_ascii=False, indent=2))

    st.download_button(
        "⬇️ JSON completo",
        data=json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        file_name=f"{S['experiment_id']}.json",
        mime="application/json",
    )

    st.download_button(
        "⬇️ CSV territorial",
        data=terr.to_csv(index=False).encode("utf-8"),
        file_name=f"{S['experiment_id']}_territorial.csv",
        mime="text/csv",
    )

    if not S["brig_report"].empty:
        st.download_button(
            "⬇️ CSV brigadistas",
            data=S["brig_report"].to_csv(index=False).encode("utf-8"),
            file_name=f"{S['experiment_id']}_brigadistas.csv",
            mime="text/csv",
        )

    st.markdown("""
### Trazabilidad

`seed → datos → modelo → simulación → indicadores → presupuesto/brigadas → respuesta → hash`

El hash identifica la configuración/salida exportada; el seed permite repetir el
experimento cuando la misma fuente de datos y configuración estén disponibles.
""")

st.sidebar.markdown("---")
st.sidebar.caption("SITER-CAE v6.1 · Mesa · GIS · Org agregada · Presupuesto · Campo/GPS · Sin PII")
