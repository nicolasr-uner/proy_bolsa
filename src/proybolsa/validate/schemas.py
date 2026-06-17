"""Esquemas Pandera para validar los DataFrames en cada etapa de la ingesta.

Uso:
    from proybolsa.validate.schemas import PRECIO_BOLSA_HORARIO
    PRECIO_BOLSA_HORARIO.validate(df)   # lanza SchemaError si hay violaciones
    # o bien con coerce=True para forzar tipos antes de validar:
    PRECIO_BOLSA_HORARIO.validate(df, lazy=True)
"""

from __future__ import annotations

import pandera as pa

# ---------------------------------------------------------------------------
# Precio de bolsa horario
# ---------------------------------------------------------------------------

PRECIO_BOLSA_HORARIO = pa.DataFrameSchema(
    {
        "timestamp": pa.Column(
            "datetime64[ns]",
            nullable=False,
            description="Timestamp de inicio de la hora (UTC-5, hora Colombia)",
        ),
        "precio_bolsa": pa.Column(
            float,
            checks=[
                pa.Check.ge(0, error="precio_bolsa no puede ser negativo"),
                # El precio puede superar el precio de escasez (~906) durante crisis hidrologicas.
                # El maximo historico registrado en el SIN supera 3000 COP/kWh en El Nino extremo.
                pa.Check.le(4000, error="precio_bolsa supera el limite de 4000 COP/kWh (revisar datos)"),
            ],
            nullable=False,
            description="COP/kWh",
        ),
    },
    checks=[
        pa.Check(
            lambda df: df["timestamp"].is_monotonic_increasing,
            error="timestamps no estan en orden cronologico estricto",
        )
    ],
    name="precio_bolsa_horario",
)

# ---------------------------------------------------------------------------
# Hidrologia diaria
# ---------------------------------------------------------------------------

HIDROLOGIA_DIARIA = pa.DataFrameSchema(
    {
        "fecha": pa.Column("object", nullable=False),  # date objects
        "aportes_pct": pa.Column(
            float,
            checks=[pa.Check.ge(0), pa.Check.le(600)],
            nullable=True,
            description="% de la media historica",
        ),
        "volumen_util_pct": pa.Column(
            float,
            checks=[pa.Check.ge(0), pa.Check.le(105)],
            nullable=True,
            description="% de la capacidad util total",
        ),
    },
    name="hidrologia_diaria",
)

# ---------------------------------------------------------------------------
# Demanda SIN diaria
# ---------------------------------------------------------------------------

DEMANDA_DIARIA = pa.DataFrameSchema(
    {
        "fecha": pa.Column("object", nullable=False),
        "demanda_sin": pa.Column(
            float,
            checks=[pa.Check.ge(0)],
            nullable=True,
            description="kWh",
        ),
    },
    name="demanda_diaria",
)

# ---------------------------------------------------------------------------
# Variables macro mensuales
# ---------------------------------------------------------------------------

MACRO_MENSUAL = pa.DataFrameSchema(
    {
        "fecha": pa.Column("object", nullable=False),
        "trm": pa.Column(
            float,
            checks=[pa.Check.ge(1000), pa.Check.le(10000)],
            nullable=True,
            description="COP/USD",
        ),
        "brent": pa.Column(
            float,
            checks=[pa.Check.ge(0), pa.Check.le(400)],
            nullable=True,
            description="USD/bbl",
        ),
        "ppi_usa": pa.Column(
            float,
            checks=[pa.Check.ge(0)],
            nullable=True,
            description="Indice PPI all commodities EE.UU.",
        ),
    },
    name="macro_mensual",
)

# ---------------------------------------------------------------------------
# IPP Colombia mensual
# ---------------------------------------------------------------------------

IPP_MENSUAL = pa.DataFrameSchema(
    {
        "fecha": pa.Column("object", nullable=False),
        "ipp": pa.Column(
            float,
            checks=[pa.Check.ge(50), pa.Check.le(500)],
            nullable=False,
            description="Indice IPP Oferta Interna, base dic-2014=100",
        ),
    },
    name="ipp_mensual",
)

# ---------------------------------------------------------------------------
# ONI mensual
# ---------------------------------------------------------------------------

ONI_MENSUAL = pa.DataFrameSchema(
    {
        "fecha": pa.Column("object", nullable=False),
        "oni": pa.Column(
            float,
            checks=[pa.Check.ge(-4), pa.Check.le(4)],
            nullable=False,
            description="Anomalia TSM Nino 3.4 (3-month average)",
        ),
    },
    name="oni_mensual",
)
