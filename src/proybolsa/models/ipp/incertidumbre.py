"""Calibración empírica de los intervalos del IPP.

EL PROBLEMA
-----------
Las bandas del IPP son nominalmente del 90% y la cobertura real, medida sobre 56 orígenes del
rolling-origin, va de **44% a 75%** según el modelo y el horizonte. Un intervalo que dice 90% y
cubre 57% no es un intervalo: es decoración, y peor que no tener banda, porque induce una
confianza que no corresponde.

La causa es de dónde salían: `EnsembleIPP.predict` tomaba el ancho de un componente (antes el
SARIMAX, hoy el ARIMA) y lo trasplantaba centrado en la predicción del ensemble. Ese ancho es el
intervalo teórico de OTRO modelo, condicional a que su especificación sea correcta, con los
parámetros tratados como conocidos y sin incertidumbre sobre las exógenas futuras. Nada de eso
se cumple.

LA SOLUCIÓN
-----------
Medir los errores reales del backtest y usar sus cuantiles. Si el error log a 12 meses tiene un
cuantil 0.05 de −0.11 y uno de 0.95 de +0.07, la banda del 90% a 12 meses es exactamente eso, sin
suponer nada sobre la forma del modelo.

Tres decisiones que vale explicar:

1. **En log-espacio.** El IPP es un índice positivo con varianza aproximadamente multiplicativa;
   los cuantiles del error log son estables donde los del error absoluto no lo son.
2. **Cuantiles asimétricos.** No se centra la banda: `q_lo` y `q_hi` se toman por separado, así
   que un sesgo sistemático se absorbe como desplazamiento del intervalo. Esto vuelve innecesario
   `sesgo_por_horizonte` para el IPP, y es lo estadísticamente correcto: un sesgo conocido no se
   resta a mano, se incorpora a la distribución predictiva.
3. **Split 70/30.** Se calibra con el 70% inicial de orígenes y la cobertura se MIDE en el 30%
   final. Sin ese corte, la cobertura reportada sería autovalidación: por construcción, los
   cuantiles empíricos cubren el 90% de los datos con que se calcularon.

HONESTIDAD TERMINOLÓGICA
------------------------
Esto es "calibración empírica de cuantiles por horizonte", **no** conformal prediction. El
conformal split exige intercambiabilidad de las observaciones, y una serie de tiempo con
regímenes (el IPP tuvo 2021 al 18.6% anual y 2025 al −1.9%) la viola. No hay garantía de
cobertura finita: hay una estimación honesta basada en el pasado reciente.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

Z90 = 1.6448536269514722


@dataclass
class CalibradorIntervalos:
    """Cuantiles empíricos del error log por horizonte, con extrapolación por ley de potencia."""

    nivel: float = 0.90
    min_origenes: int = 20
    ley_potencia: bool = True

    q_lo: dict[int, float] = field(default_factory=dict)
    q_hi: dict[int, float] = field(default_factory=dict)
    cobertura_medida: dict[int, float] = field(default_factory=dict)
    factor_vs_nominal: dict[int, float] = field(default_factory=dict)
    n_origenes: int = 0
    modelo_calibracion: str = ""
    # half(h) = a * h^b, ajustada sobre los horizontes calibrados.
    _a: float = field(default=float("nan"), repr=False)
    _b: float = field(default=float("nan"), repr=False)

    # ------------------------------------------------------------------ construcción

    @classmethod
    def desde_errores(cls, errores: pd.DataFrame, clave_modelo: str, *,
                      nivel: float = 0.90, fraccion_validacion: float = 0.3,
                      min_origenes: int = 20) -> "CalibradorIntervalos":
        """Calibra con los errores del rolling-origin de `clave_modelo`.

        Devuelve un calibrador vacío (que degrada al CI nominal) si no hay datos suficientes.
        """
        cal = cls(nivel=nivel, min_origenes=min_origenes, modelo_calibracion=clave_modelo)
        req = {"modelo", "horizonte", "y_real", "y_pred", "fecha_corte"}
        if errores is None or errores.empty or not req <= set(errores.columns):
            logger.info("Calibrador: sin errores utilizables para %s", clave_modelo)
            return cal

        e = errores[errores["modelo"] == clave_modelo].copy()
        e = e.dropna(subset=["y_real", "y_pred"])
        e = e[(e["y_real"] > 0) & (e["y_pred"] > 0)]
        if e.empty:
            logger.info("Calibrador: %s no tiene predicciones validas", clave_modelo)
            return cal

        e["e_log"] = np.log(e["y_real"].to_numpy()) - np.log(e["y_pred"].to_numpy())

        # Split temporal por origen: los primeros 70% calibran, el 30% final mide.
        origenes = sorted(e["fecha_corte"].unique())
        cal.n_origenes = len(origenes)
        corte = origenes[int(len(origenes) * (1 - fraccion_validacion))] if len(origenes) >= 10 \
            else None
        cal_df = e[e["fecha_corte"] < corte] if corte is not None else e
        val_df = e[e["fecha_corte"] >= corte] if corte is not None else e.iloc[0:0]
        if cal_df.empty:
            cal_df, val_df = e, e.iloc[0:0]

        alpha = 1 - nivel
        for h, g in cal_df.groupby("horizonte"):
            n = len(g)
            if n < 5:
                continue
            # Corrección de muestra finita. Con n observaciones, el cuantil empírico al 5%
            # subestima la cola: la correccion estandar de prediccion conforme usa el
            # ⌈(n+1)(1-α)⌉-esimo estadistico de orden, lo que equivale a pedir un cuantil algo
            # mas extremo. Con n≈39 (el 70% de 56 origenes) la diferencia es material, y sin
            # ella las bandas salen sistematicamente angostas.
            q_sup = min(1.0, np.ceil((n + 1) * (1 - alpha / 2)) / n)
            q_inf = max(0.0, 1.0 - q_sup)
            cal.q_lo[int(h)] = float(np.quantile(g["e_log"], q_inf))
            cal.q_hi[int(h)] = float(np.quantile(g["e_log"], q_sup))

        if not cal.q_lo:
            logger.info("Calibrador: %s no produjo cuantiles", clave_modelo)
            return cal

        cal._ajustar_ley_potencia()
        cal._medir_cobertura(val_df if not val_df.empty else cal_df,
                             held_out=not val_df.empty)
        cal._medir_factor_nominal(e)
        logger.info("Calibrador %s: %d horizontes, %d origenes, cobertura held-out %s",
                    clave_modelo, len(cal.q_lo), cal.n_origenes,
                    {h: round(v, 1) for h, v in sorted(cal.cobertura_medida.items())})
        return cal

    def _ajustar_ley_potencia(self) -> None:
        """half(h) = a·h^b por OLS en logs. Con b≈0.5 sería un paseo aleatorio puro."""
        hs = sorted(self.q_lo)
        halfs = [(self.q_hi[h] - self.q_lo[h]) / 2 for h in hs]
        validos = [(h, m) for h, m in zip(hs, halfs) if h > 0 and m > 0]
        if len(validos) < 2:
            return
        x = np.log([h for h, _ in validos])
        y = np.log([m for _, m in validos])
        b, log_a = np.polyfit(x, y, 1)
        self._a, self._b = float(np.exp(log_a)), float(b)

    def _medir_cobertura(self, df: pd.DataFrame, *, held_out: bool) -> None:
        for h, g in df.groupby("horizonte"):
            lo, hi = self._cuantiles(int(h))
            if not np.isfinite(lo) or not np.isfinite(hi):
                continue
            dentro = ((g["e_log"] >= lo) & (g["e_log"] <= hi)).mean()
            self.cobertura_medida[int(h)] = float(dentro * 100)
        if not held_out:
            logger.warning("Calibrador: cobertura medida SIN held-out (pocos origenes); "
                           "es autovalidacion y estara inflada")

    def _medir_factor_nominal(self, e: pd.DataFrame) -> None:
        """Cuánto más ancha debería ser la banda nominal, por horizonte."""
        if "ci_lo90" not in e.columns or "ci_hi90" not in e.columns:
            return
        for h, g in e.groupby("horizonte"):
            g = g.dropna(subset=["ci_lo90", "ci_hi90"])
            g = g[(g["ci_lo90"] > 0) & (g["ci_hi90"] > 0)]
            if len(g) < 5:
                continue
            half_nom = float(np.mean(
                (np.log(g["ci_hi90"].to_numpy()) - np.log(g["ci_lo90"].to_numpy())) / 2))
            lo, hi = self._cuantiles(int(h))
            if half_nom > 0 and np.isfinite(lo) and np.isfinite(hi):
                self.factor_vs_nominal[int(h)] = float(((hi - lo) / 2) / half_nom)

    # ------------------------------------------------------------------ uso

    @property
    def disponible(self) -> bool:
        return bool(self.q_lo) and self.n_origenes >= self.min_origenes

    def _cuantiles(self, h: int) -> tuple[float, float]:
        """Cuantiles para el horizonte h, interpolando o extrapolando por la ley de potencia."""
        if h in self.q_lo:
            return self.q_lo[h], self.q_hi[h]
        if not self.q_lo:
            return float("nan"), float("nan")

        hs = sorted(self.q_lo)
        if h < hs[0]:
            ancla = hs[0]
        elif h > hs[-1]:
            ancla = hs[-1]
        else:
            lo_h = max(x for x in hs if x <= h)
            hi_h = min(x for x in hs if x >= h)
            if lo_h == hi_h:
                return self.q_lo[lo_h], self.q_hi[lo_h]
            t = (h - lo_h) / (hi_h - lo_h)
            return ((1 - t) * self.q_lo[lo_h] + t * self.q_lo[hi_h],
                    (1 - t) * self.q_hi[lo_h] + t * self.q_hi[hi_h])

        # Fuera del rango calibrado: escalar el ancla con la ley de potencia y conservar la
        # asimetría. Extrapolar cuantiles crudos daría bandas que no crecen con el horizonte.
        lo, hi = self.q_lo[ancla], self.q_hi[ancla]
        if not (self.ley_potencia and np.isfinite(self._a) and np.isfinite(self._b)):
            return lo, hi
        half_ancla = (hi - lo) / 2
        if half_ancla <= 0:
            return lo, hi
        escala = (self._a * h ** self._b) / half_ancla if half_ancla else 1.0
        centro = (hi + lo) / 2
        return centro - half_ancla * escala, centro + half_ancla * escala

    def aplicar(self, pred: np.ndarray, horizontes: np.ndarray,
                ci_nominal: tuple[np.ndarray, np.ndarray] | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
        """Banda calibrada alrededor de `pred`. Cae al CI nominal escalado si no hay calibración."""
        pred = np.asarray(pred, dtype=float)
        horizontes = np.asarray(horizontes, dtype=int)
        log_pred = np.log(np.maximum(pred, 1e-9))

        if not self.disponible:
            return self._fallback(pred, horizontes, log_pred, ci_nominal)

        lo = np.empty_like(pred)
        hi = np.empty_like(pred)
        for i, h in enumerate(horizontes):
            q_lo, q_hi = self._cuantiles(int(h))
            if not np.isfinite(q_lo) or not np.isfinite(q_hi):
                q_lo, q_hi = -Z90 * 0.02 * np.sqrt(h), Z90 * 0.02 * np.sqrt(h)
            lo[i] = np.exp(log_pred[i] + q_lo)
            hi[i] = np.exp(log_pred[i] + q_hi)
        # Monotonía del ancho: un intervalo a 12 meses no puede ser más angosto que a 6.
        ancho = np.maximum.accumulate(hi - lo)
        centro = (hi + lo) / 2
        return centro - ancho / 2, centro + ancho / 2

    def _fallback(self, pred, horizontes, log_pred, ci_nominal):
        """Sin calibración suficiente: ensanchar el CI nominal por el factor medido."""
        if ci_nominal is None:
            half = Z90 * 0.02 * np.sqrt(horizontes)
            return np.exp(log_pred - half), np.exp(log_pred + half)
        lo_n, hi_n = (np.asarray(x, dtype=float) for x in ci_nominal)
        half = (np.log(np.maximum(hi_n, 1e-9)) - np.log(np.maximum(lo_n, 1e-9))) / 2
        factor = np.array([self.factor_vs_nominal.get(int(h), 1.0) for h in horizontes])
        half = half * np.maximum(factor, 1.0)
        return np.exp(log_pred - half), np.exp(log_pred + half)

    def resumen(self) -> pd.DataFrame:
        filas = []
        for h in sorted(set(self.q_lo) | set(self.cobertura_medida)):
            lo, hi = self._cuantiles(h)
            filas.append({
                "horizonte": h,
                "q_lo_log": round(lo, 5),
                "q_hi_log": round(hi, 5),
                "ancho_log": round(hi - lo, 5),
                "asimetria_log": round((hi + lo) / 2, 5),
                "cobertura_held_out_pct": round(self.cobertura_medida.get(h, float("nan")), 1),
                "factor_vs_nominal": round(self.factor_vs_nominal.get(h, float("nan")), 2),
            })
        df = pd.DataFrame(filas)
        if not df.empty:
            df.attrs["n_origenes"] = self.n_origenes
            df.attrs["modelo"] = self.modelo_calibracion
            df.attrs["ley_potencia"] = (round(self._a, 5), round(self._b, 3))
        return df
