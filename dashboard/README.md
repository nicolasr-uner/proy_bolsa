# Dashboard de Proyecciones — Guía de Operación

Dashboard interactivo de pronósticos de **Precio de Bolsa** e **IPP** para el mercado
eléctrico colombiano.

---

## Correr localmente

```powershell
# Desde la raíz del proyecto (con Python del sistema o un venv con streamlit+plotly+openpyxl)
$pyexe = "C:\Users\Lenovo\AppData\Local\Python\bin\python.exe"
& $pyexe -m streamlit run dashboard/app.py
# Abrir: http://localhost:8501
```

**Requisitos para el dashboard:** `streamlit>=1.55`, `plotly>=6.6`, `openpyxl>=3.1`,
`pandas>=2.2`, `statsmodels>=0.14`, `lightgbm>=4.6`.

**Requisitos para el ciclo mensual** (modelos): venv de uv (`.venv/`) con `pydataxm`.

---

## Datos necesarios

El dashboard carga los modelos en vivo desde los parquets de features:

| Archivo | Origen | Obligatorio |
|---------|--------|-------------|
| `data/processed/bolsa_features_diario.parquet` | `run_monthly_update.py` | Sí (tab Escenarios Bolsa) |
| `data/processed/ipp_features_mensual.parquet` | `construir_features.py --solo-ipp` | Sí (tab IPP) |
| `data/processed/perfil_horario.parquet` | `run_monthly_update.py` | Para perfil horario |
| `outputs/runs/{YYYY-MM}/` | `run_monthly_update.py` | Para pronósticos base |

Los primeros dos parquets se construyen automáticamente con el ciclo mensual.
Sin ellos, el dashboard carga igual pero los tabs de escenarios muestran un aviso.

---

## Sincronizar datos tras el ciclo mensual

Después de ejecutar `run_monthly_update.py` en la máquina local:

```powershell
# Si el servidor Docker tiene montado un directorio de datos compartido:
xcopy /E /Y data\processed\ \\servidor\proy_bolsa\data\processed\
xcopy /E /Y outputs\runs\    \\servidor\proy_bolsa\outputs\runs\

# O copiar solo los archivos necesarios:
copy data\processed\bolsa_features_diario.parquet \\servidor\...
copy data\processed\ipp_features_mensual.parquet  \\servidor\...
```

El dashboard recargará los datos automáticamente en la próxima sesión
(el `@st.cache_resource` para los modelos se invalida al reiniciar el contenedor).

---

## Despliegue con Docker

### Build
```bash
docker build -t proy-bolsa-dashboard .
```

### Run (con volumen para datos persistentes)
```bash
docker run -d \
  --name proy-bolsa \
  -p 8501:8501 \
  -v /ruta/en/host/data:/app/data:ro \
  -v /ruta/en/host/outputs:/app/outputs:ro \
  proy-bolsa-dashboard
```

Abrir `http://IP-DEL-SERVIDOR:8501`.

### Variables de entorno opcionales
```bash
-e STREAMLIT_SERVER_MAX_UPLOAD_SIZE=50   # MB
```

---

## Autenticación (recomendado para acceso compartido)

Para proteger el dashboard con usuario/contraseña, instalar `streamlit-authenticator`:

```bash
pip install streamlit-authenticator==0.4.2
```

Y agregar al inicio de `dashboard/app.py`:

```python
import streamlit_authenticator as stauth
import yaml

with open("dashboard/auth.yaml") as f:
    config = yaml.safe_load(f)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

name, authentication_status, username = authenticator.login("Login", "main")
if not authentication_status:
    st.stop()
```

Y crear `dashboard/auth.yaml` con credenciales hasheadas:

```bash
python -c "
import streamlit_authenticator as stauth
print(stauth.Hasher(['password_aqui']).generate())
"
```

```yaml
# dashboard/auth.yaml  (NO versionar en git si tiene contraseñas reales)
credentials:
  usernames:
    nicolas:
      email: nicolas@unergy.io
      name: Nicolas Roveda
      password: $2b$12$...  # hash bcrypt
    equipo:
      email: equipo@unergy.io
      name: Equipo Unergy
      password: $2b$12$...
cookie:
  expiry_days: 7
  key: proy_bolsa_secret_key_2026
  name: proy_bolsa_auth
```

> `dashboard/auth.yaml` debe estar en `.gitignore` para no exponer contraseñas.

---

## Estructura de tabs

| Tab | Contenido |
|-----|-----------|
| **Precio Bolsa (corto)** | Pronóstico 7 días del ciclo mensual con bandas IC 90% |
| **Perfil Horario** | Heatmap tipo_día × hora del multiplicador de perfil |
| **Escenarios Bolsa** | Sliders aportes/volumen/ONI/escasez → pronóstico en vivo; comparar/exportar |
| **IPP y Escenarios** | Histórico IPP + sliders TRM%/Brent% → escenarios interactivos |
| **Seguimiento de Precisión** | MAE/RMSE/sesgo por horizonte; evolución de pesos del ensemble |

---

## Actualización mensual (checklist)

1. Ejecutar `run_monthly_update.py --sin-descarga` (o con descarga si hay conexión a XM)
2. Copiar parquets al servidor de despliegue (ver sección Sincronizar)
3. Reiniciar el contenedor Docker: `docker restart proy-bolsa`
4. Verificar en `http://IP-SERVIDOR:8501` que el ciclo nuevo aparezca en el encabezado
5. Revisar tab "Seguimiento de Precisión" para confirmar errores del ciclo anterior
