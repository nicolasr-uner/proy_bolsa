FROM python:3.11-slim

# --- Dependencias del sistema ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Instalar dependencias Python ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Copiar codigo y datos procesados ---
COPY src/ src/
COPY dashboard/ dashboard/
COPY data/processed/ data/processed/
# outputs/runs/ se monta como volumen externo para persistir pronosticos
# (ver docker run -v)

# Instalar el paquete local
COPY pyproject.toml .
RUN pip install --no-cache-dir -e . --no-deps 2>/dev/null || true

# --- Streamlit config ---
COPY .streamlit/ .streamlit/

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["python", "-m", "streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
