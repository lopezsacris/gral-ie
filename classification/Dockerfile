FROM python:3.9-slim

# Kontenedorean lan-helbidea definitu
WORKDIR /app

# Instalatu TensorFlow-rentzat beharrezkoak diren dependentziak
RUN apt-get update && apt-get install -y \
	libgomp1 \
	&& rm -rf /var/lib/apt/lists/*

# Kopiatu kontenedorean instalatu beharreko moduluak dituen testu fitxategia
COPY shared_config.py /app/shared_config.py
COPY requirements.txt /app/requirements.txt

# Instalatu moduluak
RUN pip install --no-cache-dir -r requirements.txt

# Kopiatu programa kontenedorean
COPY classification/classification.py /app/classification.py

# Zehaztu PYTHONPATH lan-helbidera, import-ekin arazoak saihesteko.
ENV PYTHONPATH=/app

CMD ["python", "classification.py"]