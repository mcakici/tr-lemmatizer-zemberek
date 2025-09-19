FROM python:3.11-slim

# Install Java (required for Zemberek)
RUN apt-get update && apt-get install -y \
    openjdk-11-jdk \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

WORKDIR /srv

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py /srv/app.py

# Copy Zemberek JAR file (should be in the same directory as Dockerfile)
COPY zemberek-full.jar /srv/zemberek-full.jar

EXPOSE 8077

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8077/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8077", "--no-server-header"]