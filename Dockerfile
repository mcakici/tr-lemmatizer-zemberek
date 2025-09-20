FROM eclipse-temurin:17-jre-jammy AS jre

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    JAVA_HOME=/opt/java/openjdk \
    PATH="/opt/java/openjdk/bin:${PATH}"

WORKDIR /srv

COPY --from=jre /opt/java/openjdk /opt/java/openjdk
    
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /srv/app.py
COPY zemberek-full.jar /srv/zemberek-full.jar

EXPOSE 8077

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8077", "--no-server-header"]
    