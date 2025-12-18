FROM python:3.12.9-slim AS run
WORKDIR /app
COPY ["requirements.txt", "./"]
RUN ["pip", "install", "--no-cache-dir", "-r", "requirements.txt"]
COPY ["src", "./src"]
COPY ["version.txt", "./"]
ENV APP_PORT=8081
EXPOSE $APP_PORT
ENV MODEL_URL=https://github.com/doda2025-team14/model-service/releases/latest/download/model-release.tar.gz
ENTRYPOINT ["python", "src/serve_model.py"]
