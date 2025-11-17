FROM python:3.12.9-slim AS train
WORKDIR /app
COPY ["requirements.txt", "./"]
RUN ["pip", "install", "--no-cache-dir", "-r", "requirements.txt"]
COPY ["src", "./src"]
COPY ["smsspamcollection", "./smsspamcollection"]
RUN ["mkdir", "output"]
RUN ["python", "src/text_preprocessing.py"]
RUN ["python", "src/text_classification.py"]

FROM python:3.12.9-slim AS run
WORKDIR /app
COPY ["requirements.txt", "./"]
RUN ["pip", "install", "--no-cache-dir", "-r", "requirements.txt"]
COPY ["src", "./src"]
COPY --from=train /app/output /app/output
ENV APP_PORT=8081
EXPOSE $APP_PORT
ENTRYPOINT ["python", "src/serve_model.py"]
