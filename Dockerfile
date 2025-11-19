FROM python:3.12.9-slim

# Accept image version
ARG VERSION
ENV APP_VERSION=${VERSION}

WORKDIR /model-service

# Copy dependencies first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NLTK data (stopwords used during preprocessing)
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Copy project source
COPY . .

# Ensure output directory exists
RUN mkdir -p output

# Default mode is serving
ENV MODE=serve

EXPOSE 8081

# Run the correct program based on MODE
# MODE=train → run full pipeline
# MODE=serve → start REST API
CMD ["bash", "-c", " \
  echo Backend version: $APP_VERSION && \
  if [ \"$MODE\" = \"train\" ]; then \
      mkdir -p output && \
      python src/read_data.py && \
      python src/text_preprocessing.py && \
      python src/text_classification.py; \
  else \
      python src/serve_model.py; \
  fi"]
