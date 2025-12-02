# SMS Checker / Backend

The backend of this project provides a simple REST service that can be used to detect spam messages.
We have extended the base project [rohan8594/SMS-Spam-Detection](https://github.com/rohan8594/SMS-Spam-Detection), which introduces several basic classification models, and wrap one of them in a microservice.

The following sections will explain you how to get started.
The project **requires a Python 3.12 environment** to run (tested with 3.12.9).
Use the `requirements.txt` file to restore the required dependencies in your environment.


### Training the Model

To train the model, you have two options.
Either you create a local environment...

    $ python -m venv venv
    $ source venv/bin/activate
    $ pip install -r requirements.txt

... or you train in a Docker container (recommended):

    $ docker run -it --rm -v ./:/root/sms/ python:3.12.9-slim bash
    ... (container startup)
    $ cd /root/sms/
    $ pip install -r requirements.txt

Once all dependencies have been installed, the data can be preprocessed and the model trained by creating the output folder and invoking three commands:

    $ mkdir output
    $ python src/read_data.py
    Total number of messages:5574
    ...
    $ python src/text_preprocessing.py
    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Unzipping corpora/stopwords.zip.
    ...
    $ python src/text_classification.py

The resulting model files will be placed as `.joblib` files in the `output/` folder.

### Running the Server

The easiest way to run the server is by using the docker file.
First build the Dockerfile into an image that can be used locally using:
```bash
docker build -t model-service .
```

Next you can run your newly created image using:
```bash
docker run -it model-service
```

Common flags you may want to add:

* **`-v ./output:/app/model_files`**
  Mounts the trained model into the container.

* **`-p 8081:8081`**
  Exposes the service to your machine (adjust as needed).

* **`-d`**
  Runs the container in the background.

* **`--rm`**
  Removes the container after it exits.

* **`-e MODEL_URL=<URL>`**
  Sets the model download URL (e.g. https://github.com/doda2025-team14/model-service/releases/download/v1.1.0-09c6efe/model-release.tar.gz).

* **`-e APP_PORT=<PORT>`**
  Changes the server port (default: `8081`). Make sure `-p` matches.


Lastly, to verify that the model is actually running, you can visit [http://localhost:8081/apidocs/#/default](http://localhost:8081/apidocs/#/default) to interact with the API.
