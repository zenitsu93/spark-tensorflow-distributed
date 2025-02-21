# Spark TensorFlow Project

## Description
This project sets up a distributed environment for training TensorFlow models using Apache Spark and Horovod. It includes:
- A Dockerfile to create an image with Spark, TensorFlow, and Horovod.
- A `docker-compose.yml` file to deploy a Spark cluster with one master and two workers.
- A Python script utilizing TensorFlowOnSpark and Horovod to train a CNN model on the MNIST dataset in a distributed manner.

## Prerequisites
- Docker
- Docker Compose

## Installation
1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Build the Docker image:
   ```bash
   docker build -t spark:3.4.0 .
   ```
3. Start the Spark cluster:
   ```bash
   docker-compose up -d
   ```

## Usage
1. Access the Spark Master interface:
   - Open `http://localhost:8080` in a browser.
2. Run the training script:
   ```bash
   docker exec -it spark-worker-1 python /path/to/train_script.py
   ```

## Results
The script trains a CNN model on the MNIST dataset using Spark and Horovod. The results include:
- Loss and accuracy curves.
- Confusion matrix.
- ROC curves.
- A `results.txt` file containing the model performance.

## References
- [Apache Spark](https://spark.apache.org/)
- [Horovod](https://horovod.readthedocs.io/en/stable/)
- [TensorFlowOnSpark](https://github.com/yahoo/TensorFlowOnSpark)

