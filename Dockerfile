# Base image for Spark
FROM openjdk:11-jre-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    libopenmpi-dev \
    openmpi-bin \
    libjpeg-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    git \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Define environment variables
ENV SPARK_VERSION=3.4.0
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
ENV HOROVOD_WITH_TENSORFLOW=1
ENV HOROVOD_WITH_MPI=1
ENV PYTHONUNBUFFERED=1

# Download and install Spark
RUN wget -q https://archive.apache.org/dist/spark/spark-$SPARK_VERSION/spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION.tgz && \
    tar -xzf spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION.tgz -C /opt && \
    mv /opt/spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION /opt/spark && \
    rm spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION.tgz

# Install Python packages in the correct order
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    tensorflow==2.12.0 \
    'protobuf<4.0.0' \
    scikit-learn==1.2.2 \
    matplotlib==3.7.1 \
    pandas==2.0.1 \
    seaborn==0.12.2 \
    numpy==1.23.5 \
    && HOROVOD_WITH_TENSORFLOW=1 pip3 install --no-cache-dir horovod[tensorflow]==0.28.1 && \
    pip3 install --no-cache-dir tensorflowonspark==2.2.5

# Expose Spark ports
EXPOSE 4040 7077 8080 18080

# Set the entrypoint for the container
CMD ["/opt/spark/bin/spark-class", "org.apache.spark.deploy.master.Master"]