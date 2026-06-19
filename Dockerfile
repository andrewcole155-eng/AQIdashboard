# Start with an official Miniconda image
FROM continuumio/miniconda3

# Install system dependencies (git, cron, procps for pkill, util-linux for flock, nano)
# Combined into a single RUN layer for Docker cache efficiency
RUN apt-get update && apt-get install -y \
    git \
    cron \
    procps \
    nano \
    util-linux \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the environment file
COPY environment.yml .

# Prevent git from asking for credentials in a headless build environment
# This solves the "fatal: could not read Username for 'https://github.com'" error
ENV GIT_TERMINAL_PROMPT=0

# Create the Conda environment inside Docker
RUN conda env create -f environment.yml -n tf_noavx

# Installs neo4j directly into the environment without busting the Conda download cache
RUN /opt/conda/envs/tf_noavx/bin/pip install neo4j

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "tf_noavx", "/bin/bash", "-c"]

# Copy the rest of your application code
COPY . .

# Set up the cron jobs
COPY crontab.docker /etc/cron.d/aqi-cron
RUN chmod 0644 /etc/cron.d/aqi-cron
RUN crontab /etc/cron.d/aqi-cron
RUN touch /var/log/cron.log