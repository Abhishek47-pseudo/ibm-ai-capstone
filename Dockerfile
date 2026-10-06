FROM continuumio/miniconda3

LABEL maintainer="Rafael Barbosa"
WORKDIR /app

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Use environment:
SHELL ["conda", "run", "-n", "ibm-ai-capstone", "/bin/bash", "-c"]

# The code to run when container is started:
COPY . /app/

EXPOSE 8050

ENTRYPOINT ["conda", "run", "-n", "ibm-ai-capstone", "python", "/app/app.py"]
 