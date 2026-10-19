FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential cmake flex bison && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/wafer-thermal-cosim
COPY . .
RUN pip install --no-cache-dir .
CMD ["python", "-m", "wafercosim", "run", "--out", "/output/demo"]
