# Copy Openfold3 folder and build dependencies needed for testing
# FROM openfoldconsortium/openfold3:stable
FROM openfold-docker:latest

COPY openfold3 /opt/openfold3
COPY environments/requirements-test.txt /opt/openfold3/requirements-test.txt

WORKDIR /opt/openfold3
RUN pip install -r requirements-test.txt
RUN pip install --force-reinstall --no-deps .
