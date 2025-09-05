FROM python:3.12

RUN mkdir app/

COPY app/app.py

RUN mkdir models/

COPY models/RF3(0.79 acu)



COPY install requirements.txt

RUN pip install -r requirements.txt

CMD []

EXPOSE 8000




'''FROM python:3.12

# Create working directories
RUN mkdir /app
RUN mkdir /models

# Set working directory
WORKDIR /app

# Copy application code
COPY app/app.py /app/

# Copy model files
COPY models/RF3\(0.79\ acu\) /models/

# Copy and install dependencies
COPY install/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt'''