# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY ./requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Explicitly install gunicorn
RUN pip install gunicorn

# Copy the entire project into the container
COPY . /app/

# Set environment variable to tell Flask to run in production mode
ENV FLASK_ENV=production

# Make port 5000 available outside the container
EXPOSE 5000

# Define the command to run the app using gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
