
# Use an official Python runtime as the base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY ./requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app folder into the container at /app
COPY ./app .

# Set environment variable to tell Flask to run in production mode
ENV FLASK_ENV=production
ENV OPENAI_API_KEY=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

# Make port 5000 available outside the container
EXPOSE 5000

# Define the command to run the app using gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
