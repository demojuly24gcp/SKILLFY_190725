# Use official Python image
FROM python:3.10-slim

# Set default model path (can be overridden via ENV)
ENV MODEL_PATH="/app/model_output/wine_rf_model.pkl"

# Set working directory
WORKDIR /app

# Copy necessary files
COPY flask_app/wine_quality_api.py ./app.py
COPY model_output/wine_rf_model.pkl ./model_output/wine_rf_model.pkl

# Install Python dependencies directly
RUN pip install --no-cache-dir flask numpy scikit-learn

# Expose port
EXPOSE 5000

# Start the Flask app
CMD ["python", "app.py"]
