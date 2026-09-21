FROM python:3.11-slim

WORKDIR /app

# System deps needed by xgboost/shap wheels on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ configs/
COPY src/ src/
COPY api/ api/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DISABLE_PANDERA_IMPORT_WARNING=True

# Train and tune the model as part of the image build rather than
# relying on a volume-mounted or git-committed artifact. .gitignore
# deliberately excludes *.joblib (trained artifacts don't belong in
# version control), so a fresh `git clone && docker build` would
# otherwise produce an image with no model in models/model_registry/.
# Because the dataset is public (bundled with scikit-learn) and the
# random_seed is fixed in configs/config.yaml, this build step is fully
# deterministic and self-contained -- no external registry or mount
# needed for the image to be runnable immediately after building.
RUN python -m src.models.train && python -m src.models.tune

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
