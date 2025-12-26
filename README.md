
docker-compose up -d

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")


# After running training one or more times
python src/jobs/train.py --data-path data/credit_risk_dataset.csv

# Register the best model if F1 >= 0.85
python src/jobs/register.py

# Or with custom threshold
python src/jobs/register.py --min-f1-score 0.90

# Or specify different experiment/model
python src/jobs/register.py \
  --mlflow-experiment-name my-experiment \
  --model-name my-model \
  --min-f1-score 0.88