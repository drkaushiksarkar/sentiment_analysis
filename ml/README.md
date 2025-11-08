# ML Service Scaffolding

Scripts and Docker assets for running training jobs outside the notebook.

## Local Usage

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r ml/requirements.txt
```

Run helper scripts (single epoch smoke tests):

```bash
bash ml/scripts/train_imdb.sh
bash ml/scripts/train_sarcasm.sh
```

## Docker

```bash
docker build -t sentiment-ml -f ml/Dockerfile .
docker run --rm sentiment-ml
```

The default entrypoint triggers the IMDB training script; override `CMD` for other workflows.
