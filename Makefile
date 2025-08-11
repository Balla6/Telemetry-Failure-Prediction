.PHONY: setup data.generate data.validate features.build model.baseline model.train eval.report drift.simulate drift.check dash docs clean all

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

data.generate:
	python -m src.ingest.generate --conf conf/config.yaml

data.validate:
	python -m src.ingest.validate --conf conf/config.yaml

features.build:
	python -m src.features.build --conf conf/config.yaml

model.baseline:
	python -m src.models.baseline --conf conf/config.yaml

model.train:
	python -m src.models.train --conf conf/config.yaml

eval.report:
	python -m src.eval.report --conf conf/config.yaml

drift.simulate:
	# optional: create the planned drifts/glitches
	python -m src.ingest.generate --conf conf/config.yaml --inject_drift true --glitches true

drift.check:
	python -m src.drift.checks --conf conf/config.yaml

dash:
	streamlit run src/dash/app.py --server.headless true

docs:
	@echo "Build figures/reports if needed"

clean:
	rm -rf data/bronze/* data/silver/* data/gold/* reports/*

all: data.generate data.validate features.build model.baseline model.train eval.report

