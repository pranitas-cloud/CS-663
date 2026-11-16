PYTHON ?= python

.PHONY: install test validate-data

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest -q

validate-data:
	$(PYTHON) scripts/validate_dataset.py --data-root data/mvtec_anomaly_detection
