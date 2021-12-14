venv:
ifndef VIRTUAL_ENV
ifndef CONDA_PREFIX
$(error VIRTUAL / CONDA ENV is not set - please activate environment)
endif
endif


deps: venv
	pip install -U pip setuptools~=59.0
	pip install -Ue .

test: venv
	python setup.py test