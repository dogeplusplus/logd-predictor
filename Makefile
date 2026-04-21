REPO_NAME := $(shell basename `git rev-parse --show-toplevel`)
DVC_REMOTE := ${GDRIVE_FOLDER}/${REPO_NAME}

.PHONY:install
install:
	./firstTimeSetup.sh

.PHONY:test
test:
	python -m pytest

.PHONY:install-hooks
install-hooks:
	precommit install

.PHONY:format
format:
	ruff format

.PHONY:lint
lint:
	ruff lint
