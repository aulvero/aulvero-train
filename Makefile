MODEL ?= wren-50m
MIX   ?= pretrain
GPUS  ?= 1
export GPUS

.PHONY: help install prepare pilot pretrain finetune export count test lint clean

help:  ## Show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[1m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Create .venv and install with dev tools
	python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"

prepare:  ## Download + tokenise a data mix (MIX=pretrain|code|pilot)
	bash scripts/prepare.sh $(MIX)

pilot:  ## Short end-to-end run on the pilot mix
	bash scripts/pretrain.sh pilot

pretrain:  ## Pre-train MODEL (e.g. make pretrain MODEL=kestrel-250m GPUS=8)
	bash scripts/pretrain.sh $(MODEL)

finetune:  ## Instruction-tune MODEL from its pre-trained checkpoint
	bash scripts/finetune.sh $(MODEL)

export:  ## Export MODEL's SFT checkpoint to Hugging Face format in exports/
	python scripts/export_hf.py --checkpoint checkpoints/$(MODEL)/sft/final.pt --output exports/$(MODEL)-instruct

count:  ## Parameter count of every config
	python scripts/count_parameters.py

test:  ## Run the test suite
	pytest -q

lint:  ## Lint with ruff
	ruff check .

clean:  ## Remove caches (keeps data, checkpoints and exports)
	rm -rf .pytest_cache .ruff_cache **/__pycache__
