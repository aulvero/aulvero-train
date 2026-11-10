# aulvero-train

Pre-training and fine-tuning code for the **Aulvero** compact language models: bilingual
French/English models from 50M to 900M parameters.

**Documentation: [aulvero.github.io/docs](https://aulvero.github.io/docs/)**
([Architecture](https://aulvero.github.io/docs/architecture/) ·
[Data](https://aulvero.github.io/docs/data/) ·
[Training](https://aulvero.github.io/docs/training/))

| Model | Params | Tier | Layers × width | Heads (Q/KV) | Context | Config |
|---|---:|---|---|---|---:|---|
| Wren | 50M | Tiny | 20 × 384 | 6 / 2 | 2,048 | [`wren-50m`](configs/wren-50m.yaml) |
| Finch | 85M | Tiny | 21 × 512 | 8 / 2 | 2,048 | [`finch-85m`](configs/finch-85m.yaml) |
| Robin | 125M | Tiny | 27 × 576 | 9 / 3 | 2,048 | [`robin-125m`](configs/robin-125m.yaml) |
| Lark | 160M | Small | 29 × 640 | 10 / 2 | 4,096 | [`lark-160m`](configs/lark-160m.yaml) |
| Kestrel | 250M | Small | 26 × 896 | 14 / 2 | 4,096 | [`kestrel-250m`](configs/kestrel-250m.yaml) |
| Osprey | 350M | Small | 31 × 960 | 15 / 5 | 4,096 | [`osprey-350m`](configs/osprey-350m.yaml) |
| Heron | 500M | Medium | 31 × 1152 | 18 / 6 | 4,096 | [`heron-500m`](configs/heron-500m.yaml) |
| Crane | 700M | Medium | 33 × 1344 | 21 / 7 | 4,096 | [`crane-700m`](configs/crane-700m.yaml) |
| Condor | 900M | Medium | 33 × 1536 | 24 / 8 | 4,096 | [`condor-900m`](configs/condor-900m.yaml) |

Every model has a **Base** checkpoint (pre-training) and an **Instruct** checkpoint
(fine-tuning). Released weights are published as standard Hugging Face models, so you
don't need this repo to use them.

## Quick start

```bash
make install                 # .venv + dependencies (Python 3.11+)
make test                    # unit tests, CPU only, a few seconds

make prepare MIX=pilot       # small data mix + shared tokenizer
make pilot                   # 2,000-step end-to-end run on the pilot mix

make prepare                 # main French/English/code mix
make pretrain MODEL=kestrel-250m GPUS=8
make finetune MODEL=kestrel-250m GPUS=8
make export   MODEL=kestrel-250m   # -> exports/kestrel-250m-instruct (safetensors)
```

Run `make help` for every target.

## Layout

```
aulvero/          shared package: model, configs, data, chat format, LR schedules
pretrain/         pre-training entry point            -> Base models
finetune/         supervised fine-tuning entry point  -> Instruct models
configs/          one file per model + defaults.yaml shared by all of them
data/mixes/       pre-training data recipes (pretrain, code, pilot)
data/sft/         fine-tuning data (example.jsonl shows the format)
scripts/          data download, tokenizer, packing, launchers, export, sampling
tests/            unit + export tests
```

Generated files (`data/raw`, `data/packed`, `artifacts`, `checkpoints`, `exports`) are
git-ignored. Weights belong on Hugging Face, not in git.

## Pipeline

1. **Data:** `scripts/prepare.sh <mix>` streams a mix from Hugging Face, cleans and
   de-duplicates it, trains the shared 49,152-token tokenizer once, then packs the token files.
2. **Pre-training:** `pretrain/train.py` trains with a warmup-stable-decay schedule and
   writes `checkpoints/<model>/pretrain/`.
3. **Fine-tuning:** `finetune/sft.py` trains on chat data (ChatML format, loss on
   assistant turns only) and writes `checkpoints/<model>/sft/`.
4. **Export:** `scripts/export_hf.py` writes a `Qwen3ForCausalLM`-compatible folder that
   works with `transformers`, vLLM, llama.cpp and Ollama.

See the [training guide](https://aulvero.github.io/docs/training/) for details and
[Architecture](https://aulvero.github.io/docs/architecture/) for the model design.

## Data and licences

Mixes stream public datasets (FineWeb-2 French, FineWeb-Edu, permissively licensed
GitHub code). Review each dataset card and licence before any commercial use.
