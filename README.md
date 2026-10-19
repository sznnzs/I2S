<a name="readme-top"></a>

<div align="center">
  <img src="./docs/assets/i2s-logo.png" alt="I2S" width="150">
  <h1>From Intervention to Skill: Evolving Small–Large Language Model Collaboration</h1>
  <a href="https://sznnzs.github.io/I2S/">
    <img src="https://img.shields.io/badge/Project-Page-6a5acd?style=for-the-badge" alt="Project Page">
  </a>
  <a href="https://github.com/sznnzs/I2S">
    <img src="https://img.shields.io/badge/GitHub-Code-181717?style=for-the-badge&amp;logo=github&amp;logoColor=white" alt="Code">
  </a>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="MIT License">
  </a>
</div>

<a name="overview"></a>

**Intervention-to-Skill (I2S)** turns large-model assistance into procedural skills that a small model can reuse across tasks. The small model proposes an action, a calibrated confidence score guides intervention, and execution feedback supports skill acquisition and refinement. Skills carry over between tasks while the acting models remain fixed.

### 🪐 Key Features

- **Selective collaboration.** Route small-model proposals using calibrated confidence and execution feedback.
- **Reusable procedural skills.** Turn effective interventions into skills with applicability conditions, ordered operations, and outcome checks.
- **Streamwise adaptation.** Carry skills across tasks and refine them with new evidence while both acting models remain fixed.

The implementation supports **ALFWorld**, **WebShop**, and **ScienceWorld**, with local or cloud large-model backends.

<div align="center">
  <a href="./docs/assets/overview.pdf"><img src="./docs/assets/overview.png" alt="I2S: selective intervention, trajectory-aware skill induction, and streamwise skill evolution." width="100%"></a>
  <p><em>From large-model interventions to reusable skills for future tasks.</em></p>
</div>

## 🔥 News

- **[2026.10]** The I2S implementation and pretrained confidence adapter are available in this repository.

## 📑 Table of Contents

- [Quick Start](#quick-start)
- [Running](#running)
- [Confidence Calibration](#confidence-calibration)
- [Evaluation](#evaluation)
- [How It Works](#how-it-works)
- [Acknowledgements](#acknowledgements)

<a name="getting-started"></a>

## 🚀 Quick Start <span id="quick-start"></span>

### 1. Installation

Use Linux, Python 3.10, and Java 11. The Conda environment includes the benchmark dependencies. Local model serving and calibration training use CUDA GPUs.

```bash
git clone https://github.com/sznnzs/I2S.git
cd I2S
conda env create -f environment.yaml
conda activate i2s
python -m pip install --upgrade pip
python -m pip install -e '.[training]'
```

Run the following commands from the repository root.

### 2. Data

```bash
python scripts/setup_environments.py all --download --build-index
```

Assets are stored in `data/`, and the pinned WebShop source is installed in `third_party/`. The setup command builds WebShop's full-catalog Lucene index. To use existing assets, configure their paths in [configs/environments.yaml](configs/environments.yaml).

Benchmark environments run in CPU subprocesses. Set `I2S_ENV_PYTHON` to use a separate environment interpreter, or `JAVA_HOME` to select a Java installation.

### 3. Model Services

The small model is **Qwen3-8B**. Choose one large-model backend:

| Backend | Large model | Service |
| --- | --- | --- |
| `local` | Qwen3-32B | vLLM |
| `deepseek` | DeepSeek-V4-Flash | OpenAI-compatible API |
| `gemini` | Gemini-2.5-Flash | OpenAI-compatible API |

The pretrained confidence adapter is included in [models/reliability](models/reliability). The small-model service hosts both the actor and the calibrated scorer.

Create a serving environment:

```bash
conda create -n i2s-serving python=3.10 -y
conda activate i2s-serving
python -m pip install --upgrade pip
python -m pip install -e '.[serving]'
```

Start the small-model service:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/serve_models.py small
```

For the local backend, start the teacher in a second terminal with the same serving environment:

```bash
CUDA_VISIBLE_DEVICES=1,2 python scripts/serve_models.py large --tensor-parallel-size 2
```

Adjust GPU assignments and tensor parallelism to fit your hardware. `I2S_SMALL_MODEL` and `I2S_LARGE_MODEL` can point to local weights. Endpoints and generation settings are in [configs/models.yaml](configs/models.yaml).

For a cloud teacher, export its endpoint and API-key variables from [.env.example](.env.example), then select `--backend deepseek` or `--backend gemini`. Load any local `.env` file in your shell before running I2S.

## ⚙️ Running <span id="running"></span>

Return to the `i2s` environment and run an ALFWorld task stream:

```bash
conda activate i2s
i2s check
i2s run --benchmark alfworld --run-id alfworld_run
i2s evaluate runs/alfworld_run
```

`i2s check` validates the configuration and local adapter files. Runs save task selections, prompts, trajectories, model calls, and skill state under `runs/<run-id>`.

Run all task families in a benchmark:

```bash
i2s run --benchmark alfworld --all-families --run-id alfworld
i2s run --benchmark webshop --all-families --run-id webshop
i2s run --benchmark scienceworld --all-families --run-id scienceworld
```

Each family has an independent skill bank. Families run concurrently; tasks within each family run sequentially and receive the skills learned from their predecessors.

Select a cloud teacher:

```bash
i2s run --benchmark webshop --backend deepseek --all-families --run-id webshop_deepseek
```

Resume an interrupted run with its original settings and run ID:

```bash
i2s run --benchmark scienceworld --all-families --run-id scienceworld --resume
```

Completed tasks are kept, and the interrupted task restarts from its beginning. Model-call records retain usage from earlier attempts.

### Configuration

Agent settings are in [configs/default.yaml](configs/default.yaml), model settings in [configs/models.yaml](configs/models.yaml), and benchmark settings in [configs/environments.yaml](configs/environments.yaml). Inspect the assembled configuration with:

```bash
i2s resolve --benchmark alfworld --backend local
```

WebShop and ScienceWorld use `split: evaluation`. The ScienceWorld conductivity family is `conductivity`.

| Benchmark | Actions per task | Large-model actions per task | Actions per intervention |
| --- | ---: | ---: | ---: |
| ALFWorld | 50 | 16 | 4 |
| WebShop | 15 | 8 | 3 |
| ScienceWorld | 100 | 30 | 4 |

Small- and large-model actions share the task horizon. The skill bank holds up to six cards and 2,800 tokens, with at most 900 tokens and eight operations per card. The routing threshold is **0.99**.

For the local backend, actor requests allow 32,768 input tokens and 8,192 output tokens; induction requests allow 24,576 input tokens and 16,384 output tokens. Both use thinking, temperature 0.6, top-p 0.95, top-k 20, and presence penalty 1.5. Confidence scoring allows 16,000 input tokens across all backends.

## 🎯 Confidence Calibration <span id="confidence-calibration"></span>

The included adapter works across all three benchmarks. To train another adapter, collect decision contexts from ALFWorld's training split with the model services running:

```bash
i2s run --benchmark alfworld --scenario pick_and_place_simple --tasks 100 \
  --set router.reliability_model_id=qwen3_8b \
  --set policy.learning.skills_enabled=false \
  --set model_pool.models.qwen3_8b.generation.max_input_tokens=8192 \
  --set benchmark.split=train --set benchmark.partition_role=train \
  --run-id reliability_collection
```

Prepare the examples and train the adapter. Stop the model services first if you need their GPU memory:

```bash
python scripts/prepare_reliability_data.py \
  --runs runs/reliability_collection --output data/reliability
CUDA_VISIBLE_DEVICES=0 python scripts/train_reliability.py
```

Training uses rank-8 query/value adapters on Qwen3-8B, three epochs, and inverse-frequency class weights. Labels come from local execution feedback, and training and validation use disjoint task identities. The final-epoch adapter is saved to `runs/reliability_training`.

Evaluate the adapter:

```bash
python scripts/train_reliability.py --evaluate-only \
  --adapter runs/reliability_training \
  --validation data/reliability/validation.jsonl \
  --output runs/reliability_evaluation
```

Training and inference normalize the likelihoods of the complete `UNCERTAIN` and `CONFIDENT` label sequences. Evaluation uses the deployment threshold of 0.99 by default.

Set `I2S_ADAPTER` in both the model-serving and I2S terminals, then restart the small-model service:

```bash
export I2S_ADAPTER=runs/reliability_training
CUDA_VISIBLE_DEVICES=0 python scripts/serve_models.py small
```

## 📊 Evaluation <span id="evaluation"></span>

Recompute metrics from a saved run:

```bash
i2s evaluate runs/alfworld_run
```

Evaluation reports native task outcomes, interventions, and model usage, and writes `metrics.json` and `validation.json`. Native outcomes are ALFWorld completion, WebShop reward, and ScienceWorld score. ScienceWorld includes both raw and nonnegative reporting scores.

The weighted output-token metric is:

```text
weighted output tokens = large-model output tokens + 0.25 × small-model output tokens
```

Provider-reported reasoning and retry tokens are included. Input usage is recorded separately, and unavailable usage is represented as unknown.

## ✨ How It Works <span id="how-it-works"></span>

1. **Selective intervention.** The small model proposes an action. A calibrated confidence score below 0.99 or an execution failure triggers large-model assistance.
2. **Skill induction.** After a task, proposal–intervention contrasts, observed outcomes, and trajectory context identify reusable procedures with applicability conditions, ordered steps, and outcome checks.
3. **Skill evolution.** Add, retain, or revise skills using the new evidence. The updated skill bank guides the small model on later tasks.

## 🌱 Acknowledgements <span id="acknowledgements"></span>

We thank the teams behind [Qwen3](https://github.com/QwenLM/Qwen3), [ALFWorld](https://github.com/alfworld/alfworld), [WebShop](https://github.com/princeton-nlp/WebShop), [ScienceWorld](https://github.com/allenai/ScienceWorld), [vLLM](https://github.com/vllm-project/vllm), and [Transformers](https://github.com/huggingface/transformers).

This project is licensed under the [MIT License](LICENSE).

<div align="center">
  <a href="https://github.com/sznnzs/I2S">
    <img src="https://img.shields.io/badge/Star%20us%20on%20GitHub-181717?style=for-the-badge&amp;logo=github&amp;logoColor=white" alt="Star I2S on GitHub">
  </a>
  <a href="https://github.com/sznnzs/I2S/issues">
    <img src="https://img.shields.io/badge/Report%20Issues-e74c3c?style=for-the-badge&amp;logo=github" alt="Report issues">
  </a>
</div>
