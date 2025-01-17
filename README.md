<div align="center">
  <h1>🔬 ArXiv Domain Expert SLM</h1>
  <p>A SLM system trained on ArXiv papers to create a domain expert</p>
  
  
  <p><em>This project is adapted from the excellent <a href="https://github.com/PacktPublishing/LLM-Engineers-Handbook">LLM Engineer's Handbook</a> by <a>Alex Vesa</a>, <a href="https://github.com/iusztinpaul">Paul Iusztin</a>, and <a href="https://github.com/mlabonne">Maxime Labonne</a>. While maintaining the robust MLOps architecture and best practices from the original work, this adaptation implements finetunning from arxiv papers through a complete academic paper processing pipeline made possible with <a href="https://ds4sd.github.io/docling/">docling</a> a library developed by IBM.</em></p>
</div>

## 🔄 Key Adaptations from Original Work

This project maintains the robust MLOps infrastructure of the original LLM Engineer's Handbook while implementing significant changes:

- **Different Purpose**: Instead of creating a digital twin, this project aims to develop an ML expert LLM
- **New Data Source**: Replaced LinkedIn/Medium/Substack pipelines with ArXiv paper pipeline
- **Modified Training Objective**: Adapted training approach to focus on ML domain expertise
- **Enhanced ODM**: Migrated from custom MongoDB ODM to [mongoengine](https://github.com/MongoEngine/mongoengine) for production reliability
- **Custom Features**: Implemented specialized feature engineering for academic paper processing using IBM's [docling](https://ds4sd.github.io/docling/) library for comprehensive PDF parsing

The project leverages the original architecture's excellent MLOps practices while implementing these substantial changes to serve a different use case.

## 🌟 Features

The goal of this project is to create a end-to-end production-ready LLM system that can:

- 📝 Data collection & generation from ArXiv's papers
- 🔄 LLM training pipeline through finetunning from custom instruction and preference datasets
- 📊 RAG system
- 🚀 Production-ready AWS deployment
- 🔍 Comprehensive monitoring
- 🧪 Testing and evaluation framework

You can download and use the final trained model with ML's papers on Hugging Face (_WIP_).

For detailed implementation of MLOps practices and infrastructure setup, please refer to the original [LLM Engineer's Handbook Repository](https://github.com/PacktPublishing/LLM-Engineers-Handbook) and the [LLM Engineer's Handbook Book](https://www.amazon.com/LLM-Engineers-Handbook-engineering-production/dp/1836200072/).

## 🔗 Dependencies

### Local dependencies

| Tool                                                                                     | Version  | Purpose                        | Open Source |
| ---------------------------------------------------------------------------------------- | -------- | ------------------------------ | ----------- |
| [uv](https://docs.astral.sh/uv/)                                                         | latest   | Fast Python package management | Yes         |
| [Python](https://www.python.org/)                                                        | 3.11     | Runtime environment            | Yes         |
| [Docker](https://www.docker.com/)                                                        | ≥27.1.1  | Containerization               | Yes         |
| [AWS CLI](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/index.html) | ≥2.15.42 | Cloud management               | No          |
| [Git](https://git-scm.com/)                                                              | ≥2.44.0  | Version control                | Yes         |

### Cloud Services

The project uses the following services (setup instructions provided in deployment section):

| Service                                               | Purpose                          | Open Source |
| ----------------------------------------------------- | -------------------------------- | ----------- |
| [HuggingFace](https://huggingface.com/)               | Model registry                   | Yes         |
| [Comet ML](https://www.comet.com/docs/v2/)            | Experiment tracker               | No          |
| [Opik](https://www.comet.com/docs/opik/)              | Prompt monitoring                | Yes         |
| [ZenML](https://www.zenml.io/)                        | Orchestrator and artifacts layer | Yes         |
| [AWS](https://aws.amazon.com/)                        | Compute and storage              | No          |
| [MongoDB](https://www.mongodb.com/)                   | NoSQL database                   | Yes         |
| [Qdrant](https://qdrant.tech/)                        | Vector database                  | Yes         |
| [GitHub Actions](https://github.com/features/actions) | CI/CD pipeline                   | Yes         |

## 🗂️ Project Structure

Here is the directory overview:

```bash
.
├── code_snippets/       # Standalone example code
├── configs/             # Pipeline configuration files
├── llm_engineering/     # Core project package
│   ├── application/
│   ├── domain/
│   ├── infrastructure/
│   ├── model/
├── pipelines/           # ML pipeline definitions
├── steps/               # Pipeline components
├── tests/               # Test examples
├── tools/               # Utility scripts
│   ├── run.py
│   ├── ml_service.py
│   ├── rag.py
│   ├── data_warehouse.py
```

`llm_engineering/` is the main Python package implementing LLM and RAG
functionality. It follows Domain-Driven Design (DDD) principles:

- `domain/`: Core business entities and structures
- `application/`: Business logic, crawlers, and RAG implementation
- `model/`: LLM training and inference
- `infrastructure/`: External service integrations (AWS, Qdrant, MongoDB,
  FastAPI)

The code logic and imports flow as follows: `infrastructure` → `model` →
`application` → `domain`

`pipelines/`: Contains the ZenML ML pipelines, which serve as the entry
point for all the ML pipelines. Coordinates the data processing and model
training stages of the ML lifecycle.

`steps/`: Contains individual ZenML steps, which are reusable components for building and customizing ZenML pipelines. Steps perform specific tasks (e.g., data loading, preprocessing) and can be combined within the ML pipelines.

`tests/`: Covers a few sample tests used as examples within the CI
pipeline.

`tools/`: Utility scripts used to call the ZenML pipelines and inference
code:

- `run.py`: Entry point script to run ZenML pipelines.
- `ml_service.py`: Starts the REST API inference server.
- `rag.py`: Demonstrates usage of the RAG retrieval module.
- `data_warehouse.py`: Used to export or import data from the MongoDB data warehouse through JSON files.

`configs/`: ZenML YAML configuration files to control the execution of pipelines and steps.

`code_snippets/`: Independent code examples that can be executed independently.

## 💻 Installation

### 1. Clone and Setup

1. First, clone the repository and navigate to the project directory:

```bash
git clone https://github.com/danivpv/arxiv-domain-expert-llm.git
cd arxiv-domain-expert-llm
```

2. Then, install the dependencies:

```bash
uv sync
```

3. Optionally, if you plan to commit code, you can install the `pre-commit` hooks:

```bash
uv sync --extra dev
uv run pre-commit install
```

4. Optionally, activate the virtual environment. Although `uv run` will automatically discover the virtual environment of the project, this is necessary to get the right interpreter using `which python` as expected by other workflows with `venv`, `pip`, `poetry`, etc...

```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

As our task manager (similar to `Makefile`), we run all the scripts using [Poe the Poet](https://poethepoet.natn.io/index.html) defined in the `tool.poe.tasks` section on the `pyproject.toml` file.

```bash
uv run poe ...
```

### 2. Local Development Setup

After you have installed all the dependencies, you must create and fill a `.env` file with your credentials to appropriately interact with other services and run the project. Remember to add the file to your `.gitignore` file to keep it secret.

```env
OPENAI_API_KEY=your_api_key_here
HUGGINGFACE_ACCESS_TOKEN=your_token_here
COMET_API_KEY=your_api_key_here
```

> Details on how to obtain appropiate credentials can be found in the official [repository](https://github.com/PacktPublishing/LLM-Engineers-Handbook) and [book](https://www.amazon.com/LLM-Engineers-Handbook-engineering-production/dp/1836200072/).

### 3. Deployment Setup

When deploying the project to the cloud, we must set additional settings for Mongo, Qdrant, and AWS. If you are just working locally, the default values of these env vars will work out of the box.

```env
DATABASE_HOST=your_mongodb_url
USE_QDRANT_CLOUD=true
QDRANT_CLOUD_URL=your_qdrant_cloud_url
QDRANT_APIKEY=your_qdrant_api_key
AWS_REGION=eu-central-1 # Change it with your AWS region.
AWS_ACCESS_KEY=your_aws_access_key
AWS_SECRET_KEY=your_aws_secret_key
```

> Details on how to obtain appropiate credentials can be found in the official [repository](https://github.com/PacktPublishing/LLM-Engineers-Handbook) and [book](https://www.amazon.com/LLM-Engineers-Handbook-engineering-production/dp/1836200072/).

## 🏗️ Infrastructure

> More details on local and cloud infrastructure setups are available in the official [repository](https://github.com/PacktPublishing/LLM-Engineers-Handbook) and [book](https://www.amazon.com/LLM-Engineers-Handbook-engineering-production/dp/1836200072/).

## 🏃 Run project

Based on the setup and usage steps described above, assuming the local and cloud infrastructure works and the `.env` is filled as expected, follow the next steps to run the LLM system end-to-end:

### Data

- [x] Collect data: `uv run poe run-arxiv-data-etl`

- [x] Compute features: `uv run poe run-feature-engineering-pipeline`

- [x] Compute instruct dataset: `uv run poe run-generate-instruct-datasets-pipeline`

- [x] Compute preference alignment dataset: `uv run poe run-generate-preference-datasets-pipeline`

### Training

> From now on, for these steps to work, you need to properly set up AWS SageMaker, such as running `uv sync --extra aws` and filling in the AWS-related environment variables and configs.

- [ ] SFT fine-tuning Llamma 3.1: `uv run poe run-training-pipeline`

- [ ] For DPO, go to `configs/training.yaml`, change `finetuning_type` to `dpo`, and run `uv run poe run-training-pipeline` again

- [ ] Evaluate fine-tuned models: `uv run poe run-evaluation-pipeline`

### Inference

> From now on, for these steps to work, you need to properly set up AWS SageMaker, such as running `uv sync --extra aws` and filling in the AWS-related environment variables and configs.

- [ ] Call only the RAG retrieval module: `uv run poe call-rag-retrieval-module`

- [ ] Deploy the LLM Twin microservice to SageMaker: `uv run poe deploy-inference-endpoint`

- [ ] Test the LLM Twin microservice: `uv run poe test-sagemaker-endpoint`

- [ ] Start end-to-end RAG server: `uv run poe run-inference-ml-service`

- [ ] Test RAG server: `uv run poe call-inference-ml-service`
