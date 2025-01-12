<div align="center">
  <h1>🔬 ArXiv Domain Expert LLM</h1>
  <p>A comprehensive LLM system trained on ArXiv papers to create a domain expert</p>
  
  
  <p><em>This project is adapted from the excellent <a href="https://github.com/PacktPublishing/LLM-Engineers-Handbook">LLM Engineer's Handbook</a> by <a href="https://github.com/iusztinpaul">Paul Iusztin</a> and <a href="https://github.com/mlabonne">Maxime Labonne</a>. While maintaining the robust MLOps architecture and best practices from the original work, this adaptation implements a complete academic paper processing pipeline.</em></p>
</div>

## 🔄 Key Adaptations from Original Work

This project maintains the robust MLOps infrastructure of the original LLM Engineer's Handbook while implementing significant changes:

- **Different Purpose**: Instead of creating a digital twin, this project aims to develop an ML expert LLM
- **New Data Source**: Replaced LinkedIn/Medium/Substack pipelines with ArXiv paper processing
- **Modified Training Objective**: Adapted training approach to focus on ML domain expertise
- **Enhanced ODM**: Migrated from custom MongoDB ODM to mongoengine for production reliability
- **Custom Features**: Implemented specialized feature engineering for academic paper processing using IBM's docling library for
  comprehensive PDF parsing

The project leverages the original architecture's excellent MLOps practices while implementing these substantial changes to serve a different use case.

## 🌟 Features

The goal of this project is to create a end-to-end production-ready LLM system that can:

- 📝 Data collection & generation from ArXiv's papers
- 🔄 LLM training pipeline
- 📊 Simple RAG system
- 🚀 Production-ready AWS deployment
- 🔍 Comprehensive monitoring
- 🧪 Testing and evaluation framework

You can download and use the final trained model with ML's papers on [Hugging Face]
(https://huggingface.co/mlabonne/TwinLlama-3.1-8B-DPO).

For detailed implementation of MLOps practices and infrastructure setup, please refer to the original [LLM Engineer's Handbook Repository](https://github.com/PacktPublishing/LLM-Engineers-Handbook) and the [LLM Engineer's Handbook Book](https://www.amazon.com/LLM-Engineers-Handbook-engineering-production/dp/1836200072/).

## 🔗 Dependencies

### Local dependencies

| Tool    | Version  | Purpose                        |
| ------- | -------- | ------------------------------ |
| uv      | latest   | Fast Python package management |
| Python  | 3.11     | Runtime environment            |
| Docker  | ≥27.1.1  | Containerization               |
| AWS CLI | ≥2.15.42 | Cloud management               |
| Git     | ≥2.44.0  | Version control                |

### Cloud Services

The project uses the following services (setup instructions provided in deployment section):
| Service | Purpose |
| ---------------------------------------------------| -------------------------------- |
| [HuggingFace](https://huggingface.com/) | Model registry |
| [Comet ML](https://www.comet.com/site/) | Experiment tracker |
| [Opik](https://www.comet.com/site/products/opik/) | Prompt monitoring |
| [ZenML](https://www.zenml.io/) | Orchestrator and artifacts layer |
| [AWS](https://aws.amazon.com/) | Compute and storage |
| [MongoDB](https://www.mongodb.com/) | NoSQL database |
| [Qdrant](https://qdrant.tech/) | Vector database |
| [GitHub Actions](https://github.com/features/actions)| CI/CD pipeline |

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

First, clone the repository and navigate to the project directory:

```bash
git clone https://github.com/danivpv/arxiv-domain-expert-llm.git
cd arxiv-domain-expert-llm
```

Then, install the dependencies:

```bash
uv sync
```

Optionally, if you plan to commit code, you can install the `pre-commit` hooks:

```bash
uv sync --extra dev
uv run pre-commit install
```

Optionally, activate the virtual environment. Although `uv run` will automatically discover the virtual environment of the project, this is necessary to get the right interpreter using `which python` as expected by other workflows with `venv`, `pip`, `poetry`, etc...

```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

This will:

- Configure `uv` to use Python 3.11
- Install project dependencies (excluding AWS-specific packages)
- Set up pre-commit hooks for code verification

As our task manager (similar to `Makefile`), we run all the scripts using [Poe the Poet](https://poethepoet.natn.io/index.html) defined in pyproject.toml.

```bash
uv run poe ...
```

### 2. Local Development Setup

### 2. Local Development Setup

After you have installed all the dependencies, you must create and fill a `.env` file with your credentials to appropriately interact with other services and run the project. Setting your sensitive credentials in a `.env` file is a good security practice, as this file won't be committed to GitHub or shared with anyone else.

1. First, copy our example by running the following:

```bash
cp .env.example .env # The file must be at your repository's root!
```

2. Now, let's understand how to fill in all the essential variables within the `.env` file to get you started. The following are the mandatory settings we must complete when working locally:

#### OpenAI

To authenticate to OpenAI's API, you must fill out the `OPENAI_API_KEY` env var with an authentication token.

```env
OPENAI_API_KEY=your_api_key_here
```

→ Check out this [tutorial](https://platform.openai.com/docs/quickstart) to learn how to provide one from OpenAI.

#### Hugging Face

To authenticate to Hugging Face, you must fill out the `HUGGINGFACE_ACCESS_TOKEN` env var with an authentication token.

```env
HUGGINGFACE_ACCESS_TOKEN=your_token_here
```

→ Check out this [tutorial](https://huggingface.co/docs/hub/en/security-tokens) to learn how to provide one from Hugging Face.

#### Comet ML & Opik

To authenticate to Comet ML (required only during training) and Opik, you must fill out the `COMET_API_KEY` env var with your authentication token.

```env
COMET_API_KEY=your_api_key_here
```

→ Check out this [tutorial](https://www.comet.com/docs/v2/api-and-sdk/rest-api/overview/) to learn how to get the Comet ML variables from above. You can also access Opik's dashboard using 🔗[this link](https://www.comet.com/opik).

### 3. Deployment Setup

When deploying the project to the cloud, we must set additional settings for Mongo, Qdrant, and AWS. If you are just working locally, the default values of these env vars will work out of the box.

#### MongoDB

We must change the `DATABASE_HOST` env var with the URL pointing to your cloud MongoDB cluster.

```env
DATABASE_HOST=your_mongodb_url
```

→ Check out this [tutorial](https://www.mongodb.com/resources/products/fundamentals/mongodb-cluster-setup) to learn how to create and host a MongoDB cluster for free.

#### Qdrant

Change `USE_QDRANT_CLOUD` to `true`, `QDRANT_CLOUD_URL` with the URL point to your cloud Qdrant cluster, and `QDRANT_APIKEY` with its API key.

```env
USE_QDRANT_CLOUD=true
QDRANT_CLOUD_URL=your_qdrant_cloud_url
QDRANT_APIKEY=your_qdrant_api_key
```

→ Check out this [tutorial](https://qdrant.tech/documentation/cloud/create-cluster/) to learn how to create a Qdrant cluster for free

#### AWS

For your AWS set-up to work correctly, you need the AWS CLI installed on your local machine and properly configured with an admin user (or a user with enough permissions to create new SageMaker, ECR, and S3 resources; using an admin user will make everything more straightforward).

```bash
AWS_REGION=eu-central-1 # Change it with your AWS region.
AWS_ACCESS_KEY=your_aws_access_key
AWS_SECRET_KEY=your_aws_secret_key
```

## 🏗️ Infrastructure

> Details on local and cloud infrastructure setups are available in the official [repository](https://github.com/PacktPublishing/LLM-Engineers-Handbook) and [book](https://www.amazon.com/LLM-Engineers-Handbook-engineering-production/dp/1836200072/).

## 🏃 Run project

Based on the setup and usage steps described above, assuming the local and cloud infrastructure works and the `.env` is filled as expected, follow the next steps to run the LLM system end-to-end:

### Data

1. Collect data: `uv run poe run-digital-data-etl`

2. Compute features: `uv run poe run-feature-engineering-pipeline`

3. Compute instruct dataset: `uv run poe run-generate-instruct-datasets-pipeline`

4. Compute preference alignment dataset: `uv run poe run-generate-preference-datasets-pipeline`

### Training

> From now on, for these steps to work, you need to properly set up AWS SageMaker, such as running `uv sync --extra aws` and filling in the AWS-related environment variables and configs.

1. SFT fine-tuning Llamma 3.1: `uv run poe run-training-pipeline`

2. For DPO, go to `configs/training.yaml`, change `finetuning_type` to `dpo`, and run `uv run poe run-training-pipeline` again

3. Evaluate fine-tuned models: `uv run poe run-evaluation-pipeline`

### Inference

> From now on, for these steps to work, you need to properly set up AWS SageMaker, such as running `uv sync --extra aws` and filling in the AWS-related environment variables and configs.

1. Call only the RAG retrieval module: `uv run poe call-rag-retrieval-module`

2. Deploy the LLM Twin microservice to SageMaker: `uv run poe deploy-inference-endpoint`

3. Test the LLM Twin microservice: `uv run poe test-sagemaker-endpoint`

4. Start end-to-end RAG server: `uv run poe run-inference-ml-service`

5. Test RAG server: `uv run poe call-inference-ml-service`
