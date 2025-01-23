import os

from dotenv import load_dotenv
from loguru import logger
from mongoengine import connect
from pydantic_settings import BaseSettings, SettingsConfigDict
from zenml.client import Client
from zenml.exceptions import EntityExistsError

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- Required settings even when working locally. ---

    # OpenAI API
    OPENAI_MODEL_ID: str = "gpt-4o-mini"
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

    # Huggingface API
    HUGGINGFACE_ACCESS_TOKEN: str | None = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

    # Comet ML (during training)
    COMET_API_KEY: str | None = os.getenv("COMET_API_KEY", None)
    COMET_PROJECT: str = os.getenv("COMET_PROJECT", "ml-arxiv-expert")

    # --- Required settings when deploying the code. ---
    # --- Otherwise, default values values work fine. ---

    # MongoDB database
    DATABASE_HOST: str = os.getenv("DATABASE_HOST")
    DATABASE_NAME: str = "twin"

    # Qdrant vector database
    USE_QDRANT_CLOUD: bool = os.getenv("USE_QDRANT_CLOUD", False)
    QDRANT_DATABASE_HOST: str = "localhost"
    QDRANT_DATABASE_PORT: int = 6333
    QDRANT_CLOUD_URL: str = "str"
    QDRANT_APIKEY: str | None = None

    # AWS Authentication
    AWS_REGION: str = os.getenv("AWS_REGION", "eu-east-1")
    AWS_ACCESS_KEY: str | None = os.getenv("AWS_REGION", None)
    AWS_SECRET_KEY: str | None = os.getenv("AWS_REGION", None)
    AWS_ARN_ROLE: str = os.getenv("AWS_REGION", "str")

    # --- Optional settings used to tweak the code. ---

    # AWS SageMaker
    HF_MODEL_ID: str = "danivpv/Llama-ML-Expert-DPO-1b"
    GPU_INSTANCE_TYPE: str = "ml.g5.2xlarge"
    SM_NUM_GPUS: int = 1
    MAX_INPUT_LENGTH: int = 2048
    MAX_TOTAL_TOKENS: int = 4096
    MAX_BATCH_TOTAL_TOKENS: int = 4096
    COPIES: int = 1  # Number of replicas
    GPUS: int = 1  # Number of GPUs
    CPUS: int = 2  # Number of CPU cores

    SAGEMAKER_ENDPOINT_CONFIG_INFERENCE: str = "twin"
    SAGEMAKER_ENDPOINT_INFERENCE: str = "twin"
    TEMPERATURE_INFERENCE: float = 0.01
    TOP_P_INFERENCE: float = 0.9
    MAX_NEW_TOKENS_INFERENCE: int = 150

    # RAG
    TEXT_EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKING_CROSS_ENCODER_MODEL_ID: str = "cross-encoder/ms-marco-MiniLM-L-4-v2"
    RAG_MODEL_DEVICE: str = "cpu"

    # LinkedIn Credentials
    LINKEDIN_USERNAME: str | None = None
    LINKEDIN_PASSWORD: str | None = None

    @property
    def OPENAI_MAX_TOKEN_WINDOW(self) -> int:
        official_max_token_window = {
            "gpt-3.5-turbo": 16385,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
        }.get(self.OPENAI_MODEL_ID, 128000)

        max_token_window = int(official_max_token_window * 0.90)

        return max_token_window

    @property
    def validate_huggingface_token(self) -> bool:
        """Validate that Hugging Face token is set and non-empty"""
        return bool(self.HUGGINGFACE_ACCESS_TOKEN and self.HUGGINGFACE_ACCESS_TOKEN.strip())

    @classmethod
    def load_settings(cls) -> "Settings":
        """
        Tries to load the settings from the ZenML secret store. If the secret does not exist, it initializes the settings from the .env file and default values.

        Returns:
            Settings: The initialized settings object.
        """

        try:
            logger.info("Loading settings from the ZenML secret store.")

            settings_secrets = Client().get_secret("settings")
            settings = Settings(**settings_secrets.secret_values)
        except (RuntimeError, KeyError):
            logger.warning(
                "Failed to load settings from the ZenML secret store. Defaulting to loading the settings from the '.env' file."
            )
            settings = Settings()

        return settings

    def export(self) -> None:
        """
        Exports the settings to the ZenML secret store.
        """

        env_vars = settings.model_dump()
        for key, value in env_vars.items():
            env_vars[key] = str(value)

        client = Client()

        try:
            client.create_secret(name="settings", values=env_vars)
        except EntityExistsError:
            logger.warning(
                "Secret 'scope' already exists. Delete it manually by running 'zenml secret delete settings', before trying to recreate it."
            )

    def init_mongodb(self):
        """Initialize MongoDB connection."""
        try:
            # Connect to MongoDB
            connect(db=self.DATABASE_NAME, host=self.DATABASE_HOST)
            logger.info(f"Connected to MongoDB at {self.DATABASE_HOST}:{self.DATABASE_NAME}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e!s}")
            raise


settings = Settings.load_settings()
