import json
from pathlib import Path

import click
from loguru import logger
from mongoengine import Document

from llm_engineering import settings
from llm_engineering.domain.documents import ExpertDocument, PaperDocument


@click.command()
@click.option(
    "--export-raw-data",
    is_flag=True,
    default=False,
    help="Whether to export your data warehouse to a JSON file.",
)
@click.option(
    "--import-raw-data",
    is_flag=True,
    default=False,
    help="Whether to import a JSON file into your data warehouse.",
)
@click.option(
    "--data-dir",
    default=Path("data/data_warehouse_raw_data"),
    type=Path,
    help="Path to the directory containing data warehouse raw data JSON files.",
)
def main(
    export_raw_data,
    import_raw_data,
    data_dir: Path,
) -> None:
    assert export_raw_data or import_raw_data, "Specify at least one operation."

    # Initialize MongoDB connection
    settings.init_mongodb()

    if export_raw_data:
        __export(data_dir)

    if import_raw_data:
        __import(data_dir)


def __export(data_dir: Path) -> None:
    """Export collections to JSON files."""
    logger.info(f"Exporting data warehouse to {data_dir}...")
    data_dir.mkdir(parents=True, exist_ok=True)

    __export_data_category(data_dir, PaperDocument)
    __export_data_category(data_dir, ExpertDocument)


def __export_data_category(data_dir: Path, category_class: type[Document]) -> None:
    """Export a single collection to JSON."""
    # Get all documents
    documents = category_class.objects()

    # Convert to JSON-serializable format
    serialized_data = []
    for doc in documents:
        data = doc.to_mongo().to_dict()  # Convert to dict format
        # Convert ObjectId to string for JSON serialization
        data["_id"] = str(data["_id"])
        if "expert_id" in data:  # Handle ReferenceField
            data["expert_id"] = str(data["expert_id"])
        serialized_data.append(data)

    export_file = data_dir / f"{category_class.__name__}.json"

    logger.info(f"Exporting {len(serialized_data)} items of {category_class.__name__} to {export_file}...")
    with export_file.open("w") as f:
        json.dump(serialized_data, f, default=str)  # Use default=str for datetime objects


def __import(data_dir: Path) -> None:
    """Import collections from JSON files."""
    logger.info(f"Importing data warehouse from {data_dir}...")
    assert data_dir.is_dir(), f"{data_dir} is not a directory or it doesn't exist."

    data_category_classes = {
        "PaperDocument": PaperDocument,
        "ExpertDocument": ExpertDocument,
    }

    for file in data_dir.iterdir():
        if not file.is_file():
            continue

        category_class_name = file.stem
        category_class = data_category_classes.get(category_class_name)
        if not category_class:
            logger.warning(f"Skipping {file} as it does not match any data category.")
            continue

        __import_data_category(file, category_class)


def __import_data_category(file: Path, category_class: type[Document]) -> None:
    """Import a single collection from JSON."""
    with file.open("r") as f:
        data = json.load(f)

    logger.info(f"Importing {len(data)} items of {category_class.__name__} from {file}...")

    for item in data:
        # Convert string IDs back to ObjectId
        if "_id" in item:
            del item["_id"]  # Let MongoDB generate new ID
        if "expert_id" in item and isinstance(item["expert_id"], str):
            # Find the referenced expert
            expert = ExpertDocument.objects(id=item["expert_id"]).first()
            if expert:
                item["expert_id"] = expert
            else:
                logger.warning(f"Expert with id {item['expert_id']} not found, skipping paper")
                continue

        try:
            doc = category_class(**item)
            doc.save()
        except Exception as e:
            logger.error(f"Failed to import document: {e}")


if __name__ == "__main__":
    main()
