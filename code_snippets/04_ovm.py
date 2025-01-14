import json
from uuid import UUID

from qdrant_client import QdrantClient
from rich import print as rprint
from rich.console import Console
from rich.table import Table


def get_qdrant_snapshot():
    """Get a snapshot of Qdrant database state"""
    try:
        # Initialize client and console
        client = QdrantClient("localhost", port=6333)
        console = Console()
        
        # Get all collections
        collections = client.get_collections()
        
        # Create table for collections overview
        table = Table(title="Qdrant Database Snapshot")
        table.add_column("Collection Name", style="cyan")
        table.add_column("Vector Size", style="magenta")
        table.add_column("Points Count", style="green")
        table.add_column("Distance", style="yellow")
        table.add_column("Status", style="blue")
        
        total_points = 0
        
        # Detailed information for each collection
        for collection in collections.collections:
            # Retrieve collection configuration
            collection_info = client.get_collection(collection.name)
            count_result = client.count(collection_name=collection.name)
            
            # Extract relevant information
            vector_params = collection_info.config.params.vectors
            vector_size = vector_params.size if hasattr(vector_params, 'size') else "N/A"
            distance = vector_params.distance.value if hasattr(vector_params, 'distance') else "N/A"
            points_count = count_result.count
            status = collection_info.status
            
            # Add row to table
            table.add_row(
                collection.name,
                str(vector_size),
                str(points_count),
                str(distance),
                str(status)
            )
            
            total_points += points_count
        
        # Print table
        console.print(table)
        
        # Print total points
        console.print(f"\nTotal points across all collections: {total_points}")
        
    except Exception as e:
        rprint(f"[red]Error connecting to Qdrant: {str(e)}[/red]")


if __name__ == "__main__":
    get_qdrant_snapshot()
