"""Data ingestion package."""
from src.data_ingestion.ingestion_utils import compute_sha256, profile_dataframe
from src.data_ingestion.source_registry import get_source_registry, SOURCES
from src.data_ingestion.icrisat_ingestion import ICRISATIngestion
from src.data_ingestion.govt_ogd_ingestion import GovtOGDIngestion
from src.data_ingestion.faostat_ingestion import FAOSTATIngestion
