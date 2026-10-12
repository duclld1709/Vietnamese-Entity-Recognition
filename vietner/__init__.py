"""Vietnamese named entity recognition and personal-data anonymisation."""

__version__ = "2.0.0"

from vietner.anonymize import anonymize, deanonymize  # noqa: E402
from vietner.pipeline import VietNER  # noqa: E402
from vietner.schemas import AnalyzeResult, AnonymizeResult, Entity  # noqa: E402

__all__ = ["AnalyzeResult", "AnonymizeResult", "Entity", "VietNER", "__version__", "anonymize", "deanonymize"]
