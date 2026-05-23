import logging
import warnings
from surya.settings import settings


def configure_logging():
    logger = get_logger()

    # Remove any existing handlers to prevent duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add our handler
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent propagation to parent loggers to avoid double logging
    logger.propagate = False

    logger.setLevel(settings.LOGLEVEL)
    warnings.simplefilter(action="ignore", category=FutureWarning)


def get_logger():
    return logging.getLogger("surya")
