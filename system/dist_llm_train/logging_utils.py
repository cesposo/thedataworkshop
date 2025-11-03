import logging
import os


def configure_logging(default_level: str = "INFO") -> None:
    """Configure root logging from environment.

    Respects LOG_LEVEL and LOG_FORMAT. If already configured, does nothing.
    """
    if logging.getLogger().handlers:
        return

    level_name = os.getenv("LOG_LEVEL", default_level).upper()
    level = getattr(logging, level_name, logging.INFO)

    log_format = os.getenv(
        "LOG_FORMAT",
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.basicConfig(level=level, format=log_format)

