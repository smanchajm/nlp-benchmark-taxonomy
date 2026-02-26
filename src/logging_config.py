import logging
import sys

# Third-party loggers that are too noisy at WARNING level for normal use.
_QUIET_LOGGERS = {
    "acl-anthology": logging.ERROR,
}


def setup_logging(level: int = logging.INFO, verbose: bool = False) -> None:
    """Configure the root logger for the project.

    Args:
        level: Log level for project loggers (default: INFO).
        verbose: If True, third-party library loggers are not silenced.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    if not verbose:
        for name, lib_level in _QUIET_LOGGERS.items():
            logging.getLogger(name).setLevel(lib_level)
