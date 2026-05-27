from typing import Callable, Type
from src.algorithms.choices import AlgorithmChoice
from src.logging.base_logger import BaseLogger


class LoggerFactory:
    """Factory for creating algorithm-specific loggers."""

    _loggers: dict[AlgorithmChoice, Type[BaseLogger]] = {}

    @classmethod
    def register_logger(
        cls, algorithm: AlgorithmChoice, logger_class: Type[BaseLogger]
    ) -> None:
        """Register a logger for an algorithm."""
        if algorithm in cls._loggers:
            raise ImportError(f"Duplicate logger registration for {algorithm}")
        cls._loggers[algorithm] = logger_class

    @classmethod
    def create_logger(cls, algorithm: AlgorithmChoice, config) -> BaseLogger:
        """Create a logger for the specified algorithm."""
        if algorithm in cls._loggers:
            return cls._loggers[algorithm](config)
        raise ValueError(
            f"No logger registered for {algorithm}. "
            f"Ensure the logger module is imported and "
            f"@register_logger(AlgorithmChoice.{algorithm.name}) is applied."
        )


def register_logger(
    choice: AlgorithmChoice,
) -> Callable[[type[BaseLogger]], type[BaseLogger]]:
    """Class decorator that registers a logger with the LoggerFactory.

    Usage::

        @register_logger(AlgorithmChoice.DES)
        class DESLogger(BaseLogger[DESLogData]):
            ...

    Raises:
        ImportError: if the same AlgorithmChoice is registered twice.
    """

    def decorator(cls: type[BaseLogger]) -> type[BaseLogger]:
        LoggerFactory.register_logger(choice, cls)
        return cls

    return decorator
