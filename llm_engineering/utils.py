import time
from contextlib import contextmanager
from functools import wraps
from typing import Callable

from loguru import logger


# You can also create a decorator for common logging patterns
def log_function_execution(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info({"event": f"{func.__name__}_started", "args_length": len(args), "kwargs_keys": list(kwargs.keys())})

        try:
            with log_execution_time(func.__name__):
                result = func(*args, **kwargs)

            logger.info({"event": f"{func.__name__}_completed", "success": True})
            return result

        except Exception as e:
            logger.error({"event": f"{func.__name__}_failed", "error": str(e)}, exc_info=True)
            raise

    return wrapper


@contextmanager
def log_execution_time(task_name: str):
    """Context manager for timing operations"""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.info({"event": f"{task_name}_duration", "duration_seconds": duration})
