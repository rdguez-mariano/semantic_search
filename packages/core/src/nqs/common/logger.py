import inspect
import os
import platform
from enum import Enum
from logging import Formatter, Logger, StreamHandler, getLogger, handlers
from typing import Union

import pandas as pd


class LogLevel(Enum):
    INFO = "INFO"
    DEBUG = "DEBUG"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


HOSTNAME = platform.node()


class LogFormat(Enum):
    DETAIL = f"[{HOSTNAME}] [%(asctime)s] [%(process)d] \
[%(name)s] [%(levelname)s]: %(message)s"
    SIMPLE = "%(asctime)s - %(name)s - %(levelname)s: %(message)s"


LOG_LEVEL = os.environ.get("LOG_LEVEL", LogLevel.DEBUG.value).upper()
LOG_FILE_NAME = "console.log"


def get_nqs_logger(name: str) -> Logger:
    # create formatter
    formatter = Formatter(LogFormat.DETAIL.value)

    # file handler
    file_handler = handlers.RotatingFileHandler(
        LOG_FILE_NAME,
        encoding="utf-8",
        maxBytes=10000000,
        backupCount=2,
        mode="a",
    )
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(formatter)

    # create console handler
    console_handler = StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(formatter)

    # logger
    logger = getLogger(name)
    logger.setLevel(LOG_LEVEL)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # avoid log duplicates with celery
    logger.propagate = False

    return logger


main_logger = get_nqs_logger("main")
# want a more detail one:
# logger = get_nqs_logger(os.path.basename(__file__))


def params_str(**kwargs):
    joined_str = ""
    for k, v in kwargs.items():
        if isinstance(v, dict):
            v_keys = list(v)
            if len(v_keys) > 0:
                # kf, kl = v_keys[0], v_keys[-1]
                # pstr = f"{k}=" + "{" + f"{kf}:{v[kf]}, ... ,{kl}:{v[kl]}" + "}" # noqa
                kf = v_keys[0]
                pstr = f"{k}=" + "{" + f"{kf}:{v[kf]}, ... " + "}"
            else:
                pstr = f"{k}=" + "{}"
        elif isinstance(v, list):
            if len(v) > 0:
                # pstr = f"{k}=[{v[0]}, ... ,{v[-1]}]"
                pstr = f"{k}=[{v[0]}, ... ]"
            else:
                pstr = f"{k}=[]"
        elif isinstance(v, pd.DataFrame):
            pstr = f"{k}=DataFrame(\n{v}\n)"

        else:
            pstr = f"{k}={v}"
        joined_str += "\n     " + pstr
    return f"(({joined_str}\n))\n"


def log_entering_fn(
    logger: Logger = main_logger,
    suffix: str = "",
    print_fn_locals: bool = True,
) -> None:
    if print_fn_locals:
        fn_locals = inspect.stack()[1][0].f_locals
        str_locals = params_str(**fn_locals)
    else:
        str_locals = ""

    fn_name = inspect.stack()[1][3]
    logger.debug(f"---> {fn_name}{str_locals}{suffix}")


def log_exiting_fn(
    logger: Logger = main_logger,
    suffix: str = "",
    print_fn_locals: bool = False,
) -> None:
    if print_fn_locals:
        fn_locals = inspect.stack()[1][0].f_locals
        str_locals = params_str(**fn_locals)
    else:
        str_locals = ""

    fn_name = inspect.stack()[1][3]
    logger.debug(f"<- {fn_name}{str_locals}{suffix}")


def log_debug(
    msg: str,
    logger: Logger = main_logger,
):
    log_msg(msg, logger=logger, into_level=LogLevel.DEBUG, stack_depth=2)


def log_info(
    msg: str,
    logger: Logger = main_logger,
):
    log_msg(msg, logger=logger, into_level=LogLevel.INFO, stack_depth=2)


def log_error(
    msg: str,
    logger: Logger = main_logger,
):
    log_msg(msg, logger=logger, into_level=LogLevel.ERROR, stack_depth=2)


def log_critical(
    msg: str,
    logger: Logger = main_logger,
):
    log_msg(msg, logger=logger, into_level=LogLevel.CRITICAL, stack_depth=2)


def log_warning(
    msg: str,
    logger: Logger = main_logger,
):
    log_msg(msg, logger=logger, into_level=LogLevel.WARNING, stack_depth=2)


def raise_error(
    msg_obj: Union[str, Exception],
    into_level: LogLevel = LogLevel.ERROR,
    stack_depth: int = 2,
) -> None:
    if isinstance(msg_obj, str):
        msg_str = msg_obj
        msg_exception = Exception(msg_obj)
    elif issubclass(msg_obj.__class__, Exception):
        msg_exception = msg_obj
        msg_str = str(msg_obj)

    log_msg(msg_str, into_level=into_level, stack_depth=stack_depth)
    raise msg_exception


def log_msg(
    msg: str,
    logger: Logger = main_logger,
    into_level: LogLevel = LogLevel.DEBUG,
    stack_depth: int = 1,
) -> None:
    fn_name = inspect.stack()[stack_depth][3]

    if into_level == LogLevel.DEBUG:
        call_fn = logger.debug
    elif into_level == LogLevel.INFO:
        call_fn = logger.info
    elif into_level == LogLevel.ERROR:
        call_fn = logger.error
    elif into_level == LogLevel.WARNING:
        call_fn = logger.warning
    elif into_level == LogLevel.CRITICAL:
        call_fn = logger.critical

    call_fn(f"<fn={fn_name}> {msg}")
