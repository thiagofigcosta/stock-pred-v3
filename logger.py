import os
import socket
import sys
import traceback
from datetime import datetime
from enum import auto, Enum
from io import StringIO
from os import path
from typing import Any, Optional

from utils_fs import createFolder


class Level(Enum):
    VERBOSE = auto()
    INFO = auto()
    WARN = auto()
    ERROR = auto()
    EXCEPTION = auto()
    FATAL = auto()
    NO_LOGS_AT_ALL = auto()

    def __str__(self) -> str:
        return self.name

    def __gt__(self, other) -> bool:
        return self.value > other.value

    def __lt__(self, other) -> bool:
        return self.value < other.value

    def __ge__(self, other) -> bool:
        return self.value >= other.value

    def __le__(self, other) -> bool:
        return self.value <= other.value

    def __eq__(self, other) -> bool:
        return self.value == other.value

    def __ne__(self, other) -> bool:
        return self.value != other.value


class LevelFallback(Enum):
    NO_FALLBACK = auto()  # the filtered out message is ignored
    FILE_ONLY = auto()  # the filtered out message is appended only on file
    SCREEN_ONLY = auto()  # the filtered out message is shown only on screen
    ANOTHER_FILE = auto()  # the filtered out message is appended on another file

    def __str__(self) -> str:
        return self.name


LOG_FOLDER = 'logs'
FILE_EXTENSION = 'txt'
APP_NAME = 'application'
DATETIME_FORMAT = '%d/%m/%Y-%H:%M:%S'
DATE_FORMAT = '%Y%m%d'
_PRINT = True
_LEVEL = Level.INFO
_LEVEL_FALLBACK = LevelFallback.ANOTHER_FILE
_EYE_CATCHER = False
_FULL_HOSTNAME = False
_THE_HOSTNAME = None

createFolder(LOG_FOLDER)


def configure(log_folder: Optional[str] = None, level: Optional[Level] = None,
              print_on_screen: Optional[bool] = None, eye_catcher: Optional[bool] = None,
              datetime_format: Optional[str] = None, date_format: Optional[str] = None,
              full_hostname: Optional[bool] = None, name: Optional[str] = None) -> None:
    global LOG_FOLDER, _PRINT, _LEVEL, APP_NAME, _EYE_CATCHER, DATETIME_FORMAT, DATE_FORMAT, _FULL_HOSTNAME
    if log_folder is not None:
        LOG_FOLDER = log_folder
    if print_on_screen is not None:
        _PRINT = print_on_screen
    if level is not None:
        _LEVEL = level
    if name is not None:
        APP_NAME = name
    if eye_catcher is not None:
        _EYE_CATCHER = eye_catcher
    if datetime_format is not None:
        DATETIME_FORMAT = datetime_format
    if date_format is not None:
        DATE_FORMAT = date_format
    if full_hostname is not None:
        _FULL_HOSTNAME = full_hostname


def _log(message: Any, is_error: bool = False, has_traceback: bool = False, is_warn: bool = False,
         is_fatal: bool = False, is_verbose: bool = False, is_clean: bool = False) -> None:
    global _THE_HOSTNAME
    if type(message) is not str:
        message = str(message)

    filtered_out = False
    if not is_clean:
        now = datetime.now()
        now_str = f'{now.strftime(DATETIME_FORMAT)}.{now.microsecond:06d}'
        if _THE_HOSTNAME is None:
            _THE_HOSTNAME = socket.gethostname()
        hostname = _THE_HOSTNAME
        if not _FULL_HOSTNAME:
            hostname = hostname.split('.')[0]
        info_header = f"[{hostname}|{now_str}] "
        if is_fatal:
            info_header += '- FATAL: '
            if _LEVEL > Level.FATAL:
                filtered_out = True
        elif is_error:
            info_header += '- ERROR: '
            if _LEVEL > Level.ERROR:
                filtered_out = True
        elif has_traceback:
            info_header += '- EXCEPTION: '
            if _LEVEL > Level.EXCEPTION:
                filtered_out = True
        elif is_warn:
            info_header += '- WARN: '
            if _LEVEL > Level.WARN:
                filtered_out = True
        elif is_verbose:
            info_header += '- VERB: '
            if _LEVEL > Level.VERBOSE:
                filtered_out = True
        else:
            info_header += '- INFO: '
        if _LEVEL > Level.INFO:
            filtered_out = True
    else:
        info_header = ''
        if _LEVEL > Level.INFO:
            filtered_out = True

    fail_delimiter = "***********************************************"
    error_header = "*--------------------ERROR--------------------*"
    traceb_header = "*------------------TRACE_BACK------------------"
    if (is_error or has_traceback) and _EYE_CATCHER:
        formatted_message = f'{info_header}\n{fail_delimiter}\n'
        if is_error:
            formatted_message += f'{error_header}\n'
        if has_traceback:
            formatted_message += f'{traceb_header}\n'
        formatted_message += f'{fail_delimiter}\n'
        formatted_message += f'{message}\n'
        formatted_message += fail_delimiter
    else:
        formatted_message = info_header + message

    can_print = _PRINT and (not filtered_out or _LEVEL_FALLBACK == LevelFallback.SCREEN_ONLY)
    can_file_main = (not filtered_out or _LEVEL_FALLBACK == LevelFallback.FILE_ONLY)
    should_file_second = (filtered_out and _LEVEL_FALLBACK == LevelFallback.ANOTHER_FILE)

    if can_print:
        if is_error:
            sys.stderr.write(formatted_message + '\n')
            sys.stderr.flush()
        else:
            print(formatted_message, flush=True)

    if can_file_main or should_file_second:
        log_filename = getLogFilename(should_file_second)
        if log_filename is not None:
            with open(log_filename, 'a') as logfile:
                logfile.write(formatted_message + '\n')


def _handleException(e: Exception) -> str:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    exception_str = f'\n*** message: {e}\n'
    if exc_traceback is not None:
        fname = os.path.split(exc_traceback.tb_frame.f_code.co_filename)[1]
        exception_str = f"*** file_name: {fname}\n*** exception:\n{e}\n*** print_tb:\n"

    str_io = StringIO()
    traceback.print_tb(exc_traceback, limit=1, file=str_io)
    exception_str += f'{str_io.getvalue()}*** print_exception:\n'
    str_io = StringIO()
    traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=str_io)
    exception_str += f'\t{str_io.getvalue()}*** print_exc:\n'
    str_io = StringIO()
    traceback.print_exc(limit=2, file=str_io)
    exception_str += f'\t{str_io.getvalue()}*** format_exc, first and last line:\n'
    formatted_lines = traceback.format_exc().splitlines()
    exception_str += f'\t{formatted_lines[0]}\n\t{formatted_lines[-1]}\n'
    format_exception = traceback.format_exception(exc_type, exc_value, exc_traceback)
    if len(format_exception) > 0:
        exception_str += "*** format_exception:\n"
        for el in format_exception:
            str_el = str(el)
            if not str_el.startswith('\t'):
                str_el = f'\t{str_el}'
            exception_str += f'{str_el}\n'

    extract_tb = traceback.extract_tb(exc_traceback)
    if len(extract_tb) > 0:
        exception_str += "*** extract_tb:\n"
        for el in extract_tb:
            str_el = str(el)
            if not str_el.startswith('\t'):
                str_el = f'\t{str_el}'
            exception_str += f'{str_el}\n'

    format_tb = traceback.format_tb(exc_traceback)
    if len(format_tb) > 0:
        exception_str += "*** format_tb:\n"
        for el in format_tb:
            str_el = str(el)
            if not str_el.startswith('\t'):
                str_el = f'\t{str_el}'
            exception_str += f'{str_el}\n'
    if exc_traceback is not None:
        exception_str += f"*** At line: {exc_traceback.tb_lineno}"
    return exception_str


def getLogFilename(filtered_out: bool = False) -> Optional[str]:
    if LOG_FOLDER is None:
        return None
    now = datetime.now()
    filename = f'logs-{APP_NAME}-{now.strftime(DATE_FORMAT)}{"_filtered_out" if filtered_out else ""}.{FILE_EXTENSION}'
    return path.join(LOG_FOLDER, filename)


def logDict(dictionary: dict, name: Optional[str] = None, tabs: int = 0, inline: bool = False,
            do_log: bool = True) -> str:
    if not do_log:
        return ''
    start = ''
    to_print_all = ''
    inline_str = "" if inline else "\n"
    tabs_str = "\t" * tabs
    if name is not None:
        to_print = f'{tabs_str}{name}:{inline_str}'
        if inline:
            to_print_all += to_print
        elif do_log:
            info(to_print)
        start = ' | ' if inline else '\t'

    first = True
    for key, value in dictionary.items():
        first_str = " " if first and inline else start
        to_print = f'{tabs_str}{first_str}{key}: {value}{inline_str}'
        if inline:
            to_print_all += to_print
        elif do_log:
            clean(to_print)
        first = False
    return to_print_all


def multiline(messages: str, do_log: bool = True):
    if do_log:
        for message in messages.split('\n'):
            _log(message, is_error=False, has_traceback=False, is_warn=False, is_verbose=False)


def info(message: Any = '', do_log: bool = True):
    if do_log:
        _log(message, is_error=False, has_traceback=False, is_warn=False, is_verbose=False)


def fatal(message_or_exception: Any):
    if isinstance(message_or_exception, Exception):
        message_or_exception = _handleException(message_or_exception)
    error(message_or_exception, is_fatal=True)


def error(message: Any, is_fatal=False):
    _log(message, is_error=True, has_traceback=False, is_warn=False, is_verbose=False, is_fatal=is_fatal)
    if is_fatal:
        exit(1)


def exception(e: Exception, raise_it: bool):
    _log(_handleException(e), is_error=False, has_traceback=True, is_warn=False, is_verbose=False, is_fatal=False)
    if raise_it:
        raise e


def warn(message: Any, do_log: bool = True):
    if do_log:
        _log(message, is_error=False, has_traceback=False, is_warn=True, is_verbose=False)


def clean(message: Any, is_error: bool = False, is_warn: bool = False, is_verbose: bool = False, do_log: bool = True):
    if do_log:
        _log(message, is_clean=True, is_error=is_error, is_warn=is_warn, is_verbose=is_verbose)


def verbose(message: Any, do_log: bool = True):
    if _LEVEL and do_log:
        _log(message, is_error=False, has_traceback=False, is_warn=False, is_verbose=True)


def getLevel() -> Level:
    return _LEVEL


def getVerbose() -> bool:
    return _LEVEL == Level.VERBOSE
