import os
import socket
import sys
import traceback
from datetime import datetime
from io import StringIO
from os import path
from typing import Any, Optional

from utils_fs import createFolder

LOG_FOLDER = 'logs'
FILE_EXTENSION = 'txt'
APP_NAME = 'application'
DATETIME_FORMAT = '%d/%m/%Y-%H:%M:%S'
DATE_FORMAT = '%Y%m%d'
_PRINT = True
_VERBOSE = False
_EYE_CATCHER = False
_FULL_HOSTNAME = False

createFolder(LOG_FOLDER)


def configure(log_folder: Optional[str] = None, verbose: Optional[bool] = None,
              print_on_screen: Optional[bool] = None, eye_catcher: Optional[bool] = None,
              datetime_format: Optional[str] = None, date_format: Optional[str] = None,
              full_hostname: Optional[bool] = None, name: Optional[str] = None) -> None:
    global LOG_FOLDER, _PRINT, _VERBOSE, APP_NAME, _EYE_CATCHER, DATETIME_FORMAT, DATE_FORMAT, _FULL_HOSTNAME
    if log_folder is not None:
        LOG_FOLDER = log_folder
    if print_on_screen is not None:
        _PRINT = print_on_screen
    if verbose is not None:
        _VERBOSE = verbose
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


def _log(message: Any, error: bool = False, traceback: bool = False, warn: bool = False, fatal: bool = False,
         verb: bool = False,
         clean: bool = False) -> None:
    if type(message) is not str:
        message = str(message)

    if not clean:
        now = datetime.now()
        nowstr = f'{now.strftime(DATETIME_FORMAT)}.{now.microsecond:06d}'
        hostname = socket.gethostname()
        if not _FULL_HOSTNAME:
            hostname = hostname.split('.')[0]
        info_header = f"[{hostname}|{nowstr}] "
        if fatal:
            info_header += '- FATAL: '
        elif error:
            info_header += '- ERROR: '
        elif traceback:
            info_header += '- EXCEPTION: '
        elif warn:
            info_header += '- WARN: '
        elif verb:
            info_header += '- VERB: '
        else:
            info_header += '- INFO: '
    else:
        info_header = ''

    fail_delimiter = "***********************************************"
    error_header = "*--------------------ERROR--------------------*"
    traceb_header = "*------------------TRACE_BACK------------------"
    formatted_message = ""
    if (error or traceback) and _EYE_CATCHER:
        formatted_message = f'{info_header}\n{fail_delimiter}\n'
        if error:
            formatted_message += f'{error_header}\n'

        if traceback:
            formatted_message += f'{traceb_header}\n'

        formatted_message += f'{fail_delimiter}\n'
        formatted_message += f'{message}\n'
        formatted_message += fail_delimiter
    else:
        formatted_message = info_header + message

    if _PRINT:
        if error:
            sys.stderr.write(formatted_message + '\n')
            sys.stderr.flush()
        else:
            print(formatted_message, flush=True)

    log_filename = getLogFilename()
    if log_filename is not None:
        with open(log_filename, 'a') as logfile:
            logfile.write(formatted_message + '\n')


def _handleException(e: Exception) -> str:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    exceptionstr = f'\n*** message: {e}\n'
    if exc_traceback is not None:
        fname = os.path.split(exc_traceback.tb_frame.f_code.co_filename)[1]
        exceptionstr = "*** file_name: "
        exceptionstr += fname + '\n'
        exceptionstr += "*** exception:\n"
        exceptionstr += str(e) + "\n"
        exceptionstr += "*** print_tb:\n"

    str_io = StringIO()
    traceback.print_tb(exc_traceback, limit=1, file=str_io)
    exceptionstr += str_io.getvalue()
    exceptionstr += "*** print_exception:\n"
    str_io = StringIO()
    traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=str_io)
    exceptionstr += '\t' + str_io.getvalue()
    exceptionstr += "*** print_exc:\n"
    str_io = StringIO()
    traceback.print_exc(limit=2, file=str_io)
    exceptionstr += '\t' + str_io.getvalue()
    exceptionstr += "*** format_exc, first and last line:\n"
    formatted_lines = traceback.format_exc().splitlines()
    exceptionstr += '\t' + formatted_lines[0] + "\n"
    exceptionstr += '\t' + formatted_lines[-1] + "\n"
    format_exception = traceback.format_exception(exc_type, exc_value, exc_traceback)
    if len(format_exception) > 0:
        exceptionstr += "*** format_exception:\n"
        for el in format_exception:
            str_el = str(el)
            if not str_el.startswith('\t'):
                str_el = '\t' + str_el

            exceptionstr += f'{el}\n'

    extract_tb = traceback.extract_tb(exc_traceback)
    if len(extract_tb) > 0:
        exceptionstr += "*** extract_tb:\n"
        for el in extract_tb:
            str_el = str(el)
            if not str_el.startswith('\t'):
                str_el = '\t' + str_el

            exceptionstr += f'{el}\n'

    format_tb = traceback.format_tb(exc_traceback)
    if len(format_tb) > 0:
        exceptionstr += "*** format_tb:\n"
        for el in format_tb:
            str_el = str(el)
            if not str_el.startswith('\t'):
                str_el = '\t' + str_el

            exceptionstr += f'{el}\n'

    if exc_traceback is not None:
        exceptionstr += "*** At line: "
        exceptionstr += str(exc_traceback.tb_lineno)

    return exceptionstr


def getLogFilename() -> Optional[str]:
    if LOG_FOLDER is None:
        return None
    now = datetime.now()
    filename = f'logs-{APP_NAME}-{now.strftime(DATE_FORMAT)}.{FILE_EXTENSION}'
    return path.join(LOG_FOLDER, filename)


def logDict(dictionary: dict, name: Optional[str] = None, tabs: int = 0, inline: bool = False):
    start = ''
    to_print_all = ''
    inline_str = "" if inline else "\n"
    tabs_str = "\t" * tabs
    if name is not None:
        to_print = f'{tabs_str}{name}:{inline_str}'
        if inline:
            to_print_all += to_print
        else:
            info(to_print)
        start = ' | ' if inline else '\t'

    first = True
    for key, value in dictionary.items():
        first_str = " " if first and inline else start
        to_print = f'{tabs_str}{first_str}{key}: {value}{inline_str}'
        if inline:
            to_print_all += to_print
        else:
            info(to_print)
        first = False


def multiline(messages: str):
    for message in messages.split('\n'):
        _log(message, error=False, traceback=False, warn=False, verb=False)


def info(message: Any = '', do_log: bool = True):
    if do_log:
        _log(message, error=False, traceback=False, warn=False, verb=False)


def fatal(message: Any):
    error(message, fatal=True)


def error(message: Any, fatal: bool = False):
    _log(message, error=True, traceback=False, warn=False, verb=False, fatal=fatal)
    if fatal:
        exit(1)


def exception(e: Exception, raise_it: bool = False, fatal: bool = False):
    _log(_handleException(e), error=False, traceback=True, warn=False, verb=False, fatal=fatal)
    if fatal:
        exit(1)
    elif raise_it:
        raise e


def warn(message: Any):
    _log(message, error=False, traceback=False, warn=True, verb=False)


def clean(message: Any, error: bool = False, warn: bool = False, verb: bool = False, do_log: bool = True):
    if do_log:
        _log(message, clean=True, error=error, warn=warn, verb=verb)


def verbose(message: Any, do_log: bool = True):
    if _VERBOSE and do_log:
        _log(message, error=False, traceback=False, warn=False, verb=True)


def getVerbose() -> bool:
    return _VERBOSE
