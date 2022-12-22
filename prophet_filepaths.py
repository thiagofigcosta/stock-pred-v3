import re
from typing import Optional

from hyperparameters import Hyperparameters
from utils_date import changeDateStrFormat
from utils_fs import getBasename, pathJoin, getDirName, createFolder, removeFileExtension, deleteFolder

DATASET_DIR = 'datasets'
RAW_SUBDIR = 'raw'
ENRICHED_SUBDIR = 'enriched'
TRANSFORMED_SUBDIR = 'transformed'
SCALER_SUBDIR = 'scaler'

TICKER_REGEX = r'[A-Z\^\.0-9]*'
_extract_ticker_c_regex = re.compile(fr'^({TICKER_REGEX})_.*')
_remove_uuid_c_regex = re.compile(fr'^({TICKER_REGEX}[a-zA-Z0-9_]*from-[0-9\-_]*to-[0-9\-]*)_uuid-[a-z0-9]*(\..*)')


def getTickerFromFilename(filename: str) -> str:
    filename = getBasename(filename)
    ticker = re.match(_extract_ticker_c_regex, filename).group(1)
    return ticker


def removeUuidFromTickerFilename(filename: str) -> str:
    filename = getBasename(filename)
    if '_uuid-' not in filename:
        return filename
    re_result = re.match(_remove_uuid_c_regex, filename)
    filename = re_result.group(1) + re_result.group(2)
    return filename


def removeUuidFromTickerFilepath(filepath: str) -> str:
    clean_basename = removeUuidFromTickerFilename(filepath)
    dir_name = getDirName(filepath)
    return pathJoin(dir_name, clean_basename)


def getTickerFilename(ticker: str, start_date: str, end_date: str, include_uuid: bool = True,
                      configs: Optional[object] = None) -> str:
    if include_uuid:
        if configs is None:
            configs = Hyperparameters.getDefault()
        the_id = f'_uuid-{configs.genDatasetEnricherPcaUuid()}'
    else:
        the_id = ''
    fmt_start_date = changeDateStrFormat(start_date)
    fmt_end_date = changeDateStrFormat(end_date)
    return f'{ticker}_daily_from-{fmt_start_date}_to-{fmt_end_date}{the_id}.csv'


def getRawTickerFilepath(filename: str, remove_uuid: bool = False) -> str:
    filepath = pathJoin(DATASET_DIR, RAW_SUBDIR, filename)
    if remove_uuid:
        return removeUuidFromTickerFilepath(filepath)
    else:
        return filepath


def getEnrichedTickerFilepath(filename: str) -> str:
    return pathJoin(DATASET_DIR, ENRICHED_SUBDIR, filename)


def getTransformedTickerFilepath(filename: str) -> str:
    return pathJoin(DATASET_DIR, TRANSFORMED_SUBDIR, filename)


def getTickerScalerFilepath(filename: str) -> str:
    filename_no_ext = removeFileExtension(filename)
    if not filename_no_ext.strip().endswith('.') and filename_no_ext.strip() != '':
        filename = f'{filename_no_ext}-scaler.bin'
    return pathJoin(DATASET_DIR, SCALER_SUBDIR, filename)


def clearFiles():
    from hyperparameters import HYPERPARAMETERS_DIR
    from logger import LOG_FOLDER
    from nas import NAS_DIR
    from plotter import SAVED_PLOTS_PATH
    from prophet import MODELS_DIR, PROPHET_DIR
    deleteFolder(DATASET_DIR)
    deleteFolder(HYPERPARAMETERS_DIR)
    deleteFolder(LOG_FOLDER)
    deleteFolder(MODELS_DIR)
    deleteFolder(NAS_DIR)
    deleteFolder(PROPHET_DIR)
    deleteFolder(SAVED_PLOTS_PATH)


createFolder(DATASET_DIR)
createFolder(getRawTickerFilepath(''))
createFolder(getEnrichedTickerFilepath(''))
createFolder(getTransformedTickerFilepath(''))
createFolder(getTickerScalerFilepath(''))
