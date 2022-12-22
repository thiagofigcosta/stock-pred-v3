import codecs
import re
import urllib
import urllib.request
from typing import Optional

from hyperparameters import Hyperparameters
from logger import info, exception, verbose
from prophet_filepaths import getTickerFilename, getRawTickerFilepath
from utils_date import timestampToDateStr, dateStrToTimestamp, now
from utils_fs import pathExists

YAHOO_API = 'https://query1.finance.yahoo.com/v7/finance'
_is_null_or_nan_field_c_regex = re.compile(r'((.*null|nan),.*|.*,(null|nan).*)', flags=re.IGNORECASE)


def downloadTicker(ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None,
                   force: bool = False, configs: Optional[Hyperparameters] = None) -> str:
    if start_date is None:
        start_date = '01/01/1970'
    if end_date is None:
        end_date = timestampToDateStr(now())
    if configs is None:
        configs = Hyperparameters.getDefault()
    ticker = ticker.upper()
    filename = getTickerFilename(ticker, start_date, end_date, configs=configs)
    uuid_filepath = getRawTickerFilepath(filename)
    filepath = getRawTickerFilepath(filename, remove_uuid=True)

    if pathExists(filepath) and not force:
        return uuid_filepath

    download_url = f'{YAHOO_API}/download/{ticker}?period1={int(dateStrToTimestamp(start_date))}' \
                   f'&period2={int(dateStrToTimestamp(end_date))}&interval=1d&events=history&includeAdjustedClose=true'
    info(f'Downloading ticker `{ticker}` data: {download_url}...')
    content = None
    with urllib.request.urlopen(download_url) as response:
        if response.code == 200:
            content = response.read().decode('utf-8')
            info(f'Downloaded ticker `{ticker}` data successfully!')
        else:
            exception(Exception(f'Failed to download, server returned response code {response.code}'), raise_it=True)
    if content is not None:
        content = filterOutNullLines(content)
        with codecs.open(filepath, 'w', 'utf-8') as file:
            file.write(content)
    info(f'Saved ticker `{ticker}` at {filepath}.')
    return uuid_filepath


def filterOutNullLines(content: str) -> str:
    verbose('Dropping null rows...')
    lines = content.split('\n')
    content = ''
    for line in lines:
        if not re.match(_is_null_or_nan_field_c_regex, line):
            content += line + '\n'
    verbose('Dropped null rows!')
    return content[:-1]
