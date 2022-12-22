import time
from datetime import datetime, timedelta
from typing import Union

import holidays

HUMAN_DATE_FORMAT = '%d/%m/%Y'
SAVE_DATE_FORMAT = '%Y-%m-%d'
HOLIDAY_DATE_FORMAT = '%d-%m-%Y'
DATETIME_FORMAT = '%d/%m/%Y-%H:%M:%S'


def dateObjToDateStr(date: datetime, output_format: str = HUMAN_DATE_FORMAT) -> str:
    return date.strftime(output_format)


def dateStrToDateObj(date: str, input_format: str = HUMAN_DATE_FORMAT) -> datetime:
    return datetime.strptime(date, input_format)


def dateStrToTimestamp(date: str, input_format: str = HUMAN_DATE_FORMAT) -> float:
    return float(time.mktime(datetime.strptime(date, input_format).timetuple()))


def timestampToDateStr(timestamp: Union[int, float], output_format: str = HUMAN_DATE_FORMAT) -> str:
    return datetime.fromtimestamp(timestamp).strftime(output_format)


def timestampToDateObj(timestamp: Union[int, float]) -> datetime:
    return datetime.fromtimestamp(timestamp)


def dateObjToTimestamp(date: datetime) -> float:
    return float(time.mktime(date.timetuple()))


def changeDateStrFormat(date: str, input_format: str = HUMAN_DATE_FORMAT, output_format: str = SAVE_DATE_FORMAT) -> str:
    return timestampToDateStr(dateStrToTimestamp(date, input_format), output_format)


def assertStrDateFormat(date_str, date_format=HUMAN_DATE_FORMAT):
    try:
        datetime.strptime(date_str, date_format)
    except ValueError:
        raise ValueError(f'The date ({date_str}) must obey the {date_format} format and must be a valid date')


def now() -> float:
    return time.perf_counter()


def getNowStr(output_format: str = DATETIME_FORMAT, include_micros: bool = False) -> str:
    now_dt = datetime.now()
    now_str = f'{now_dt.strftime(output_format)}' + (f'.{now_dt.microsecond:06d}' if include_micros else '')
    return now_str


def getNextWorkDays(from_date: datetime, n_days: int, where: str = 'usa') -> list[datetime]:
    if where.lower().replace('-', ' ').replace('_', ' ') in ('usa', 'us', 'united states', 'united states of america'):
        holiday_ck = holidays.USA()
    elif where.lower() in ('brasil', 'brazil', 'bra'):
        holiday_ck = holidays.Brazil()
    else:
        raise AttributeError(f'Unknown location: {where}')
    business_days_to_add = n_days
    current_date = from_date
    dates = []
    while business_days_to_add > 0:
        current_date += timedelta(days=1)
        weekday = current_date.weekday()
        if weekday >= 5 or dateObjToDateStr(current_date,
                                            output_format=HOLIDAY_DATE_FORMAT) in holiday_ck:  # sunday = 6
            continue
        business_days_to_add -= 1
        dates.append(current_date)
    return dates


def getNextStrWorkDays(from_date: str, n_days: int, date_format: str = SAVE_DATE_FORMAT,
                       cast_output: bool = True) -> list[str]:
    from_date = dateStrToDateObj(from_date, date_format)
    dates = getNextWorkDays(from_date, n_days)
    if cast_output:
        dates = [dateObjToDateStr(date, date_format) for date in dates]
    return dates
