import warnings
from typing import Optional

import numpy as np
import pandas as pd
# Kinda messy to install TA-Lib:
# > brew install python@3.9 ; CPPFLAGS="$CPPFLAGS $(python3.9-config --include)" ; brew install ta-lib ; \
# pip install TA-Lib # https://mrjbq7.github.io/ta-lib/install.html
import talib

from hyperparameters import Hyperparameters
from logger import info, verbose
from prophet_filepaths import getEnrichedTickerFilepath, removeUuidFromTickerFilepath
from utils_date import dateStrToTimestamp, SAVE_DATE_FORMAT
from utils_fs import getBasename, pathExists
from utils_misc import getFibonacciSeq, safeLog
from utils_persistance import loadDataframe, saveDataframe

_fibonacci_sequence = None


def enrich(filepath: str, force: bool = False, configs: Optional[Hyperparameters] = None) -> str:
    filename = getBasename(filepath)
    dst_filepath = getEnrichedTickerFilepath(filename)

    if pathExists(dst_filepath) and not force:
        return dst_filepath

    info(f'Enriching dataset `{filename}`...')
    if configs is None:
        configs = Hyperparameters.getDefault()
    if not pathExists(filepath):
        filepath = removeUuidFromTickerFilepath(filepath)
    df = loadDataframe(filepath)
    if 'Close' in df:
        df['close_to_predict'] = df['Close'].copy()

    if configs.enricher.price_ratios_on:
        enrichPriceRatios(df)
    if configs.enricher.price_delta_on:
        enrichPriceDeltas(df)
    if configs.enricher.price_averages_on:
        enrichPriceAverages(df, configs.enricher.fast_average_window, configs.enricher.slow_average_window,
                            configs.enricher.fibonacci_seq_size)
    if configs.enricher.ta_lib_on:
        enrichTALib(df)

    filterOutNullRows(df)
    info(f'Enriched dataset `{filename}` successfully!')

    saveDataframe(df, dst_filepath)
    info(f'Saved enriched dataset at {dst_filepath}.')
    return dst_filepath


def enrichPriceRatios(df: pd.DataFrame) -> None:
    if 'Open' not in df or 'Close' not in df or 'Low' not in df or 'High' not in df:
        pass
    verbose('Enriching price ratios...')
    open_column = df['Open']
    close_column = df['Close']
    low_column = df['Low']
    high_column = df['High']
    df['oc'] = (close_column - open_column) / open_column
    df['oh'] = (high_column - open_column) / open_column
    df['ol'] = (low_column - open_column) / open_column
    df['ch'] = (high_column - close_column) / close_column
    df['cl'] = (low_column - close_column) / close_column
    df['lh'] = (high_column - low_column) / low_column
    verbose('Enriched price ratios!')


def enrichPriceDeltas(df: pd.DataFrame) -> None:
    if 'Close' not in df:
        return
    verbose('Enriching price deltas...')
    df['delta'] = df['Close'] - df['Close'].shift(1)
    df['up'] = df['delta'].apply(lambda x: x if pd.isnull(x) else 1 if x > 0 else 0)
    df['up'] = df['up'].astype('Int64')
    df['log_return'] = safeLog(df['Close']) - safeLog(df['Close'].shift(1))
    verbose('Enriched price deltas...')


def enrichPriceAverages(df: pd.DataFrame, fast_window: int = 13, slow_window: int = 21,
                        fibonacci_seq_sz: int = 10) -> None:
    if 'Close' not in df:
        return
    verbose('Enriching price averages on Close price...')
    df['fast_moving_avg'] = df['Close'].rolling(window=fast_window).mean()
    df.loc[:fast_window, 'fast_moving_avg'] = np.nan

    df['slow_moving_avg'] = df['Close'].rolling(window=slow_window).mean()
    df.loc[:slow_window, 'slow_moving_avg'] = np.nan

    df['fast_exp_moving_avg'] = df['Close'].ewm(span=fast_window).mean()
    df.loc[:fast_window, 'fast_exp_moving_avg'] = np.nan

    df['slow_exp_moving_avg'] = df['Close'].ewm(span=slow_window).mean()
    df.loc[:slow_window, 'slow_exp_moving_avg'] = np.nan

    df['fibonacci_moving_avg_close'] = fibonacciMovingAverage(df['Close'], fibonacci_seq_sz)
    df.loc[:fibonacci_seq_sz, 'fibonacci_moving_avg_close'] = np.nan
    df['fibonacci_weighted_moving_avg_close'] = fibonacciMovingAverage(df['Close'], fibonacci_seq_sz, weighted=True)
    df.loc[:fibonacci_seq_sz, 'fibonacci_weighted_moving_avg_close'] = np.nan
    verbose('Enriched price averages on Close price!')

    if 'High' not in df or 'Low' not in df:
        return
    verbose('Enriching price averages on High and Low prices...')
    df['fibonacci_moving_avg_high'] = fibonacciMovingAverage(df['High'], fibonacci_seq_sz)
    df.loc[:fibonacci_seq_sz, 'fibonacci_moving_avg_high'] = np.nan
    df['fibonacci_weighted_moving_avg_high'] = fibonacciMovingAverage(df['High'], fibonacci_seq_sz, weighted=True)
    df.loc[:fibonacci_seq_sz, 'fibonacci_weighted_moving_avg_high'] = np.nan
    df['fibonacci_moving_avg_low'] = fibonacciMovingAverage(df['Low'], fibonacci_seq_sz)
    df.loc[:fibonacci_seq_sz, 'fibonacci_moving_avg_low'] = np.nan
    df['fibonacci_weighted_moving_avg_low'] = fibonacciMovingAverage(df['Low'], fibonacci_seq_sz, weighted=True)
    df.loc[:fibonacci_seq_sz, 'fibonacci_weighted_moving_avg_low'] = np.nan
    verbose('Enriched price averages on High and Low prices!')


def filterOutNullRows(df: pd.DataFrame) -> None:
    verbose('Dropping null rows...')
    df.dropna(inplace=True)
    verbose('Dropped null rows!')


def fibonacciMovingAverage(data: pd.Series, fibonacci_size=10, weighted=False) -> pd.Series:
    global _fibonacci_sequence
    if _fibonacci_sequence is None or len(_fibonacci_sequence) < fibonacci_size:
        _fibonacci_sequence = getFibonacciSeq(fibonacci_size + 1)[1:]  # span must be >=1
    emas = []
    for fibonacci in _fibonacci_sequence[:fibonacci_size]:
        emas.append(data.ewm(span=fibonacci).mean())
    if weighted:
        weight = _fibonacci_sequence[0]
        weights_sum = weight
        fibonacci_weighted_average = emas[0] * weight
        for i in range(1, len(emas), 1):
            weight = _fibonacci_sequence[i]
            weights_sum += weight
            fibonacci_weighted_average += emas[i] * weight
        return fibonacci_weighted_average / weights_sum
    else:
        fibonacci_average = emas[0]
        for i in range(1, len(emas), 1):
            fibonacci_average += emas[i]
        return fibonacci_average / len(emas)


def enrichTALib(df: pd.DataFrame) -> None:
    # https://mrjbq7.github.io/ta-lib/funcs.html
    with warnings.catch_warnings():  # dont care about performance now, I just want to make easier to understand
        warnings.simplefilter('ignore', category=pd.errors.PerformanceWarning)
        if 'Close' not in df:
            return
        verbose('Enriching price TA-Lib on Close price...')
        close = df['Close']

        df['BBANDS-upperband'], df['BBANDS-middleband'], df['BBANDS-lowerband'] = talib.BBANDS(close)
        df['DEMA'] = talib.DEMA(close)
        df['EMA'] = talib.EMA(close)
        df['HT_TRENDLINE'] = talib.HT_TRENDLINE(close)
        df['KAMA'] = talib.KAMA(close)
        df['MA'] = talib.MA(close)  # I know that some averages are repeated, however they have different window sizes
        df['MAMA-mama'], df['MAMA-fama'] = talib.MAMA(close)
        df['MAVP'] = talib.MAVP(close, df['Date'].apply(dateStrToTimestamp, input_format=SAVE_DATE_FORMAT))
        df['MIDPOINT'] = talib.MIDPOINT(close)
        df['SMA'] = talib.SMA(close)
        df['T3'] = talib.T3(close)
        df['TEMA'] = talib.TEMA(close)
        df['TRIMA'] = talib.TRIMA(close)
        df['WMA'] = talib.WMA(close)
        df['APO'] = talib.APO(close)
        df['CMO'] = talib.CMO(close)
        df['MACD-macd'], df['MACD-macdsignal'], df['MACD-macdhist'] = talib.MACD(close)
        df['MACDEXT-macd'], df['MACDEXT-macdsignal'], df['MACDEXT-macdhist'] = talib.MACDEXT(close)
        df['MACDFIX-macd'], df['MACDFIX-macdsignal'], df['MACDFIX-macdhist'] = talib.MACDFIX(close)
        df['MOM'] = talib.MOM(close)
        df['PPO'] = talib.PPO(close)
        df['ROC'] = talib.ROC(close)
        df['ROCP'] = talib.ROCP(close)
        df['ROCR'] = talib.ROCR(close)
        df['ROCR100'] = talib.ROCR100(close)
        df['RSI'] = talib.RSI(close)
        df['STOCHRSI-fastk'], df['STOCHRSI-fastd'] = talib.STOCHRSI(close)
        df['TRIX'] = talib.TRIX(close)
        df['HT_DCPERIOD'] = talib.HT_DCPERIOD(close)
        df['HT_DCPHASE'] = talib.HT_DCPHASE(close)
        df['HT_PHASOR-inphase'], df['HT_PHASOR-quadrature'] = talib.HT_PHASOR(close)
        df['HT_DCPHASE-sine'], df['HT_DCPHASE-leadsine'] = talib.HT_SINE(close)
        df['HT_TRENDMODE'] = talib.HT_TRENDMODE(close)
        df['LINEARREG'] = talib.LINEARREG(close)
        df['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(close)
        df['LINEARREG_INTERCEPT'] = talib.LINEARREG_INTERCEPT(close)
        df['LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(close)
        df['STDDEV'] = talib.STDDEV(close)
        df['TSF'] = talib.TSF(close)
        df['VAR'] = talib.VAR(close)

        if 'Volume' in df:
            df['OBV'] = talib.OBV(close, df['Volume'])
        verbose('Enriched price TA-Lib on Close price!')

        if 'High' not in df or 'Low' not in df:
            return
        verbose('Enriching price TA-Lib on High and Low prices...')
        high = df['High']
        low = df['Low']

        df['MIDPRICE'] = talib.MIDPRICE(high, low)
        df['SAR'] = talib.SAR(high, low)
        df['SAREXT'] = talib.SAREXT(high, low)
        df['AROON-aroondown'], df['AROON-aroonup'] = talib.AROON(high, low)
        df['AROONOSC'] = talib.AROONOSC(high, low)
        df['MINUS_DM'] = talib.MINUS_DM(high, low)
        df['PLUS_DM'] = talib.PLUS_DM(high, low)
        df['MEDPRICE'] = talib.MEDPRICE(high, low)

        df['ADX'] = talib.ADX(high, low, close)
        df['ADXR'] = talib.ADXR(high, low, close)
        df['CCI'] = talib.CCI(high, low, close)
        df['DX'] = talib.DX(high, low, close)
        df['MINUS_DI'] = talib.MINUS_DI(high, low, close)
        df['PLUS_DI'] = talib.PLUS_DI(high, low, close)
        df['STOCH-slowk'], df['STOCH-slowd'] = talib.STOCH(high, low, close)
        df['ULTOSC'] = talib.ULTOSC(high, low, close)
        df['WILLR'] = talib.WILLR(high, low, close)
        df['ATR'] = talib.ATR(high, low, close)
        df['NATR'] = talib.NATR(high, low, close)
        df['TRANGE'] = talib.TRANGE(high, low, close)
        df['TYPPRICE'] = talib.TYPPRICE(high, low, close)
        df['WCLPRICE'] = talib.WCLPRICE(high, low, close)
        df['BETA'] = talib.BETA(high, low)
        df['CORREL'] = talib.CORREL(high, low)

        if 'Volume' in df:
            df['MFI'] = talib.MFI(high, low, close, df['Volume'])
            df['AD'] = talib.AD(high, low, close, df['Volume'])
            df['ADOSC'] = talib.ADOSC(high, low, close, df['Volume'])
        verbose('Enriched price TA-Lib on High and Low prices!')

        if 'Open' not in df:
            return
        verbose('Enriching price TA-Lib on Open price...')
        open_p = df['Open']

        df['BOP'] = talib.BOP(open_p, high, low, close)
        df['AVGPRICE'] = talib.AVGPRICE(open_p, high, low, close)
        df['CDL2CROWS'] = talib.CDL2CROWS(open_p, high, low, close)
        df['CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS(open_p, high, low, close)
        df['CDL3INSIDE'] = talib.CDL3INSIDE(open_p, high, low, close)
        df['CDL3LINESTRIKE'] = talib.CDL3LINESTRIKE(open_p, high, low, close)
        df['CDL3OUTSIDE'] = talib.CDL3OUTSIDE(open_p, high, low, close)
        df['CDL3STARSINSOUTH'] = talib.CDL3STARSINSOUTH(open_p, high, low, close)
        df['CDL3WHITESOLDIERS'] = talib.CDL3WHITESOLDIERS(open_p, high, low, close)
        df['CDLABANDONEDBABY'] = talib.CDLABANDONEDBABY(open_p, high, low, close)
        df['CDLADVANCEBLOCK'] = talib.CDLADVANCEBLOCK(open_p, high, low, close)
        df['CDLBELTHOLD'] = talib.CDLBELTHOLD(open_p, high, low, close)
        df['CDLBREAKAWAY'] = talib.CDLBREAKAWAY(open_p, high, low, close)
        df['CDLCLOSINGMARUBOZU'] = talib.CDLCLOSINGMARUBOZU(open_p, high, low, close)
        df['CDLCONCEALBABYSWALL'] = talib.CDLCONCEALBABYSWALL(open_p, high, low, close)
        df['CDLCOUNTERATTACK'] = talib.CDLCOUNTERATTACK(open_p, high, low, close)
        df['CDLDARKCLOUDCOVER'] = talib.CDLDARKCLOUDCOVER(open_p, high, low, close)
        df['CDLDOJI'] = talib.CDLDOJI(open_p, high, low, close)
        df['CDLDOJISTAR'] = talib.CDLDOJISTAR(open_p, high, low, close)
        df['CDLDRAGONFLYDOJI'] = talib.CDLDRAGONFLYDOJI(open_p, high, low, close)
        df['CDLENGULFING'] = talib.CDLENGULFING(open_p, high, low, close)
        df['CDLEVENINGDOJISTAR'] = talib.CDLEVENINGDOJISTAR(open_p, high, low, close)
        df['CDLEVENINGSTAR'] = talib.CDLEVENINGSTAR(open_p, high, low, close)
        df['CDLGAPSIDESIDEWHITE'] = talib.CDLGAPSIDESIDEWHITE(open_p, high, low, close)
        df['CDLGRAVESTONEDOJI'] = talib.CDLGRAVESTONEDOJI(open_p, high, low, close)
        df['CDLHAMMER'] = talib.CDLHAMMER(open_p, high, low, close)
        df['CDLHANGINGMAN'] = talib.CDLHANGINGMAN(open_p, high, low, close)
        df['CDLHARAMI'] = talib.CDLHARAMI(open_p, high, low, close)
        df['CDLHARAMICROSS'] = talib.CDLHARAMICROSS(open_p, high, low, close)
        df['CDLHIGHWAVE'] = talib.CDLHIGHWAVE(open_p, high, low, close)
        df['CDLHIKKAKE'] = talib.CDLHIKKAKE(open_p, high, low, close)
        df['CDLHIKKAKEMOD'] = talib.CDLHIKKAKEMOD(open_p, high, low, close)
        df['CDLHOMINGPIGEON'] = talib.CDLHOMINGPIGEON(open_p, high, low, close)
        df['CDLIDENTICAL3CROWS'] = talib.CDLIDENTICAL3CROWS(open_p, high, low, close)
        df['CDLINNECK'] = talib.CDLINNECK(open_p, high, low, close)
        df['CDLINVERTEDHAMMER'] = talib.CDLINVERTEDHAMMER(open_p, high, low, close)
        df['CDLKICKING'] = talib.CDLKICKING(open_p, high, low, close)
        df['CDLKICKINGBYLENGTH'] = talib.CDLKICKINGBYLENGTH(open_p, high, low, close)
        df['CDLLADDERBOTTOM'] = talib.CDLLADDERBOTTOM(open_p, high, low, close)
        df['CDLLONGLEGGEDDOJI'] = talib.CDLLONGLEGGEDDOJI(open_p, high, low, close)
        df['CDLLONGLINE'] = talib.CDLLONGLINE(open_p, high, low, close)
        df['CDLMARUBOZU'] = talib.CDLMARUBOZU(open_p, high, low, close)
        df['CDLMATCHINGLOW'] = talib.CDLMATCHINGLOW(open_p, high, low, close)
        df['CDLMATHOLD'] = talib.CDLMATHOLD(open_p, high, low, close)
        df['CDLMORNINGDOJISTAR'] = talib.CDLMORNINGDOJISTAR(open_p, high, low, close)
        df['CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(open_p, high, low, close)
        df['CDLONNECK'] = talib.CDLONNECK(open_p, high, low, close)
        df['CDLPIERCING'] = talib.CDLPIERCING(open_p, high, low, close)
        df['CDLRICKSHAWMAN'] = talib.CDLRICKSHAWMAN(open_p, high, low, close)
        df['CDLRISEFALL3METHODS'] = talib.CDLRISEFALL3METHODS(open_p, high, low, close)
        df['CDLSEPARATINGLINES'] = talib.CDLSEPARATINGLINES(open_p, high, low, close)
        df['CDLSHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(open_p, high, low, close)
        df['CDLSHORTLINE'] = talib.CDLSHORTLINE(open_p, high, low, close)
        df['CDLSPINNINGTOP'] = talib.CDLSPINNINGTOP(open_p, high, low, close)
        df['CDLSTALLEDPATTERN'] = talib.CDLSTALLEDPATTERN(open_p, high, low, close)
        df['CDLSTICKSANDWICH'] = talib.CDLSTICKSANDWICH(open_p, high, low, close)
        df['CDLTAKURI'] = talib.CDLTAKURI(open_p, high, low, close)
        df['CDLTASUKIGAP'] = talib.CDLTASUKIGAP(open_p, high, low, close)
        df['CDLTHRUSTING'] = talib.CDLTHRUSTING(open_p, high, low, close)
        df['CDLTRISTAR'] = talib.CDLTRISTAR(open_p, high, low, close)
        df['CDLUNIQUE3RIVER'] = talib.CDLUNIQUE3RIVER(open_p, high, low, close)
        df['CDLUPSIDEGAP2CROWS'] = talib.CDLUPSIDEGAP2CROWS(open_p, high, low, close)
        df['CDLXSIDEGAP3METHODS'] = talib.CDLXSIDEGAP3METHODS(open_p, high, low, close)
        verbose('Enriched price TA-Lib on Open price!')
