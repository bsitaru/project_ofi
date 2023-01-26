"""
Generate OFI  as Alpha and Return as Target
"""

import pandas as pd
import numpy as np
import os
from os.path import *
from functools import partial
from multiprocessing import cpu_count, Pool


def OFI_raw(subdf):
    return subdf['OFI_1'].sum()


def OFI_scale_Vol(subdf, nlevels):
    return pd.DataFrame([subdf['OFI_%d' % lev].sum() / subdf['Volume_%d' % lev].mean() for lev in range(1, 1+nlevels)]).T


def Ret(subdf):
    start_price = subdf['StartPriceInBucket'].tolist()[0]
    end_price = subdf['EndPriceInBucket'].tolist()[-1]
    if start_price <= 0:
        ret = np.nan
    else:
        ret = np.log(end_price / start_price)
    return ret


def Split_DF(df, seconds, nlevels):
    subdf_group = df.groupby(df.index // seconds)
    rets = subdf_group.apply(Ret) * 100
    OFI_V = subdf_group.apply(OFI_scale_Vol, nlevels)
    OFI_V.index = rets.index
    OFI_R = subdf_group.apply(OFI_raw)

    sumdf = pd.concat([rets, OFI_R, OFI_V], axis=1)
    sumdf.columns=['Ret', 'OFI_R'] + ['ofi_%d' % lev for lev in range(1, 1+nlevels)]
    return sumdf


def Load_Data(folderpath, ticker, date, tscale, seconds, nlevels):
    folderdatepath = join(folderpath, '_'.join([ticker, date, str(nlevels)]))
    saveresultpathfile = join(folderdatepath, '_'.join([ticker, date, 'ts%d' % tscale, 'PriceOFI', str(nlevels)]) + '.csv')
    if isfile(saveresultpathfile):
        df = pd.read_csv(saveresultpathfile, index_col=0)
        df = df.loc[34200:57599]
        df.index -= 34200
        sumdf = Split_DF(df, seconds, nlevels)
        # sumdf = sumdf.to_frame()
        # sumdf.columns = ['OFI_%d' % ]
        # sumdf.dropna(inplace=True)
        sumdf.fillna(0, inplace=True)
        sumdf = sumdf[:int(390 * 60 / seconds)]

        missing_index = list(set(range(int(390 * 60 / seconds))) - set(sumdf.index))
        for idx in missing_index:
            sumdf.loc[idx] = 0

        sumdf.sort_index(inplace=True)

        if seconds == 60:
            sumdf['Time'] = pd.date_range('%s 09:30' % date, '%s 16:00' % date, freq='1min', closed='right')
            sumdf['Time'] = sumdf['Time'].apply(lambda x: x.strftime('%H:%M'))
        elif seconds == 10:
            sumdf['Time'] = pd.date_range('%s 09:30:00' % date, '%s 16:00:00' % date, freq='10s', closed='right')
            sumdf['Time'] = sumdf['Time'].apply(lambda x: x.strftime('%H:%M:%S'))
        return sumdf
    else:
        return None


def Save_Alpha_OFI(folderpath, ticker, date, tscale, seconds, nlevels, alphapath):
    sumdf = Load_Data(folderpath, ticker, date, tscale, seconds, nlevels)

    if sumdf is not None:
        alpham_aver_datepath = join(alphapath, '%dSeconds' % seconds, 'OFI_1-%d' % nlevels, date)
        os.makedirs(alpham_aver_datepath, exist_ok=True)
        sumdf.to_csv(join(alpham_aver_datepath, ticker + '.csv'))


def OFI_Ticker(ticker, data_path, tscale, seconds, year_l, nlevels, alphapath):
    folderpath = join(data_path, ticker)
    dir_l = os.listdir(folderpath)
    date_l = [date.split('_')[1] for date in dir_l for year in year_l
              if date.startswith('%s_%d' % (ticker, year)) and isdir(join(folderpath, date))]
    date_l.sort()
    for i, date in enumerate(date_l):
        print("* - " * 5 + ticker + ' || ' + date + ' || ' + " - *" * 5)
        Save_Alpha_OFI(folderpath=folderpath,
                       ticker=ticker,
                       date=date,
                       tscale=tscale,
                       seconds=seconds,
                       nlevels=nlevels,
                       alphapath=alphapath)


def OFI_Ticker_list(sub_ticker_l, data_path, tscale, seconds, year_l, nlevels, alphapath):
    for ticker in sub_ticker_l:
        OFI_Ticker(ticker=ticker,
                   data_path=data_path,
                   tscale=tscale,
                   seconds=seconds,
                   year_l=year_l,
                   nlevels=nlevels,
                   alphapath=alphapath)


# parallel process
def parallelize(ticker_l, data_path, tscale, seconds, year_l, nlevels, alphapath):
    cores = cpu_count()  # Number of CPU cores on your system
    partitions = 20 #cores - 2  # Define as many partitions as you want
    data_split = np.array_split(ticker_l, partitions)
    pool = Pool(partitions)
    partial_func = partial(OFI_Ticker_list,
                           data_path=data_path,
                           tscale=tscale,
                           seconds=seconds,
                           year_l=year_l,
                           nlevels=nlevels,
                           alphapath=alphapath)
    pool.map(partial_func, data_split)
    pool.close()
    pool.join()


if __name__ == '__main__':
    data_path = '/data/localhost/not-backed-up/scratch/chzhang/LOB/LOBData/'
    nlevels = 10
    tscale = 1
    seconds = 10
    year_l = [2017, 2018, 2019]
    alphapath = '/data/localhost/not-backed-up/scratch/chzhang/LOB/Alphas/'

    # For all tickers, execute the complete procedure
    ticker_l = os.listdir(data_path)
    ticker_l = [ticker for ticker in ticker_l if isdir(join(data_path, ticker))]
    ticker_l.sort()
    parallelize(ticker_l=ticker_l,
                data_path=data_path,
                tscale=tscale,
                seconds=seconds,
                year_l=year_l,
                nlevels=nlevels,
                alphapath=alphapath)
