"""
Generate the dataframe for a specific time bucket
"""

import csv
import numpy as np
from os.path import *
import pandas as pd
import time
import os
from functools import partial
from multiprocessing import cpu_count, Pool

year = 2017


def Find_Non_TSOFI(basepath):
    folders_l = os.listdir(basepath)
    folders_l = [ticker for ticker in folders_l if isdir(join(basepath, ticker))]
    folders_l.sort()
    ticker_l = []
    for ticker in folders_l:
        ticker_path = join(basepath, ticker)
        ticker_date_l = os.listdir(ticker_path)
        ticker_date_l = [ticker_date for ticker_date in ticker_date_l if isdir(join(ticker_path, ticker_date)) and str(year) in ticker_date]
        ticker_date_l.sort()
        ticker_date = ticker_date_l[-1]
        ticker_date_path = join(ticker_path, ticker_date)
        ticker_date_files_l = os.listdir(ticker_date_path)
        ticker_date_files_l.sort()
        idx = False
        for f in ticker_date_files_l:
            if 'ts1_PriceOFI' in f:
                idx = True
        if not idx:
            ticker_l.append(ticker)
    return ticker_l


def Find_Non_TSOFI_Date(ticker, folderpath):
    dir_l = os.listdir(folderpath)
    tmp_date_l = [date for date in dir_l if
                  date.startswith('%s_%d' % (ticker, year)) and isdir(join(folderpath, date))]
    tmp_date_l.sort()
    date_l = []
    for date in tmp_date_l:
        ticker_date_path = join(folderpath, date)
        ticker_date_files_l = os.listdir(ticker_date_path)
        ticker_date_files_l.sort()
        idx = False
        for f in ticker_date_files_l:
            if 'ts1_PriceOFI' in f:
                idx = True
        if not idx:
            date_l.append(date.split('_')[1])
    return date_l


def myfloat(s):
    return float(s) if s != '' else np.nan


def find_now_time(t, bound, msgline, tscale):
    for kk in range(t+1, len(bound)):
        if bound[kk] < float(msgline[0]) <= (bound[kk] + tscale):
            return kk
            break


def TimeScale(folderpath, ticker, date, tscale=None, saveOFIpathfile=None,
                saveresultpath=None, nlevels=None):
    """
    OFI and Price at given intervals
    ------
    :param folderpath: Folder path
    :param ticker: TICKER
    :param date: "%Y-%m-%d"
    :param tscale: time intervals in seconds
    :param saveOFIpathfile: Folder path saved Price_OFI file
    :param saveresultpath: Folder path to save result file
    :param nlevels: the number of levels to be analyzed
    :return: DataFrame
    """
    start_time = time.time()

    # Name of MSGbook
    folder_date = join(folderpath, '_'.join([ticker, date, str(nlevels)]))
    # MSGBOOK = join(folder_date, '_'.join([ticker, date, str(starttime), str(endtime), 'message', str(nlevels)]) + '.csv')
    files_l = os.listdir(folder_date)
    MSGBOOK = [s for s in files_l if 'message' in s][0]
    starttime = float(MSGBOOK.split('_')[2])
    endtime = float(MSGBOOK.split('_')[3])
    MSGBOOK = join(folder_date, MSGBOOK)

    # Load Delta Price and OFI
    if saveOFIpathfile is None:
        saveOFIpathfile = join(folder_date, '_'.join([ticker, date, 'PriceOFI', str(nlevels)]) + '.csv')

    # Trading hours (start & end)
    startTrad = starttime / 1000.  # 9:30:00.000 in ms after midnight
    endTrad = endtime / 1000.  # 16:00:00.000 in ms after midnight

    # For each tscale seconds, Run Linear Models
    bound = np.arange(int(startTrad), int(endTrad), tscale)

    Price_OFI_l = []
    with open(MSGBOOK, 'r') as msgfile, open(saveOFIpathfile, 'r') as ofifile:
        msgdatareader = csv.reader(msgfile)
        # Skip the first line in Messgae file because in the computation process of OFI,
        # the first OFI uses the first two lines info in order book file.
        # So the second line in message file corresponds to the first observation in OFI file.
        msg_first_line = next(msgdatareader)
        # Skip the first line in OFI as it is column names.
        ofidatareader = csv.reader(ofifile)
        ofi_first_line = next(ofidatareader)

        i = 0  # Record the number of lines which has been scanned
        t = 0  # Record Time
        t_l = []
        tmp_l = []
        for msgline, ofiline in zip(msgdatareader, ofidatareader):
            i += 1
            if bound[t] < float(msgline[0]) <= (bound[t] + tscale):
                tmp_l.append(list(map(myfloat, ofiline)))
            elif float(msgline[0]) > endTrad:
                break
            else:
                if t % 1000 == 0:
                    print('         %d / %d at %.4f : %.2f' % (t, len(bound), float(msgline[0]), time.time() - start_time))
                    start_time = time.time()
                if len(tmp_l) == 0:
                    pass
                else:
                    cols = ['Price'] + ['OFI_%d' % k for k in range(1, nlevels + 1)] + ['Volume_%d' % k for k in
                                                                                        range(1, nlevels + 1)]
                    cols += ['Depth_%d' % k for k in range(1, 5)]

                    # start, end price in the bucket
                    start_price_in_bucket = myfloat(tmp_l[0][0])
                    end_price_in_bucket = myfloat(tmp_l[-1][0])
                    # Using the first next price after the current bucket
                    next_price = myfloat(ofiline[0])

                    # ofi on different levels
                    array_ofi = np.array(tmp_l)[:, 1:(1+nlevels)]
                    ofi_sum_l = np.nansum(array_ofi, axis=0).tolist()

                    # volume on different levels
                    array_vol = np.array(tmp_l)[:, (1+nlevels):(1+2*nlevels)]
                    vol_mean_l = np.nanmean(array_vol, axis=0).tolist()

                    # depth on the first level
                    array_depth = np.array(tmp_l)[:, (1+2*nlevels):(5+2*nlevels)]
                    depth_sum_l = np.nansum(array_depth, axis=0).tolist()

                    tmp_P_OFI_V_D = [start_price_in_bucket, end_price_in_bucket, next_price] + ofi_sum_l + vol_mean_l + depth_sum_l
                    Price_OFI_l.append(tmp_P_OFI_V_D)
                    t_l.append(bound[t])

                t = find_now_time(t, bound, msgline, tscale)
                tmp_l = []
                tmp_l.append(list(map(myfloat, ofiline)))

        # Last Period
        if len(tmp_l) == 0:
            pass
        else:
            # Using last price
            start_price_in_bucket = myfloat(tmp_l[0][0])
            end_price_in_bucket = myfloat(tmp_l[-1][0])
            last_price = myfloat(ofiline[0])

            # ofi on different levels
            array_ofi = np.array(tmp_l)[:, 1:(1 + nlevels)]
            ofi_sum_l = np.nansum(array_ofi, axis=0).tolist()

            # volume on different levels
            array_vol = np.array(tmp_l)[:, (1 + nlevels):(1 + 2 * nlevels)]
            vol_mean_l = np.nanmean(array_vol, axis=0).tolist()

            # depth on the first level
            array_depth = np.array(tmp_l)[:, (1 + 2 * nlevels):(5 + 2 * nlevels)]
            depth_sum_l = np.nansum(array_depth, axis=0).tolist()

            tmp_P_OFI_V_D = [start_price_in_bucket, end_price_in_bucket,
                             last_price] + ofi_sum_l + vol_mean_l + depth_sum_l

            Price_OFI_l.append(tmp_P_OFI_V_D)
            t_l.append(bound[t])

    cols = ['StartPriceInBucket', 'EndPriceInBucket', 'NextPrice'] + ['OFI_%d' % iii for iii in range(1, 1 + nlevels)]
    cols += ['Volume_%d' % iii for iii in range(1, nlevels+1)]
    cols += ['Depth_%d' % iii for iii in range(1, 5)]
    Price_OFI_df = pd.DataFrame(np.array(Price_OFI_l), columns=cols, index=t_l)

    if saveresultpath is None:
        saveresultpath = folder_date

    saveresultpathfile = join(saveresultpath, '_'.join([ticker, date, 'ts%d' % tscale, 'PriceOFI', str(nlevels)]) + '.csv')

    # Save
    Price_OFI_df.to_csv(saveresultpathfile)

    # Free Space
    os.system('rm %s' % MSGBOOK)


def TimeScale_list(sub_date_l, folderpath, ticker, tscale=None, saveOFIpathfile=None,
                saveresultpath=None, nlevels=None):
    for date in sub_date_l:
        print(" *" * 5 + " || " + date + " || " + "* " * 5)
        TimeScale(folderpath=folderpath,
                  ticker=ticker,
                  date=date,
                  tscale=tscale,
                  saveOFIpathfile=saveOFIpathfile,
                  saveresultpath=saveresultpath,
                  nlevels=nlevels)


# parallel process
def parallelize(folderpath, ticker, tscale=None, saveOFIpathfile=None,
                saveresultpath=None, nlevels=None):
    cores = cpu_count()                 # Number of CPU cores on your system
    partitions = 20 #cores - 2              # Define as many partitions as you want
    date_l = Find_Non_TSOFI_Date(ticker, folderpath)
    data_split = np.array_split(date_l, partitions)
    pool = Pool(partitions)
    partial_func = partial(TimeScale_list,
                           folderpath=folderpath,
                           ticker=ticker,
                           tscale=tscale,
                           saveOFIpathfile=saveOFIpathfile,
                           saveresultpath=saveresultpath,
                           nlevels=nlevels)
    pool.map(partial_func, data_split)
    pool.close()
    pool.join()


if __name__ == "__main__":
    basepath = '/data/localhost/not-backed-up/scratch/chzhang/LOB/LOBData/'
    tscale = 1
    nlevels = 10

    ticker_l = Find_Non_TSOFI(basepath)
    # ticker_l = os.listdir(basepath)
    # ticker_l = [ticker for ticker in ticker_l if isdir(join(basepath, ticker))]
    # ticker_l.sort()

    for ticker in ticker_l:
        print("* - " * 10 + ticker + " - *" * 10)
        folderpath = join(basepath, ticker)
        parallelize(folderpath=folderpath,
                    ticker=ticker,
                    tscale=tscale,
                    saveOFIpathfile=None,
                    saveresultpath=None,
                    nlevels=nlevels)
