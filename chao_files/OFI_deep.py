"""
Computation of OFI on various levels
"""

import os
import csv
import numpy as np
from os.path import *
import pandas as pd
import time
from functools import partial
from multiprocessing import cpu_count, Pool


missing_value = 9999999999
year = 2016


def Find_Non_OFI(basepath):
    folders_l = os.listdir(basepath)
    folders_l = [ticker for ticker in folders_l if isdir(join(basepath, ticker))]
    folders_l.sort()
    ticker_l = []
    for ticker in folders_l:
        ticker_path = join(basepath, ticker)
        ticker_date_l = os.listdir(ticker_path)
        ticker_date_l = [ticker_date for ticker_date in ticker_date_l if isdir(join(ticker_path, ticker_date)) and str(year) in ticker_date]
        ticker_date_l.sort()
        if len(ticker_date_l) == 0:
            # os.system("rm -rf %s" % ticker_path)
            pass
        else:
            ticker_date = ticker_date_l[-1]
            ticker_date_path = join(ticker_path, ticker_date)
            ticker_date_files_l = os.listdir(ticker_date_path)
            ticker_date_files_l.sort()
            idx = False
            for f in ticker_date_files_l:
                if 'PriceOFI' in f:
                    idx = True
            if not idx:
                ticker_l.append(ticker)
    return ticker_l


def Find_Non_OFI_Date(ticker, folderpath):
    dir_l = os.listdir(folderpath)
    tmp_date_l = [date for date in dir_l if
                  date.startswith('%s_%s' % (ticker, year)) and isdir(join(folderpath, date))]
    tmp_date_l.sort()
    date_l = []
    for date in tmp_date_l:
        ticker_date_path = join(folderpath, date)
        ticker_date_files_l = os.listdir(ticker_date_path)
        ticker_date_files_l.sort()
        idx = False
        for f in ticker_date_files_l:
            if 'PriceOFI' in f:
                idx = True
        if not idx:
            date_l.append(date.split('_')[1])
    return date_l


def Compute_OFI(two_lines_l, level):
    idx = 4 * (level - 1)
    Pa_n1, qa_n1, Pb_n1, qb_n1 = map(float, two_lines_l[0][idx:idx+4])
    Pa_n, qa_n, Pb_n, qb_n = map(float, two_lines_l[1][idx:idx+4])

    return qb_n * (Pb_n >= Pb_n1) - qb_n1 * (Pb_n <= Pb_n1) - qa_n * (Pa_n <= Pa_n1) + qa_n1 * (Pa_n >= Pa_n1)


def Compute_Volume(two_lines_l, level):
    idx = 4 * (level - 1)
    _, qa_n, _, qb_n = map(float, two_lines_l[1][idx:idx+4])
    return (qa_n + qb_n) / 2.


def Compute_Depth(two_lines_l, level):
    idx = 4 * (level - 1)
    Pa_n1, qa_n1, Pb_n1, qb_n1 = map(float, two_lines_l[0][idx:idx+4])
    Pa_n, qa_n, Pb_n, qb_n = map(float, two_lines_l[1][idx:idx+4])
    numerator1 = qb_n * (Pb_n < Pb_n1) + qb_n1 * (Pb_n > Pb_n1)
    denominator1 = (Pb_n != Pb_n1)
    numerator2 = qa_n * (Pa_n > Pa_n1) + qa_n1 * (Pa_n < Pa_n1)
    denominator2 = (Pa_n != Pa_n1)
    return [numerator1, denominator1, numerator2, denominator2]


def Compute_Price(two_lines_l):
    Pa_n = float(two_lines_l[1][0])
    Pb_n = float(two_lines_l[1][2])
    return (Pb_n + Pa_n) / 20000.


def P_OFI_V(two_lines_l, nlevels):
    if len(two_lines_l) != 2:
        return [None] * (1 + nlevels)
    else:
        # OFI
        OFI_l = [Compute_OFI(two_lines_l, level=i) for i in range(1, 1+nlevels)]

        # Volume
        Volume_l = [Compute_Volume(two_lines_l, level=i) for i in range(1, 1+nlevels)]

        # Depth only for the first level
        Depth_l = Compute_Depth(two_lines_l, level=1)

        # Price Change
        Price = Compute_Price(two_lines_l)

        all_l = [Price] + OFI_l + Volume_l + Depth_l
        return all_l


def Price_OFI_All(folderpath, ticker, date, saveOFIpath=None, nlevels=200):
    """
    Computation of OFI and Price Changes
    -------
    :param folderpath: Folder path
    :param ticker: TICKER
    :param date: "%Y-%m-%d"
    :param saveOFIpath: Folder path to save Price_OFI file
    :param nlevels: the number of levels to be analyzed
    :return: Dataframe containing Delta Price and OFI for each order change
    """
    # Name of Orderbook
    folder_date = join(folderpath, '_'.join([ticker, date, str(nlevels)]))
    files_l = os.listdir(folder_date)
    # ORDERBOOK = join(folder_date, '_'.join([ticker, date, str(starttime), str(endtime), 'orderbook', str(nlevels)]) + '.csv')
    ORDERBOOK = [s for s in files_l if 'orderbook' in s][0]
    ORDERBOOK = join(folder_date, ORDERBOOK)

    # Record Price Change and OFI
    PriceOFI_l = []
    PriceOFIappend = PriceOFI_l.append
    start_time = time.time()

    with open(ORDERBOOK, "r") as csvfile:
        datareader = csv.reader(csvfile)
        first_line = next(datareader)
        i = 0
        two_lines_l = [first_line]
        for line in datareader:
            i += 1
            two_lines_l.append(line)
            PriceOFIappend(P_OFI_V(two_lines_l, nlevels))
            two_lines_l = [line]

    cols = ['Price'] + ['OFI_%d' % k for k in range(1, nlevels+1)] + ['Volume_%d' % k for k in range(1, nlevels+1)]
    cols += ['Depth_%d' % k for k in range(1, 5)]
    PriceOFI_df = pd.DataFrame(data=np.array(PriceOFI_l),
                               columns=cols,
                               index=range(1, 1+len(PriceOFI_l)))

    if saveOFIpath is None:
        saveOFIpath = folder_date

    # Save This Dataframe, which is the essential file for subsequent processing
    saveOFIpathfile = join(saveOFIpath, '_'.join([ticker, date, 'PriceOFI', str(nlevels)]) + '.csv')

    PriceOFI_df.to_csv(saveOFIpathfile, index=False)
    
    # Free Space
    # os.system('rm %s' % ORDERBOOK)


def Price_OFI_All_list(sub_date_l, folderpath, ticker, saveOFIpath=None, nlevels=200):
    for date in sub_date_l:
        print(" *" * 5 + " || " + date + " || " + "* " * 5)
        Price_OFI_All(folderpath=folderpath,
                      ticker=ticker,
                      date=date,
                      saveOFIpath=saveOFIpath,
                      nlevels=nlevels)


# parallel process
def parallelize(folderpath, ticker, saveOFIpath=None, nlevels=10):
    cores = cpu_count()                 # Number of CPU cores on your system
    partitions = 1 #cores - 2              # Define as many partitions as you want
    date_l = Find_Non_OFI_Date(ticker, folderpath)
    data_split = np.array_split(date_l, partitions)
    pool = Pool(partitions)
    partial_func = partial(Price_OFI_All_list,
                           folderpath=folderpath,
                           ticker=ticker,
                           saveOFIpath=saveOFIpath,
                           nlevels=nlevels)
    pool.map(partial_func, data_split)
    pool.close()
    pool.join()


if __name__ == "__main__":
    nlevels = 10
    # basepath = '/data/localhost/not-backed-up/scratch/chzhang/LOB/LOBData/'
    basepath = '../lobster_sample/tickers/'

    ticker_l = Find_Non_OFI(basepath)
    # ticker_l = os.listdird(basepath)
    # ticker_l = [ticker for ticker in ticker_l if isdir(join(basepath, ticker))]
    # ticker_l.sort()
    for ticker in ticker_l:
        print("+ - " * 10 + ticker + " - +" * 10)
        folderpath = join(basepath, ticker)
        parallelize(folderpath, ticker, saveOFIpath=None, nlevels=nlevels)