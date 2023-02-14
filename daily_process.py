import pandas as pd
import numpy as np
import os

from datetime import date, timedelta
from constants import TICKERS

from models.lin_reg_model import SplitOFIModel, OFIModel

START_DATE = date.fromisoformat('2019-02-01')
PATH = './results/h30/'
TKR = TICKERS[:100]


def run_models(name: str, d: date, df_train, df_test):
    print(name)
    path = os.path.join(PATH, str(d), name)
    if not os.path.exists(path):
        os.mkdir(path)
    model_list = []
    for i in range(1, 11):
        model_list.append(SplitOFIModel(train_df=df_train, levels=i, return_type='current', test_df=df_test))
        model_list.append(SplitOFIModel(train_df=df_train, levels=i, return_type='future', test_df=df_test))
        model_list.append(OFIModel(train_df=df_train, levels=i, return_type='current', test_df=df_test))
        model_list.append(OFIModel(train_df=df_train, levels=i, return_type='future', test_df=df_test))

    for model in model_list:
        model.fit()
        model.score_test()
        summary = model.summary()
        file_path = os.path.join(path, model.name + '.summary')
        file = open(file_path, 'w')
        print(summary, file=file)
        file.close()


def main():
    df = pd.read_csv('./lobster_sample/combined_30m_rol10m.csv')
    df['date'] = np.array(list(map(date.fromisoformat, df['date'])))
    dates = np.unique(df['date'].to_numpy())
    dates = dates[dates >= START_DATE]
    for d in dates:
        print(f'Processing date {d}...')
        folder = os.path.join(PATH, str(d))
        if not os.path.exists(folder):
            os.mkdir(folder)

        past_d = d - timedelta(days=30)
        df_train = df[(df['date'] >= past_d) & (df['date'] < d)]
        df_test = df[df['date'] == d]
        run_models('universal', d, df_train, df_test)

        for ticker in TKR:
            if ticker == 'TFC':
                continue
            df_train_ticker = df_train[df_train['ticker'] == ticker]
            df_test_ticker = df_test[df_test['ticker'] == ticker]
            run_models(ticker, d, df_train_ticker, df_test_ticker)


if __name__ == '__main__':
    main()
