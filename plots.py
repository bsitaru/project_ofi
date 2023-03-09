import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from summary_parser import Summary

def main():
    df = pd.read_csv('./results/h30/results.csv')
    ticker = 'universal'
    model = 'SplitOFI_10_future'
    df = df[(df['ticker'] == ticker) & (df['name'] == model)]
    ax = df[['date', 'adj_r2', 'oos_r2']].set_index('date').plot()
    ax.set(ylim=(-1., 1.))
    plt.show()
    # print(data)
    # ax = sns.lineplot(data)
    # ax.set(ylim=(-1., 1.))
    # plt.show()

def color_histogram():
    model_name = 'SplitOFI_10_current'
    dates = sorted(list(filter(lambda x: x.startswith('2'), os.listdir('./results/h1'))))

    dct = {}
    for d in dates:
        s = Summary(f'./results/h1/{d}/universal/{model_name}.summary', model_name)
        v = s.vars

        for k in v:
            if k['name'] not in dct:
                dct[k['name']] = []
            dct[k['name']].append(k['coef'])

    for x in dct.keys():
        m = 0
        n = 0
        for i in dct[x]:
            n += 1
            m += i
        m /= float(n)
        print(x, m)

    # print(dct['const'])
    # data = [dct[x] for x in dct.keys()]
    # plt.hist(data, label=list(dct.keys()))
    # plt.legend()
    # plt.show()

    # print(dct)


if __name__ == '__main__':
    # main()
    color_histogram()
