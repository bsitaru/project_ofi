import os
import typer
from experiment import load_yaml
from ml_collections import ConfigDict

main = typer.Typer()


def get_is_os_from_log(log_path: str):
    r_is, r_os = None, None
    with open(log_path, 'r') as f:
        for line in f.readlines():
            l = line.split(' ')
            if l[3] == 'in_r2:':
                r_is = round(float(l[4]), 3)
            if l[3] == 'os_r2:':
                r_os = round(float(l[4]), 3)
    return r_is, r_os


def get_log_results(log_path: str):
    dict = {}
    dict_std = {}
    found = False
    with open(log_path, 'r') as f:
        for line in f.readlines():
            l = line.split(' ')
            if l[3] == 'in_r2:':
                found = True

            if found:
                key = l[3][:-1]

                try:
                    val = float(l[4].strip())
                    dict[key] = val
                except:
                    pass

                try:
                    std = float(l[7].strip())
                    dict_std[key] = std
                except:
                    pass
    return dict, dict_std


def get_results_from_folder(path: str):
    folders = os.listdir(path)
    folders = sorted(list(filter(lambda x: os.path.isdir(os.path.join(path, x)), folders)))
    lns = []
    for f in folders:
        f_path = os.path.join(path, f)
        args = ConfigDict(load_yaml(os.path.join(f_path, 'config.yaml')))
        dct, dct_std = get_log_results(os.path.join(f_path, 'logs.log'))

        levels = args.selector.levels
        regression = args.regression.type
        model = args.selector.type

        if model not in ['OFI', 'SplitOFI']:
            continue
        if args.experiment.name != 'clustered_price_impact':
            continue
        if args.processor.normalize!= False:
            continue

        bk = '\\'
        modpi_param = levels
        if 'pca' in args.processor:
            modpi_param = 'I'
            if model == 'SplitOFI':
                modpi_param += f', {args.processor.pca}'
        model_name = f"{bk}{'modpi' if model == 'OFI' else 'modpid'}{{{modpi_param}}}"
        fitting = 'OLS' if regression == 'linear' else 'Lasso'

        l = [dct['in_r2'], dct_std['in_r2'], dct['os_r2'], dct_std['os_r2']]
        l = list(map(lambda x: round(x * 100.0, 2), l))

        cls_data = 'OFI' if args.clustering.data == 'x' else 'Returns'

        table_line = f"${model_name}$ & {fitting} & {args.clustering.n_clusters} & K-Means & {cls_data} & {l[0]} & {l[1]} & {l[2]} & {l[3]} {bk}{bk}"
        lns.append(table_line)
    return lns


@main.command()
def table_maker(path: str):
    header = \
        """\\begin{table}[]
            \\centering
            \\begin{tabular}{lcccccc|ll}
            \\toprule
                Exp & Model & Levels & PCA & Regression & Cluster Size & Data & In Sample & Out of Sample \\\\
            \\midrule"""

    ending = \
        """\\bottomrule
            \\end{tabular}
            \\caption{Results}
            \\label{tab.results}
        \\end{table}
        """

    l = get_results_from_folder(path)
    answer = '\n'.join([header] + l + [ending])
    print(answer)


def get_portfolio_results_from_log(log_path: str):
    dict = {}
    dict_std = {}
    found = False
    with open(log_path, 'r') as f:
        for line in f.readlines():
            l = line.split(' ')
            if l[3] == 'overall':
                found = True
                continue

            if found:
                val = float(l[11])
                val = round(val, ndigits=4)
                return val
    return None


def get_future_results_from_folder(path: str):
    folders = os.listdir(path)
    folders = sorted(list(filter(lambda x: os.path.isdir(os.path.join(path, x)), folders)))
    lns = []
    for f in folders:
        f_path = os.path.join(path, f)
        args = ConfigDict(load_yaml(os.path.join(f_path, 'config.yaml')))
        r_is, r_os = get_is_os_from_log(os.path.join(f_path, 'logs.log'))
        pnl = get_portfolio_results_from_log(os.path.join(f_path, 'portfolio.log'))

        # name = 'Individual' if args.experiment.name == 'individual_price_impact' else 'Universal'
        name = args.experiment.name.split('_')[0]
        model = args.selector.type
        levels = args.selector.levels
        pca = '' if 'pca' not in args.processor else args.processor.pca
        if 'multipca' in args.processor:
            pca = f'multi-{args.processor.multipca.groups}-{args.processor.multipca.components}'
        if pca != '':
            pca = 'yes'
        regression = args.regression.type
        horizonts = len(args.selector.multi_horizonts)

        # table_line = f"{name} & {model} & {levels} & {pca} & {regression} & {normalization} & {r_is} & {r_os} \\\\"
        table_line = f"{name} & {model} & {levels} & {pca} & {regression} & {horizonts} & {r_is} & {r_os} & {pnl} \\\\"
        lns.append(table_line)
    return lns


@main.command()
def future_table_maker(path: str):
    header = \
        """\\begin{table}[]
            \\centering
            \\begin{tabular}{lccccc|ll|l}
            \\toprule
                Exp & Model & Levels & PCA & Regression & Horizonts & IS & OS & PnL \\\\
            \\midrule"""

    ending = \
        """\\bottomrule
            \\end{tabular}
            \\caption{Results}
            \\label{tab.results}
        \\end{table}
        """

    l = get_future_results_from_folder(path)
    answer = '\n'.join([header] + l + [ending])
    print(answer)


def transpose(l):
    return list(map(list, zip(*l)))


@main.command()
def custom():
    def get_name(l):
        return f"individual_price_impact_issize_1800_ossize_1800_rolling_1800_seltype_SplitOFI_lvl_{l}_volnrm_regtype_lasso"

    experiments = [get_name(l) for l in range(1, 11)]
    centering = f"r|{'c' * 10}"

    bk = '\\'
    header = \
        f"""{bk}begin{{table}}[]
            {bk}centering
            {bk}resizebox{{{bk}textwidth}}{{!}}{{
            {bk}begin{{tabular}}{{{centering}}}
            {bk}toprule
                 & {' & '.join([f"${bk}modpid{{{l}}}$" for l in range(1, 11)])} {bk}{bk}
            {bk}midrule"""

    def get_rows(name):
        path = f'./results_server/{name}'
        dct, dct_std = get_log_results(os.path.join(path, 'logs.log'))

        is_r2 = str(round(dct['in_r2'] * 100, 2))
        std_is_r2 = "(" + str(round(dct_std['in_r2'] * 100, 2)) + ")"
        os_r2 = str(round(dct['os_r2'] * 100, 2))
        std_os_r2 = "(" + str(round(dct_std['os_r2'] * 100, 2)) + ")"

        return [is_r2, std_is_r2, os_r2, std_os_r2]

    rows = [get_rows(n) for n in experiments]
    rows = transpose(rows)
    rows[0] = ['\multirow{ 2}{*}{IS $R^2$} '] + rows[0]
    rows[1] = [' '] + rows[1]
    rows[2] = ['\multirow{ 2}{*}{OS $R^2$}'] + rows[2]
    rows[3] = [' '] + rows[3]

    ending = \
        """\\bottomrule
            \\end{tabular}}
            \\caption{Results}
            \\label{tab.results}
        \\end{table}
        """

    rows = list(map(lambda x: ' & '.join(x), rows))
    rows = ' \\\\ \n'.join(rows)
    answer = '\n'.join([header, rows, ending])
    print(answer)


if __name__ == '__main__':
    main()
