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

def get_results_from_folder(path: str):
    folders = os.listdir(path)
    folders = sorted(list(filter(lambda x: os.path.isdir(os.path.join(path, x)), folders)))
    lns = []
    for f in folders:
        f_path = os.path.join(path, f)
        args = ConfigDict(load_yaml(os.path.join(f_path, 'config.yaml')))
        r_is, r_os = get_is_os_from_log(os.path.join(f_path, 'logs.log'))

        name = 'Individual' if args.experiment.name == 'individual_price_impact' else 'Universal'
        model = args.selector.type
        levels = args.selector.levels
        pca = '' if 'pca' not in args.processor else args.processor.pca
        regression = args.regression.type
        table_line = f"{name} & {model} & {levels} & {pca} & {regression} & {r_is} & {r_os} \\\\"
        lns.append(table_line)
    return lns

@main.command()
def table_maker(path: str):
    header = \
"""\\begin{table}[]
    \\centering
    \\begin{tabular}{lcccc|rr}
    \\toprule
        Exp & Model & Levels & PCA & Regression & In Sample & Out of Sample \\\\
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

if __name__ == '__main__':
    main()
