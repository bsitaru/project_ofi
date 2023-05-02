import os
import data_manipulation.prediction as pred
import typer
import strategy.sign as stsign

from logging_utils import get_logger

main = typer.Typer()


@main.command()
def sign(folder_path: str):
    logger = get_logger(folder_path, 'pred_signs')
    files_list = os.listdir(folder_path)
    all_correct, all_tot = 0, 0
    for file_name in files_list:
        if 'predict' not in file_name:
            continue
        df = pred.get_prediction_df(os.path.join(folder_path, file_name))
        ratio, correct, tot_num = stsign.get_heat_ratio(df)
        text = f"{file_name} --- ratio: {ratio} --- correct: {correct} --- total: {tot_num}"
        print(text)
        logger.info(text)
        all_correct += correct
        all_tot += tot_num

    ratio = all_correct / all_tot
    text = f"TOTAL --- ratio: {ratio} --- correct: {all_correct} --- total: {all_tot}"
    print(text)
    logger.info(text)


if __name__ == '__main__':
    main()
