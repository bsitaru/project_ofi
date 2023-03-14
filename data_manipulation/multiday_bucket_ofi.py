import pandas as pd
import data_manipulation.bucket_ofi as bucket_ofi

c_dtype = bucket_ofi.c_dtype
c_dtype['date'] = str
c_dtype['ticker'] = str


def prepare_df_for_multiday(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    # Add ticker and date
    [ticker, d, _, _] = file_name[:-4].split('_')
    df['ticker'] = ticker
    df['date'] = str(d)

    return df


def get_multiday_df(file_path: str):
    try:
        df = pd.read_csv(file_path, dtype=c_dtype)
        return df
    except:
        return pd.DataFrame(columns=list(c_dtype.keys()))
