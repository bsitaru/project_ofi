import logging
import sys
import os
def get_logger(folder_path: str, logger_name: str = 'myapp') -> logging.Logger:
    logger = logging.getLogger(logger_name)
    if len(logger.handlers) == 0 and folder_path is not None:
        hdlr = logging.FileHandler(os.path.join(folder_path, f"{logger_name}.log"))
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
        logger.setLevel(logging.INFO)
    return logger

def log(text: str, logger=None):
    print(text, file=sys.stderr, flush=True)
    if logger is not None:
        logger.info(text)


def log_tickers(tickers):
    if len(tickers) == 1:
        return tickers[0]
    elif len(tickers) <= 5:
        return ', '.join(tickers)
    else:
        return ', '.join(tickers[:5]) + ' ...'