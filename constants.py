from tickers import TICKERS

LEVELS = 10
ROUNDING_RET = 4
ROUNDING = 4

START_TRADE = (9 * 60 + 30) * 60  # 09:30
END_TRADE = (16 * 60 + 0) * 60  # 16:00
VOLATILE_TIMEFRAME = 30 * 60  # volatile period to not consider

START_TIME = START_TRADE + VOLATILE_TIMEFRAME
END_TIME = END_TRADE - VOLATILE_TIMEFRAME

BUCKET_SIZE = 10  # seconds


def levels_list(name: str, levels: int = LEVELS) -> list[str]:
    return [f"{name}_{i}" for i in range(1, levels + 1)]


OFI_TYPES = ['add', 'cancel', 'trade']
OFI_NAMES = ['ofi', 'ofi_add', 'ofi_cancel', 'ofi_trade']
SPLIT_OFI_NAMES = ['ofi_add', 'ofi_cancel', 'ofi_trade']
OFI_COLS = sum([levels_list(i) for i in OFI_NAMES], [])
SPLIT_OFI_COLS = sum([levels_list(i) for i in SPLIT_OFI_NAMES], [])
VOLUME_COLS = levels_list('volume')

DEFAULT_TICKERS = TICKERS[:100]
