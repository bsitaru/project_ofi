from data_manipulation.tick_ofi import generate_tick_ofi_file
from data_manipulation.bucket_ofi import generate_bucket_ofi_file

if __name__ == '__main__':
    file = './lobster_sample/tickers/TFX/TFX_2016-04-27_10/TFX_2016-04-27_24900000_57900000'
    message_file_path = f"{file}_message_10.csv"
    orderbook_file_path = f"{file}_orderbook_10.csv"

    tick_ofi_file_path = "./tick_ofis.csv"
    bucket_ofi_file_path = "./bucket_ofis.csv"

    generate_tick_ofi_file(message_file_path, orderbook_file_path, tick_ofi_file_path)
    generate_bucket_ofi_file(message_file_path, orderbook_file_path, bucket_ofi_file_path)


