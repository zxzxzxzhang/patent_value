from df_translator import df_Translator

translator = df_Translator(input_file='df2_new.xlsx', output_file='df2_new.csv')
translator.process_batch()

"""
Df_Translator class is designed for translating large datasets in batches.

Parameters:
-----------
input_file : str
    Path to the input file (either .xlsx or .csv) containing the data to be translated.
output_file : str
    Path to the output file (either .xlsx or .csv) where the translated results will be saved.
source_column : str, optional
    The column name containing the text to be translated, default is '权利要求'.
target_column : str, optional
    The column name where the translated text will be stored, default is '权利要求 (英文)'.
source_lang : str, optional
    Source language code, default is 'auto' for automatic detection.
target_lang : str, optional
    Target language code, default is 'en' for English translation.
batch_size : int, optional
    Number of rows to process in each batch, default is 500.
max_retries : int, optional
    Maximum number of retry attempts if a translation fails, default is 5.
max_len : int, optional
    Maximum allowed character length per translation segment, default is 600 characters.
sleep_time : int, optional
    Sleep time in seconds between batches to avoid rate limiting, default is 100 seconds.
"""