import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm
import time
import re
import os

class df_Translator:
    def __init__(self, input_file, output_file, source_column='权利要求', target_column='权利要求 (英文)',
                 source_lang='auto', target_lang='en', batch_size=500, max_retries=5, max_len=600, sleep_time=100):
        self.input_file = input_file
        self.output_file = output_file
        self.source_column = source_column
        self.target_column = target_column
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.max_len = max_len
        self.sleep_time = sleep_time

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

        self.input_type = os.path.splitext(self.input_file)[1]
        self.output_type = os.path.splitext(self.output_file)[1]

        self.translator = GoogleTranslator(source=self.source_lang, target=self.target_lang)

        # reead the input file
        if self.input_type == '.xlsx':
            self.df = pd.read_excel(self.input_file)
        else:
            self.df = pd.read_csv(self.input_file)

        # ensure that the target column is of object type to store strings
        self.df[self.target_column] = self.df[self.target_column].astype(object)
        self.df[self.source_column] = self.df[self.source_column].astype(str)

    def split_text(self, text):
        """
        Split text by sentence while ensuring that each chunk is under the maximum length.
        """
        sentences = re.split(r'(?<=[。\.])', text)  # split by period
        split_sentences = []
        current_sentence = ''

        for sentence in sentences:
            if len(current_sentence) + len(sentence) <= self.max_len:
                current_sentence += sentence
            else:
                split_sentences.append(current_sentence)
                current_sentence = sentence

        if current_sentence:
            split_sentences.append(current_sentence)

        return split_sentences

    def translate_text(self, text):
        """
        Translate the input text by splitting it into chunks if needed and handling retries.
        """
        split_sentences = self.split_text(text)
        translated_sentences = []

        for sentence in split_sentences:
            # retry the translation if it fails
            for attempt in range(self.max_retries):
                try:
                    translated_sentence = self.translator.translate(sentence)
                    translated_sentences.append(translated_sentence)
                    break
                except Exception as e:
                    print(f"Error translating sentence: {e}, retrying... (Attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(10)
                    if attempt == self.max_retries - 1:
                        print(f"Failed to translate sentence after {self.max_retries} attempts.")
                        time.sleep(10)
                        translated_sentences.append("Translation Failed")

        return ' '.join(translated_sentences)

    def process_batch(self):
        """
        Process the translation in batches.
        """
        for i in range(0, len(self.df), self.batch_size):
            batch = self.df.iloc[i:i + self.batch_size].copy()
            translated_batch = []

            for idx, row in tqdm(batch.iterrows(), desc=f"batch {i // self.batch_size + 1}", total=len(batch)):
                try:
                    # skip if the translation already exists
                    if pd.notna(row[self.target_column]) and row[self.target_column].strip() != '':
                        translated_batch.append(row[self.target_column])  # keep the existing translation
                        continue

                    # translate the text
                    translated_text = self.translate_text(row[self.source_column])
                    translated_batch.append(translated_text)

                except Exception as row_error:
                    print(f"Line {idx} failed: {row_error}")
                    translated_batch.append(None)  # fill in None for rows with errors

            # 将翻译结果写回 DataFrame
            self.df.loc[i:i + self.batch_size - 1, self.target_column] = translated_batch

            # 每处理完一个批次，保存整个 DataFrame
            print(f"saving file")
            if self.output_type == '.xlsx':
                self.df.to_excel(self.output_file, index=False)
            else:
                self.df.to_csv(self.output_file, index=False)

            print(f"Done")

            # sleep for a while to avoid rate limiting
            print('sleeping for 100 seconds')
            time.sleep(self.sleep_time)


# Translator for the given DataFrame
'''
if __name__ == "__main__":
    translator = df_Translator(input_file='df2_new.xlsx', output_file='df2_new.csv')
    translator.process_batch()
'''