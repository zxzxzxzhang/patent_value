import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm
import time
import re

# 读取原始数据
df = pd.read_excel(f'df2_new.xlsx')

# 确保 '权利要求 (英文)' 列为 object 类型，以存储字符串
df['权利要求 (英文)'] = df['权利要求 (英文)'].astype(object)
df['权利要求'] = df['权利要求'].astype(str)
# 初始化翻译器
translator = GoogleTranslator(source='auto', target='en')

# 每次处理的行数
batch_size = 500

# 设置最大重试次数
max_retries = 5

# 定义一个函数用于切分长句子，依据句号。或.切分，并确保每个片段不超过100个字符
def split_text(text, max_len=600):
    sentences = re.split(r'(?<=[。\.])', text)  # 按句号（。或.）进行切分
    split_sentences = []
    current_sentence = ''

    for sentence in sentences:
        if len(current_sentence) + len(sentence) <= max_len:
            current_sentence += sentence
        else:
            split_sentences.append(current_sentence)
            current_sentence = sentence

    if current_sentence:
        split_sentences.append(current_sentence)

    return split_sentences

# 定义一个函数用于翻译并拼接
def translate_text(text):
    # 切分文本
    split_sentences = split_text(text)

    translated_sentences = []

    for sentence in split_sentences:
        # 尝试翻译，失败时重试
        for attempt in range(max_retries):
            try:
                translated_sentence = translator.translate(sentence)
                translated_sentences.append(translated_sentence)
                break
            except Exception as e:
                print(f"Error translating sentence: {e}, retrying... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(10)
                if attempt == max_retries - 1:
                    print(f"Failed to translate sentence after {max_retries} attempts.")
                    #time.sleep(30)
                    translated_sentences.append("Translation Failed")

    # 将翻译后的句子拼接回完整的段落
    return ' '.join(translated_sentences)

# 按批次遍历 DataFrame
for i in range(0, len(df), batch_size):

    batch = df.iloc[i:i + batch_size].copy()
    translated_batch = []

    for idx, row in tqdm(batch.iterrows(), desc=f"batch {i // batch_size + 1}", total=len(batch)):
        try:
            # 判断是否已经有翻译结果，若有则跳过
            if pd.notna(row['权利要求 (英文)']) and row['权利要求 (英文)'].strip() != '':
                translated_batch.append(row['权利要求 (英文)'])  # 保持已有翻译结果
                continue

            # 如果没有翻译结果，执行翻译
            translated_text = translate_text(row['权利要求'])
            translated_batch.append(translated_text)

        except Exception as row_error:
            # 如果某行发生错误，跳过该行，记录错误信息
            print(f"翻译第 {idx} 行时发生错误: {row_error}")
            translated_batch.append(None)  # 对于错误的行，填入 None 值，或者填入一个错误提示文本

    # 使用 .loc[] 将翻译结果写回 DataFrame
    df.loc[i:i + batch_size - 1, '权利要求 (英文)'] = translated_batch

    # 每处理完一个批次，保存整个 DataFrame 到 Excel 文件
    print(f"saving file")
    df.to_csv(f'df2_new.csv', index=False)
    print(f"Done")

    # 每处理完一个批次暂停160秒
    #time.sleep(100)

