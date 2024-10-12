# --------------------------------------------------------
# Sage_Gated
# Copyright (c) 2021 Xiang Zhang
# All Rights Reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# Written by Xiang Zhang
# --------------------------------------------------------

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

df = pd.read_excel('pbat_cn_low.xlsx')
df = df[['权利要求 (英文)','摘要 (英文)','标题 (英文)']]

# cut sentence
def truncate_text(text, max_length=500, redundancy=20):
    if len(text) <= max_length:
        return text
    
    end_idx_cn = text.rfind('。', 0, max_length)
    end_idx_en = text.rfind('.', 0, max_length)
    end_idx = max(end_idx_cn, end_idx_en)
    
    if end_idx == -1:
        start_idx = max_length - redundancy if max_length > redundancy else 0
        end_idx = max_length + redundancy if len(text) > max_length + redundancy else len(text)
        return text[start_idx:end_idx]
    else:
        return text[:end_idx + 1]

# 应用该函数到摘要和权利要求字段
df['摘要 (英文)'] = df['摘要 (英文)'].apply(lambda x: truncate_text(x, max_length=500))
df['权利要求 (英文)'] = df['权利要求 (英文)'].apply(lambda x: truncate_text(x, max_length=500))

summaries = df['权利要求 (英文)'].tolist()
abstract = df['摘要 (英文)'].tolist()
title = df['标题 (英文)'].tolist()
all_texts = summaries + title + abstract

model = SentenceTransformer('distilbert-base-nli-mean-tokens')
device = torch.device("cuda:0")
torch.cuda.set_device(device)

#
train_examples = [InputExample(texts=[text, text], label=1.0) for text in all_texts]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

# fit model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    show_progress_bar=True,
)
model = model.to(device)

model.save('distilbert_finetuned')

