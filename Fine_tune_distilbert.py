'''
Pre-release Notice

This repository contains code associated with our ongoing research project titled "Research on patent portfolio valuation based on Multi-SAGE-TechNexus model". The code is being made available for **review purposes only** and is subject to the following restrictions:

1. Non-commercial use only: This code may only be used for academic or non-commercial purposes.
2. No redistribution or modification**: Redistribution or modification of this code is not permitted until the associated research paper has been officially published.
3. Temporary access: The code in this repository is subject to updates and may change without notice until the final release.

After the publication of the corresponding research paper, we plan to release the code under a more permissive open-source license (e.g., MIT License).

For any questions or specific permissions, please contact zhangx2293@gmail.com with the subject "Pre-release Code Inquiry".

Written by Xiang Zhang
'''

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

