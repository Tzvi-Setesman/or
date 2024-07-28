# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:08:35 2024

@author: DEKELCO
"""

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#df_o = pd.read_excel('./experiments/' + dataset_path+ '.xlsx')

suffix= '_mistral_instruct-v02'
dataset_path = 'data/2024-01-12_15-08-46__2024-01-21_18-42-15_HouthisTweet_classified' + suffix
df = pd.read_parquet(dataset_path + '.parquet')

df.info()
df['gpt4_houthis_sentiment'].value_counts()

df = df[(df.gpt4_houthis_sentiment != 'Azure content filter') & (df.llm_houthis_sentiment != -1)]
# df[df.gpt4_houthis_sentiment.isin([0,1,2,3,4,5])]

df['llm_houthis_sentiment'] = df.llm_houthis_sentiment.astype(int)
df['gpt4_houthis_sentiment'] = df.gpt4_houthis_sentiment.astype(int)






# Assuming your DataFrame is named 'df'
y_true = df['gpt4_houthis_sentiment']
y_pred = df['llm_houthis_sentiment']

# Compute the classification report
report = classification_report(y_true, y_pred)
print("Classification Report:")
print(report)

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)


# Create the confusion matrix plot
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax)
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
plt.show()

# Create errors .xlsx for review where there are errors 
df[df.llm_houthis_sentiment != df.gpt4_houthis_sentiment].to_excel(dataset_path+ f'_errors{suffix}.xlsx')
