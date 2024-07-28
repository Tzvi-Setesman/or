# Run this after verifying the "conflict_sentiment" column to get a confusion matrix and plot of
# precision and recall values over the weeks.
# ASSUMPTIONS:
# 1. The column "conflict_sentiment" hold GPT's answers
# 2. The column "conflict_sentiment_v" holds the "real" value.
#    Empty cells are allowed (will be filled with "conflict_sentiment" column values).
# PAY ATTENTION: if there is no data for a specific date - the precision and recall values are 0

import matplotlib
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
matplotlib.use('TkAgg')


def read_file(file_path):
    return pd.read_excel(file_path)


def filter_df(data):
    # filter out text (Azure content filter) rows
    cond_not_txt = data["conflict_sentiment"].apply(lambda x: str(x) in ["0", "1", "2", "3", "9"])
    filtered_data = data[cond_not_txt]

    # # fill GPT's answer to all posts that wasn't manually checked
    # condition = (filtered_data['conflict_sentiment_v'].isna()) & ((filtered_data['conflict_sentiment'] == 0) |
    #                                                               (filtered_data['conflict_sentiment'] == 1) |
    #                                                               (filtered_data['conflict_sentiment'] == 2) |
    #                                                               (filtered_data['conflict_sentiment'] == 3) |
    #                                                               (filtered_data['conflict_sentiment'] == 9) |
    #                                                               (filtered_data['conflict_sentiment'] == "0") |
    #                                                               (filtered_data['conflict_sentiment'] == "1") |
    #                                                               (filtered_data['conflict_sentiment'] == "2") |
    #                                                               (filtered_data['conflict_sentiment'] == "3") |
    #                                                               (filtered_data['conflict_sentiment'] == "9"))
    # filtered_data.loc[condition, 'conflict_sentiment_v'] = filtered_data.loc[condition, 'conflict_sentiment'].astype(int)

    # # filter out posts that wasn't checked or filled as checked
    # cond_verified = ~filtered_data["conflict_sentiment_v"].isna()
    # filtered_data = filtered_data[cond_verified]

    filtered_data["conflict_sentiment"] = filtered_data["conflict_sentiment"].astype(int)
    filtered_data["conflict_sentiment_v"] = filtered_data["conflict_sentiment_v"].astype(int)

    # randomly select 20% out of each category - uncomment if there is a vast amount of data
    # df_orig_0 = filtered_data[filtered_data["conflict_sentiment"] == 0]
    # df_orig_1 = filtered_data[filtered_data["conflict_sentiment"] == 1]
    # df_orig_2 = filtered_data[filtered_data["conflict_sentiment"] == 2]
    # df_orig_3 = filtered_data[filtered_data["conflict_sentiment"] == 3]
    # df_orig_9 = filtered_data[filtered_data["conflict_sentiment"] == 9]
    #
    # random_orig_0 = df_orig_0.sample(frac=0.2, random_state=42)
    # random_orig_1 = df_orig_1.sample(frac=0.2, random_state=42)
    # random_orig_2 = df_orig_2.sample(frac=0.2, random_state=42)
    # random_orig_3 = df_orig_3.sample(frac=0.2, random_state=42)
    # random_orig_9 = df_orig_9.sample(frac=0.2, random_state=42)
    # filtered_data = pd.concat([random_orig_0, random_orig_1, random_orig_2, random_orig_3, random_orig_9],
    #                           ignore_index=True)

    return filtered_data

# compute precision or recall, by the value of FP_or_FN
def comp_pre_re(TP, FP_or_FN):
    if TP + FP_or_FN == 0:
        return 0
    return TP / (TP + FP_or_FN)


def compute_precision_recall_per_class(class_number, orig, verified):
    if class_number == 4:
        class_number = 9
    TP = [1 for i in range(len(orig)) if orig[i] == verified[i] == class_number]
    FP = [1 for i in range(len(orig)) if orig[i] == class_number and verified[i] != class_number]
    FN = [1 for i in range(len(orig)) if orig[i] != class_number and verified[i] == class_number]
    TN = len(orig) - (sum(TP) + sum(FP) + sum(FN))

    precision = comp_pre_re(sum(TP), sum(FP))
    recall = comp_pre_re(sum(TP), sum(FN))
    return precision, recall, sum(TP), sum(FP), sum(FN), TN


def compute_f1_score(precision, recall, n_classes):
    f1_lst = [2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) != 0 else 0 for i
              in range(n_classes)]
    total_f1_score = sum(f1_lst) / n_classes

    return f1_lst, total_f1_score


def compute_accuracy(orig, verified):
    correct = [1 for i in range(len(orig)) if orig[i] == verified[i]]
    return sum(correct) / len(orig)


def compute_measures(orig, verified, n_classes):
    acc = compute_accuracy(orig, verified)

    precision_lst, recall_lst = [None for _ in range(n_classes)], [None for _ in range(n_classes)]
    TP_cnt_lst, FP_cnt_lst, FN_cnt_lst, TN_cnt_lst = [None for _ in range(n_classes)], [None for _ in range(n_classes)], \
                                                     [None for _ in range(n_classes)], [None for _ in range(n_classes)]
    for i in range(n_classes):
        precision_lst[i], recall_lst[i], TP_cnt_lst[i], FP_cnt_lst[i], FN_cnt_lst[i], TN_cnt_lst[i] = \
            compute_precision_recall_per_class(i, orig, verified)

    macro_avg_precision = sum(precision_lst) / n_classes
    macro_avg_recall = sum(recall_lst) / n_classes

    micro_avg_precision = sum(TP_cnt_lst) / (sum(TP_cnt_lst) + sum(FP_cnt_lst)) if (sum(TP_cnt_lst) + sum(
        FP_cnt_lst)) != 0 else 0
    micro_avg_recall = sum(TP_cnt_lst) / (sum(TP_cnt_lst) + sum(FN_cnt_lst)) if (sum(TP_cnt_lst) + sum(
        FN_cnt_lst)) != 0 else 0

    return acc, precision_lst, recall_lst, macro_avg_precision, macro_avg_recall, micro_avg_precision, micro_avg_recall, \
           TP_cnt_lst, FP_cnt_lst, FN_cnt_lst, TN_cnt_lst

# print computed precision, recall, F1 and accuracy
def print_to_screen(accuracy, precision_lst, recall_lst, macro_p, macro_r, micro_p, micro_r, f1_lst, total_f1, TP, FP,
                    FN, n_classes):
    print(f"accuracy is {round(accuracy * 100, 2)}%, F1 score is: {round(total_f1, 2)}")
    print("\nmeasures per each class:")
    for i in range(4):
        print(
            f"\nclass {i}:\nnumber of TP: {TP[i]}, FP: {FP[i]}, FN: {FN[i]}\nprecision: {round(precision_lst[i] * 100, 2)}%, recall: {round(recall_lst[i] * 100, 2)}%, F1 score: {round(f1_lst[i], 2)}")

    if n_classes == 5:
        print(
            f"\nclass 9:\nnumber of TP: {TP[-1]}, FP: {FP[-1]}, FN: {FN[-1]}\nprecision: {round(precision_lst[-1] * 100, 2)}%, recall: {round(recall_lst[-1] * 100, 2)}%, F1 score: {round(f1_lst[-1], 2)}")

    print("\nmicro and macro measures:")
    # macro average gives equal weight to each class
    print(f"macro average precision: {round(macro_p * 100, 2)}%, macro average recall: {round(macro_r * 100, 2)}%")

    # micro average gives equal weight to each instance
    print(f"micro average precision: {round(micro_p * 100, 2)}%, micro average recall: {round(micro_r * 100, 2)}%")


def plot_confusion_matrix(orig, verified):
    cm = confusion_matrix(orig, verified)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=[f'Class {i}' if i <= 3 else 'Class 9' for i in range(len(cm))],
                yticklabels=[f'Class {i}' if i <= 3 else 'Class 9' for i in range(len(cm))])

    plt.title('Confusion Matrix')
    plt.xlabel('Real Classification')
    plt.ylabel("GPT's Classification")


def create_list(n_days, n_classes):
    return [[0 for _ in range(n_days)] for _ in range(n_classes)]


def plot_per_week(data, n_classes):
    # create list of dates
    dates = data["upload_date"].tolist()
    dates = [str(date)[:10] for date in dates]
    dates = list(set(dates))
    if 'NaT' in dates:
        dates.remove('NaT')
    dates.sort()

    precision_lst = create_list(len(dates), n_classes)
    recall_lst = create_list(len(dates), n_classes)

    df_lst = []  # a list of df per date
    for j in range(len(dates)):
        df_lst.append(data[data["upload_date"].astype(str).str.contains(dates[j], case=False, na=False)])

    for i in range(n_classes):  # for each class
        for j in range(len(dates)):  # for each date
            orig = df_lst[j]["conflict_sentiment"].tolist()
            ver = df_lst[j]["conflict_sentiment_v"].tolist()

            res = compute_precision_recall_per_class(i, orig, ver)
            precision_lst[i][j], recall_lst[i][j] = res[:2]

    n_weeks = math.ceil(len(dates) / 7)
    precision_per_week = create_list(n_weeks, n_classes)
    precision_number_of_days = create_list(n_weeks, n_classes)
    recall_per_week = create_list(n_weeks, n_classes)
    recall_number_of_days = create_list(n_weeks, n_classes)
    for class_num in range(n_classes):
        week_idx = 0
        for j in range(len(dates)):
            if j % 7 == 0 and j != 0:
                week_idx += 1
            amount = precision_lst[class_num][j]
            if amount != 0:
                precision_number_of_days[class_num][week_idx] += 1
                precision_per_week[class_num][week_idx] += amount
            amount = recall_lst[class_num][j]
            if amount != 0:
                recall_number_of_days[class_num][week_idx] += 1
                recall_per_week[class_num][week_idx] += amount

    fig, axes = plt.subplots(nrows=(n_classes % 2) + 2, ncols=2, figsize=(10, 1.3*n_classes))

    measures_lst = [precision_per_week, recall_per_week]
    measures_number_per_week = [precision_number_of_days, recall_number_of_days]
    measures_names_lst = ["precision", "recall"]

    # divide by 7 for results in range 0-1
    for i in range(len(measures_lst)):
        for j in range(len(measures_lst[i])):
            for k in range(len(measures_lst[i][j])):
                if measures_number_per_week[i][j][k] != 0:
                    measures_lst[i][j][k] /= measures_number_per_week[i][j][k]

    dates_names = [date[5:] for date in dates]
    for i in range(len(dates_names)):
        if dates_names[i][:2] == "01":
            dates_names[i] = "Jan " + dates_names[i].split("-")[1]
        elif dates_names[i][:2] == "02":
            dates_names[i] = "Feb " + dates_names[i].split("-")[1]
        elif dates_names[i][:2] == "03":
            dates_names[i] = "Mar " + dates_names[i].split("-")[1]
        elif dates_names[i][:2] == "04":
            dates_names[i] = "Apr " + dates_names[i].split("-")[1]
        elif dates_names[i][:2] == "05":
            dates_names[i] = "May " + dates_names[i].split("-")[1]
        elif dates_names[i][:2] == "06":
            dates_names[i] = "June " + dates_names[i].split("-")[1]
        elif dates_names[i][:2] == "07":
            dates_names[i] = "July " + dates_names[i].split("-")[1]
        elif dates_names[i][:2] == "08":
            dates_names[i] = "Aug " + dates_names[i].split("-")[1]
        elif dates_names[i][:2] == "09":
            dates_names[i] = "Sept " + dates_names[i].split("-")[1]
        elif dates_names[i][:2] == "10":
            dates_names[i] = "Oct " + dates_names[i].split("-")[1]
        elif dates_names[i][:2] == "11":
            dates_names[i] = "Nov " + dates_names[i].split("-")[1]
        elif dates_names[i][:2] == "12":
            dates_names[i] = "Dec " + dates_names[i].split("-")[1]

    # create weeks names list
    weeks_names = [None for _ in range(n_weeks)]
    week_idx = 0
    for i in range(len(dates_names)):
        if i % 7 == 0:
            weeks_names[week_idx] = dates_names[i] + " - "
        if (i+1) % 7 == 0:
            weeks_names[week_idx] += dates_names[i]
            week_idx += 1

    if "" in dates_names:
        dates_names.remove("")

    # if the last week isn't a full week
    if weeks_names[-1][-1] == " ":
        weeks_names[-1] += dates_names[-1]

    for i, inner_list in enumerate(precision_lst):
        row_index = i // 2
        col_index = i % 2

        ax = axes[row_index, col_index]

        for j in range(len(measures_lst)):
            ax.plot(weeks_names, measures_lst[j][i], label=measures_names_lst[j])

        if i != 4:
            ax.set_title(f'Class {i}')
        else:
            ax.set_title(f'Class {9}')

        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score')

        ax.tick_params(axis='x', rotation=20, labelsize=6)

    ax.legend()
    plt.tight_layout()


def main():
    data_file_path = "all_data_us__updated_to_2023-12-26__13_15.xlsx"  # insert an Excel file path
    data = read_file(data_file_path)

    # How many unique classes in data (9 was added later) before or after added the option "9" to the sentiment query
    if "9" in data['conflict_sentiment'].values or 9 in data['conflict_sentiment'].values or \
            "9" in data['conflict_sentiment_v'].values or 9 in data['conflict_sentiment_v'].values:
        n_classes = 5
    else:
        n_classes = 4

    # filter gpt results that are not classe (0,1,2 ....) optional: randomly get 20% of the data
    filtered_data = filter_df(data)

    # compute accuracy, precision, recall, f1 - sklearn like 
    orig = filtered_data["conflict_sentiment"].tolist()
    verified = filtered_data["conflict_sentiment_v"].tolist()
    accuracy, precision_lst, recall_lst, macro_p, macro_r, micro_p, micro_r, TP_lst, FP_lst, FN_lst, TN_lst = compute_measures(
        orig, verified, n_classes)
    f1_lst, total_f1 = compute_f1_score(precision_lst, recall_lst, n_classes)

    print_to_screen(accuracy, precision_lst, recall_lst, macro_p, macro_r, micro_p, micro_r, f1_lst, total_f1, TP_lst,
                    FP_lst, FN_lst, n_classes)

    plot_confusion_matrix(orig, verified)
    plot_per_week(filtered_data, n_classes)
    plt.show()


if __name__ == '__main__':
    main()
