# -*- coding: utf-8 -*-

import json
import re
from scipy.stats import spearmanr, pearsonr, kendalltau
import pandas as pd


def read_jsonl(file_name):
    data = []
    with open(file_name, 'r') as jsonl_file:
        for line in jsonl_file:
            # Parse each line as a JSON object and append it to the data list
            data.append(json.loads(line))
    return data


def read_tsv(file_name):
    return pd.read_csv(file_name, sep='\t', encoding='utf-8', on_bad_lines='skip')


def read_json(file_name):
    with open(file_name, 'r') as json_file:
        return json.load(json_file)
    

def read_txt(file_name):
    with open(file_name, 'r', encoding='utf-8') as txt_file:
        return txt_file.readlines()


def extract_num_from_str(pattern, prediction_str, prediction_idx, num_position, result, index):
    '''Extract the number from the prediction string and append it to the result list, 
    append the index of the prediction to the index list if there is no number in the prediction string.'''
    try:
        num = float(pattern.findall(prediction_str)[num_position])
        if num  == float('inf'):
            result.append(0.0)
        else:
            result.append(num)
    except IndexError:
        index.append(prediction_idx)

def extract_number(data, key="predict", position=0):
    result = []
    index = [] # record the index of predictions that do not have a number
    pattern = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")
    if key:
        if isinstance(data, list): # if the data is a list of dictionaries
            for idx, item in enumerate(data):
                pred = item[key]
                extract_num_from_str(pattern, pred, idx, position, result, index)

        else: # if the data is a pandas dataframe
            for idx, item in data[key].items():
                pred = item
                extract_num_from_str(pattern, pred, idx, position, result, index)
    else: # if the data is a list of strings
        for idx, item in enumerate(data):
            pred = item
            extract_num_from_str(pattern, pred, idx, position, result, index)

    return result, index


def compute_correlation_score(num, label, dropped_index=None):
    if len(num) != len(label):
        if dropped_index:
            # drop the corresponding label if there is no number in the prediction
            new_label = [label[i] for i in range(len(label)) if i not in dropped_index]
        else:
            print("The number of predictions and labels are not equal. Please check the data.")
            new_label = label
    else:
        new_label = label
    
    print(spearmanr(num, new_label))
    print(pearsonr(num, new_label))
    print(kendalltau(num, new_label))

    return round(spearmanr(num, new_label)[0], 4), round(pearsonr(num, new_label)[0], 4), round(kendalltau(num, new_label)[0], 4)


if __name__ == "__main__":
    
    template_version = "03-mixtral"
    subfolder = "./llm_output_samples/"

    language_pairs = ["en-de", "en-mr", "en-zh", "et-en", "ne-en", "ro-en", "ru-en", "si-en"]

    #put results into a dataframe
    df = pd.DataFrame(columns=['spearman','pearson','kendall','dropped_rows'], index=language_pairs)
    for language_pair in language_pairs:
        print("%s:" % language_pair)
        data = pd.read_csv(subfolder + language_pair.upper() + "_outputs_t" + template_version + ".tsv", sep='\t', encoding='utf-8', on_bad_lines='skip')
        num, dropped_index = extract_number(data, key="vllm_output", position=0)
        label = list(map(round, read_tsv("./raw_data/" + language_pair + "/" + language_pair + "_overlaps_test.tsv")["score"].tolist()))

        corrs = compute_correlation_score(num, label, dropped_index)
        print("Dropped rows: ", len(dropped_index))

        df.loc[language_pair] = [corrs[0], corrs[1], corrs[2], len(dropped_index)]
        
    df.to_excel(subfolder + "correlation_scores_t" + template_version + ".xlsx", index=True)