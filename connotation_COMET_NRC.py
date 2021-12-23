import argparse
import csv
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict, Counter
import pandas as pd
from scipy.stats import zscore
import math
import warnings
from pandas.core.common import SettingWithCopyWarning
import pickle
from collections import Counter

def get_NRC_lexicon():
    '''
    @output:
    - A dictionary of format {word : score}
    '''
    lexicon = 'NRC-VAD-Lexicon.txt'
    val_dict = {}
    aro_dict = {}
    dom_dict = {}
    with open(lexicon, 'r') as infile:
        reader = csv.DictReader(infile, delimiter="\t")
        for row in reader:
            word = row['Word']
            val_dict[word] = float(row['Valence'])
            aro_dict[word] = float(row['Arousal'])
            dom_dict[word] = float(row['Dominance'])
    return (val_dict, aro_dict, dom_dict)


def write_output(writer, df, label,title):
    df = df.copy()
    # print(df.shape[0])
    z = zscore(df['Value'])
    filehandler = open("VAD_5/"+title+label+".obj", "wb")
    pickle.dump(z, filehandler)
    for i in range(int(df.shape[0])):
        df.loc[i, 'value'] = z[i]
    df = df.drop(columns=['Value'])

    stats = df.groupby(['Category']).agg(['mean', 'median','count', 'std'])
    print(stats)
    # assert (1 == 0)
    cis = []
    for i in stats.index:
        mean,median, c, s = stats.loc[i]
        cis.append(1.96 * s / math.sqrt(c))

    stats['ci'] = cis
    for index, row in stats.iterrows():
        writer.writerow({'category': row.name,
                         'dimension': label,
                         'median': row['value']['median'],
                         'mean': row['value']['mean'],
                         "ci": row['ci'].values[0]})
    # stats = []
    # df = []


def compute(inputt):
    adj = {'Category': [], 'Measurement': [], 'Value': [], 'Word': []}
    val_dict, aro_dict, dom_dict = get_NRC_lexicon()
    print(inputt)
    file = open(inputt, 'rb')
    tups = pickle.load(file)
    words = []


    for i in tups:
        words += set(i)

    words = list(filter(lambda a: a != 'none', words))
    file.close()
    # category = "male" if input[0] == "m" else "female"
    category = inputt

    # print(words)
    # assert(1==0)
    # words = set(words)
    #     print(len(words))


    c = Counter(words)
    # print
    removed = c.most_common(5)

    removed = [i[0] for i in removed]
    print(removed)
    # assert(False)
    for word in words:
        if word in removed:
            continue
        if word in val_dict:
            val = val_dict[word]
            aro = aro_dict[word]
            dom = dom_dict[word]
            adj['Category'].append(category)
            adj['Measurement'].append('Valence')
            adj['Value'].append(val)
            adj['Word'].append(word)
            adj['Category'].append(category)
            adj['Measurement'].append('Arousal')
            adj['Value'].append(aro)
            adj['Word'].append(word)
            adj['Category'].append(category)
            adj['Measurement'].append('Dominance')
            adj['Value'].append(dom)
            adj['Word'].append(word)

    adj_df = pd.DataFrame.from_dict(adj)
    val_df = adj_df[adj_df['Measurement'] == 'Valence']
    aro_df = adj_df[adj_df['Measurement'] == 'Arousal']
    dom_df = adj_df[adj_df['Measurement'] == 'Dominance']
    #     print(adj_df[0:50])
    with open('VA_5_median.csv', 'a') as outfile:
        fieldnames = ['category', 'dimension', 'median','mean', 'ci']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        # write_output(writer, power_df, 'power')
        # write_output(writer, agen_df, 'agency')
        # write_output(writer, sent_df, 'sentiment')
        write_output(writer, val_df, 'valence',category)
        write_output(writer, aro_df, 'arousal',category)
        write_output(writer, dom_df, 'dominance',category)
print("yoho")
filehandler = open("VAD_remove_5/aba.obj", "wb")
pickle.dump("1", filehandler)

# directs = ["Generated/generated","humanWritten/humanwritten"]
# for head in directs:
compute("male_masked_subj.txt_at_dict")
compute("female_masked_subj.txt_at_dict")

compute("male_masked_subj.txt_xr_dict")
compute("female_masked_subj.txt_xr_dict")

compute("male_two_and_above_obj.txt_or_dict")
compute("female_two_and_above_obj.txt_or_dict")

compute("male_two_and_above_subj.txt_or_dict")
compute("female_two_and_above_subj.txt_or_dict")
