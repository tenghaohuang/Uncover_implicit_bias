import pandas as pd
df = pd.read_csv('Lexicons of bias - Gender stereotypes.csv')
import gensim
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

model1 = KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin.gz',binary=True)

from empath import Empath
lexicon = Empath()

intellect = ["intellectual", "intuitive", "imaginative", "knowledgeable", "ambitious", "intelligent", "opinionated", "admirable", "eccentric", "crude", "likable", "empathetic", "superficial", "tolerant", "resourceful", "uneducated", "academically", "studious", "temperamental", "exceptional", "cynical", "outspoken", "destructive", "dependable", "amiable", "impulsive", "frivolous", "insightful", "overconfident", "charismatic", "prideful", "influential", "likeable", "unconventional", "educated", "flawed", "articulate", "pretentious", "perceptive", "vulgar", "easygoing", "listener", "skillful", "assertive", "philosophical", "rebellious", "selfless", "cunning", "deceptive", "artistic", "appalling", "overbearing", "temperament", "diligent", "charitable", "disposition", "quirky", "strategic", "compulsive", "benevolent", "pessimistic", "scientific", "flamboyant", "obsessive", "selective", "oriented", "humorous", "narcissistic", "reliable", "headstrong", "manipulative", "practical", "rewarding", "refined", "resilient", "desirable", "spiritual", "tendencies", "pompous", "judgmental", "respected", "inexperienced", "compassionate", "promiscuous", "argumentative", "conventional", "intellectually", "expressive", "impractical", "observant", "fickle", "hyperactive", "immoral", "straightforward", "vindictive"]

import numpy as np
import math
def get_words(cates):
    tmp = df.loc[:,cates].values
    words = []
    for i in tmp:
        for j in i:

            if(type(j) is not float):
                words.append(j)
    return set(words)
appearence = ["beautiful","sexual"]
appear = get_words(appearence)
power = ["dominant","strong"]
power = get_words(power)
weak = ['submissive','weak','dependent','afraid']
weak = get_words(weak)
print("loaded")

from scipy import stats
import pickle
from scipy import spatial
import pickle


file_list = []

file_list.append("male_masked_subj.txt_at_dict")
file_list.append("female_masked_subj.txt_at_dict")

file_list.append("male_masked_subj.txt_xr_dict")
file_list.append("female_masked_subj.txt_xr_dict")

file_list.append("male_two_and_above_obj.txt_or_dict")
file_list.append("female_two_and_above_obj.txt_or_dict")

file_list.append("male_two_and_above_subj.txt_or_dict")
file_list.append("female_two_and_above_subj.txt_or_dict")


def load_file(file):
    file = open(file, 'rb')

    words = pickle.load(file)

    file.close()
    return words


weak_vecs = [model1.wv[i] for i in weak if i in model1]
power_vecs = [model1.wv[i] for i in power if i in model1]
appear_vecs = [model1.wv[i] for i in appear if i in model1]
intellect_vecs = [model1.wv[i] for i in intellect if i in model1]


def calculateSubspace(A, B):
    A_vecs = [model1.wv[i] for i in A if i in model1]
    B_vecs = [model1.wv[i] for i in B if i in model1]

    suma = A_vecs[0].copy()

    for i in range(1, len(A_vecs)):
        suma += A_vecs[i]
    sumb = B_vecs[0].copy()
    for i in range(1, len(B_vecs)):
        suma += B_vecs[i]
    return suma / len(A) - sumb / len(B)


def compute(words_clusters, title):
    intel_sum = []
    appear_sum = []
    power_sum = []
    power_subspace = calculateSubspace(power, weak)

    for x in words_clusters:
        if x not in model1:
            continue
        # weak
        intel_sims = 0
        appear_sims = 0
        for j in intellect:

            if (j in model1):
                intel_sims += model1.similarity(x, j)
        for k in appear:
            if (k in model1):
                appear_sims += model1.similarity(x, k)

        power_sum.append(1 - spatial.distance.cosine(model1.wv[x], \
                                                     power_subspace))
        intel_sum.append(intel_sims / len(intellect))

        appear_sum.append(appear_sims / len(appear))
    print("dumping")
    f = open(title + "_intellect.pkl", "wb")
    pickle.dump(intel_sum, f)
    f.close()
    f = open(title + "_appear.pkl", "wb")
    pickle.dump(appear_sum, f)
    f.close()
    f = open(title + "_power.pkl", "wb")
    pickle.dump(power_sum, f)
    f.close()
    return [np.median(stats.zscore(intel_sum)), np.median(stats.zscore(appear_sum)), np.median(stats.zscore(power_sum))]


def get_stats(l):
    return (np.max(l), np.min(l), np.percentile(l, 25), np.percentile(l, 75), \
            np.median(l))


def getLexiconScore(f1, f2):
    male = load_file(f1)
    female = load_file(f2)
    if (type(female[0]) is list):
        male = [i[0] for i in male]
        female = [i[0] for i in female]
    if (type(female[0]) is tuple):
        male = [i[1] for i in male]
        female = [i[1] for i in female]

    male_removed = Counter(male).most_common(5)
    female_removed = Counter(female).most_common(5)

    male_removed = [i[0] for i in male_removed]
    female_removed = [i[0] for i in female_removed]

    m = []
    for i in male:
        if i in male_removed:
            continue
        m.append(i)
    f = []
    for i in female:
        if i in female_removed:
            continue
        f.append(i)

    if "curious" in m:
        print("wrong")
    else:
        print("OK")

    m = list(filter(lambda a: a != 'none', m))
    f = list(filter(lambda a: a != 'none', f))

    return compute(m, "male"), compute(f, "male")


from collections import Counter


def getLexiconScore_b5(f1, f2):
    male = load_file(f1)
    female = load_file(f2)

    mm = []

    for i in male:
        mm += set(i)
    ff = []

    for i in female:
        ff += set(i)

    m = list(filter(lambda a: a != 'none', mm))
    f = list(filter(lambda a: a != 'none', ff))

    return compute(m, "../0515replotting/" + f1.split("/")[1] + "/" + f1.split("/")[3]) \
        , compute(f, "../0515replotting/" + f2.split("/")[1] + "/" + f2.split("/")[3])


directs = ["../Generated/Beam_5/", "../humanWritten/Beam_5/"]
result = []
for direct in directs:
    if "Beam" in direct:
        print(direct)
        print("subj_at")
        result.append(
            getLexiconScore_b5(direct + "male_masked_subj.txt_at_dict", direct + "female_masked_subj.txt_at_dict"))

    else:
        print(direct)
        print("subj_at")
        result.append(
            getLexiconScore(direct + "male_masked_subj.txt_at_dict", direct + "female_masked_subj.txt_at_dict"))

print(result)