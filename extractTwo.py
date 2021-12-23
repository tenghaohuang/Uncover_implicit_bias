import nltk
from nltk.tag.stanford import StanfordNERTagger
from allennlp.predictors.predictor import Predictor

predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz")
import spacy

nlp = spacy.load('en_core_web_lg')
import os

java_path = "C://Program Files//Java//jre1.8.0_281//bin//java.exe"
os.environ['JAVAHOME'] = java_path
jar = '.\\stanford-ner\\stanford-ner.jar'
model = '.\\stanford-ner\\classifiers\\english.conll.4class.distsim.crf.ser.gz'
ner_tagger = StanfordNERTagger(model, jar, encoding='utf8')


def getPron(para):

    output = predictor.predict(
        document=para
    )
    rt = []
    characters = [i[0] for i in getCharacters(para)]
    female_pron = ["She", "she", "Her", "her"]
    male_pron = ["He", "he", "him", "Him"]
    unresolved_pron = ["I", "Me", "me"]
    characters = characters + female_pron + male_pron + unresolved_pron

    for i in output['clusters']:
        for j in i:

            ent = " ".join(output['document'][j[0]:j[1] + 1])
            #             print("look",ent)
            if (ent in characters):
                rt.append(i)
                break

    return rt


def getCharacters(setence):
    words = nltk.word_tokenize(setence)
    characters = [i for i in ner_tagger.tag(words) if (i[1] == 'PERSON')]


    return set(characters)


def getObjSts(sentence):

    doc = nlp(sentence)
    token_dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
    clusters = getPron(sentence)

    start = 0
    end = 0
    rec = []

    while (end < len(token_dependencies)):
        if (token_dependencies[end][0] != "."):
            end += 1
            continue
        else:
            count = 0
            entities = []
            for i in clusters:

                for j in i:

                    if (j[0] > end):
                        break
                    if (start <= j[0] and j[0] <= end):
                        entities.append((j[0], j[1] - j[0] + 1))
                        #                         count += 1
                        break
            uni_ent = {}
            for tup in entities:
                if tup[0] not in uni_ent:
                    uni_ent[tup[0]] = tup[1]
                elif (uni_ent[tup[0]] < tup[1]):
                    uni_ent[tup[0]] = tup[1]

            sorted(uni_ent.items(), key=lambda item: item[0])
            for i in uni_ent:
                if (start <= i and i <= end):
                    count += 1
            rec.append(count)
            start = end
            end += 1

    sts = sentence.split(".")

    rt = []
    selected = [num for num, i in enumerate(rec) if i >= 2]
    for i in selected:
        rt.append(sts[i])
    return rt


def writeFile(file_name, content):
    a = open(file_name, 'a')
    a.write(content)
    a.close()


def process(inputf, outputf):
    f = open(inputf, "r")
    cnt = 0
    for para in f.readlines():
        cnt += 1
        if (cnt % 100 == 0):
            print(cnt)

        if ("protagonistB" not in para and "ProtagonistB" not in para):
            continue

        writeFile(outputf + '_two_and_above.txt', para + ".\n")


test2 = " ProtagonistA was a junior , but ProtagonistA had a huge crush on ProtagonistB . ProtagonistA loved to write songs on ProtagonistA's guitar , so ProtagonistA wrote one just for ProtagonistB . Then , gathering ProtagonistA's nerve ProtagonistA played it for ProtagonistB during free period . ProtagonistB's face lit up at the emotion of ProtagonistA's song . ProtagonistB asked ProtagonistA for a date right then and there ! "

process("male_masked.txt", "male")
process("female_masked.txt", "female")