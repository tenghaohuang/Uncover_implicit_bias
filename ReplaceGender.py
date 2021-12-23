import nltk
from nltk.tag.stanford import StanfordNERTagger
from allennlp.predictors.predictor import Predictor

predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz",cuda_device = 0)
predictor._model = predictor._model.cuda()
print(predictor.cuda_device)
print("predictor loaded")
import spacy

nlp = spacy.load('en_core_web_lg')
import os

java_path = "" #include java path
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

            if (ent in characters):
                rt.append(i)
                break

    return rt


def getCharacters(setence):
    words = nltk.word_tokenize(setence)
    characters = [i for i in ner_tagger.tag(words) if (i[1] == 'PERSON')]

    return set(characters)


def isPron(pt, li):
    for num, tup in enumerate(li):
        if (tup[0] <= pt and pt < tup[1]):
            return tup[1], True
        if (tup[0] == pt):
            return pt, True

    return pt, False


def replaceAll(sentence):

    doc = nlp(sentence)
    token_dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
    clusters = getPron(sentence)
    init = 65

    curpos = 0
    snt = []
    while curpos < len(token_dependencies):
        flags = []
        positions = []
        characters = []
        for num, group in enumerate(clusters):
            (curpos, flag) = isPron(curpos, group)
            flags.append(flag)

            if flag:
                characters.append("Protagonist" + chr(init + num))
                positions.append(curpos)

        if (len(positions) > 0):
            curpos = max(positions)
            character = characters[positions.index(curpos)]
        if True in flags:
            #                 print(token_dependencies[curpos])
            if (token_dependencies[curpos][1] == 'poss'):
                #                     if(token_dependencies[curpos+1][1] == 'case'):
                #                         curpos += 1
                snt.append(character + "'s")
            elif (token_dependencies[curpos][1] == 'case'):
                snt.append(character + "'s")
            else:
                snt.append(character)
        else:
            snt.append(token_dependencies[curpos][0])
        curpos += 1

    ans = " ".join(snt) + '\n'

    return ans


def writeFile(file_name, content):
    a = open(file_name, 'a')
    a.write(content)
    a.close()


def process(inputf, outputf):
    f = open(inputf, "r")
    for para in f.readlines():

        #         print(repr(para))
        if (para == "\n"):
            continue

        processed = replaceAll(para)

        sts = processed.strip("\n").split('"')
        sts = ("").join(sts)
        if (processed != []):
            #             for i in sts:
            writeFile(outputf, sts + '\n')

print("Male")
process("male.txt", "male_masked.txt")
print("Female")
process("female.txt", "female_masked.txt")
