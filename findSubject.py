
from allennlp.predictors.predictor import Predictor
predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz",cuda_device = 0)
import nltk
import spacy
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nlp = spacy.load('en_core_web_lg')

male = []
female = []
unresolved = []


def writeFile(file_name, content):
    a = open(file_name, 'a')
    a.write(content)
    a.close()


def getPron(para):

    output = predictor.predict(
        document=para
    )

    for i in output['clusters']:
        for j in i:
            print(output['document'][j[0]:j[1] + 1])
        print('\n')

    return output['clusters']


def findSubj(li):
    root = ""
    for i in li:
        if (i[1] == "ROOT"):
            root = i[0]
    for i in li:
        if (i[1] == "nsubj" and i[2] == root and (i[0] == "protagonistA" \
                                                  or i[0] == "ProtagonistA")):
            return True
    return False


def process(inputf, outputf):
    f = open(inputf, "r")
    for p in f.readlines():
        #         p = "i like hiking."
        p.replace("!", ".")
        p = p.strip("\n").split(".")
        p = [i + "\n" for i in p if i != '']

        for para in p:
            if (para == "\n" or para == ".\n"):
                continue

            tokens = nltk.word_tokenize(para)
            tagged_sent = nltk.pos_tag(tokens)
            doc = nlp(para)


            token_dependencies = [(token.text, token.dep_, token.head.text) for token in doc]

            flag = findSubj(token_dependencies)

            if (flag == True):

                continue
            else:
                if(len(para)>5):
                    writeFile(outputf + '_obj.txt', para)

process("male_two_and_above.txt","male_two_and_above")
process("female_two_and_above.txt","female_two_and_above")