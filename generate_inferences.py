from comet2.comet_model import PretrainedCometModel
import pickle
import random

comet_model = PretrainedCometModel(device=0)


def writeFile(file_name, content):
    a = open(file_name, 'a')
    a.write(content)
    a.close()


def saveFile(outputname, content):
    file = open(outputname, 'wb')
    pickle.dump(content, file)


def getDict(file_name, mode):
    f = open(file_name, "r")

    attrDict = []
    xrDict = []
    orDict = []

    wt_dict = []
    nd_dict = []
    it_dict = []
    cnt = 0
    txt = []
    for i in f.readlines():
        if (i == " \n"):
            continue
        txt.append(i)
    print(txt[0:20])

    for st in txt:

        cnt += 1


        if mode == "xAttr":
            inference = comet_model.predict(st, "xAttr", num_beams=5)

            attrDict.append(inference)
        elif mode == "xReact":
            xr_inference = comet_model.predict(st, "xReact", num_beams=5)

            xrDict.append(xr_inference)
        elif mode == "oReact":
            or_inference = comet_model.predict(st, "oReact", num_beams=5)
            orDict.append(or_inference)
        elif mode == "motivation":
            wt_inference = comet_model.predict(st, "xWant", num_beams=5)
            wt_dict.append(wt_inference[0])

            nd_inference = comet_model.predict(st, "xNeed", num_beams=5)
            nd_dict.append(nd_inference[0])

            it_inference = comet_model.predict(st, "xIntent", num_beams=5)
            it_dict.append(it_inference[0])
        if (cnt % 1000 == 0):
            print(cnt)
            print(xrDict[-5:])
            print(orDict[-5:])
            print(attrDict[-5:])

    if mode == "xAttr":
        saveFile(file_name + "_at_dict", attrDict)
    elif mode == "xReact":
        saveFile(file_name + "_xr_dict", xrDict)
    elif mode == "oReact":
        saveFile(file_name + "_or_dict", orDict)
    elif mode == "motivation":
        saveFile(file_name + "_wt_dict", wt_dict)

        saveFile(file_name + "_nd_dict", nd_dict)
        saveFile(file_name + "_it_dict", it_dict)


getDict("male_masked_subj.txt", "xAttr")
getDict("female_masked_subj.txt","xAttr")

getDict("male_masked_subj.txt","xReact")
getDict("female_masked_subj.txt","xReact")

getDict("male_two_and_above_subj.txt","oReact")
getDict("female_two_and_above_subj.txt","oReact")

getDict("male_two_and_above_obj.txt","oReact")
getDict("female_two_and_above_obj.txt","oReact")

getDict("male_masked_subj.txt","motivation")
getDict("female_masked_subj.txt","motivation")
getDict("female_two_and_above_obj.txt")
