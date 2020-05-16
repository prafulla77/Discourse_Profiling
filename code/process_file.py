from collections import OrderedDict
import random


def sort_order(sids):
    temp = sorted([int(elem[1:]) for elem in sids])
    return ['S'+str(elem) for elem in temp]

class Document(object):

    def __init__(self, fname, domain):
        self.name = fname.split('/')[-1]
        self.domain = domain
        self.url = 'None'
        self.headline = ['headline']
        self.lead = []
        self.sentences = OrderedDict()
        self.tags = dict()
        self.sent_to_speech = None
        self.sent_to_event = None
        self.sent_to_topic = None
        self.sids = []


def process_doc(fname, domain):

    # Process text document
    f = open(fname, 'r')
    # print(fname)
    doc = Document(fname, domain)
    lead_para = False
    sids = []
    for line in f:
        temp = line.strip()
        if temp == '':
            if lead_para == True:
                for key in doc.sentences:
                    doc.lead += doc.sentences[key]
            lead_para = False
            continue

        temp = temp.split()
        if temp[0] == 'URL':
            doc.url = temp[1]
        elif temp[0] == 'DATE/':
            pass
        elif temp[0] == 'H':
            if len(temp[1:]) > 0:
                doc.headline = temp[1:]
        else:
            if temp[0] == 'S1':
                lead_para = True
            doc.sentences[temp[0]] = temp[1:]
            sids.append(temp[0])

    # Process annotation file
    f = open(fname[:-3]+'ann')
    prev_label = "headline"
    sent_to_event = dict()
    sent_to_speech = dict()
    sent_to_topic = dict()
    for line in f:
        temp = line.strip().split('\t')
        if len(temp) == 3:
            label = temp[1].split()[0]
            if label == 'Speech':
                sent_to_speech[temp[2]] = label
            else:
                # print(temp)
                sent_to_event[temp[2]] = label
                sent_to_topic[temp[2]] = 0
                #################################################################
                ################ Can't use this, not always sorted ##############
                #################################################################
                # if label != prev_label and label in ['Main']:  # 'Distant_Anecdotal',
                #     sent_to_topic[temp[2]] = 1
                # # elif label != prev_label and prev_label in ['Distant_Anecdotal']:
                # #     sent_to_topic[temp[2]] = 1
                # # elif prev_label not in ['Distant_Expectations_Consequences', 'Distant_Evaluation'] and label in ['Distant_Expectations_Consequences', 'Distant_Evaluation']:
                # #     sent_to_topic[temp[2]] = 1
                # # elif label != prev_label and prev_label in ['Cause_Specific']:
                # #     sent_to_topic[temp[2]] = 1
                # elif prev_label == "headline" or (label == "Main_Consequence" and "Main" not in prev_label):
                #     sent_to_topic[temp[2]] = 1
                # elif "Distant" in prev_label and 'Cause' in label:
                #     sent_to_topic[temp[2]] = 1
                # else:
                #     sent_to_topic[temp[2]] = 0
                # prev_label = label

    doc.sent_to_event = sent_to_event
    doc.sent_to_speech = sent_to_speech
    doc.sids = sort_order(sids)
    first_sentence = True
    for ind in range(len(doc.sids)-1):
        # print(doc.sids[ind], sent_to_event[doc.sids[ind]])
        if first_sentence and sent_to_event[doc.sids[ind]] != 'NA':  # First sentence always transition
            first_sentence = False
            sent_to_topic[doc.sids[ind]] = 1
        if sent_to_event[doc.sids[ind]] not in ['Main', 'Main_Consequence'] and 'Main' in sent_to_event[doc.sids[ind+1]]:  # 'Distant_Anecdotal',
            sent_to_topic[doc.sids[ind+1]] = 1
        elif "Distant" in sent_to_event[doc.sids[ind]] and 'Cause' in sent_to_event[doc.sids[ind+1]]:
            sent_to_topic[doc.sids[ind+1]] = 1
        elif sent_to_event[doc.sids[ind+1]] != sent_to_event[doc.sids[ind]] and sent_to_event[doc.sids[ind+1]] == 'Distant_Anecdotal':
            sent_to_topic[doc.sids[ind+1]] = 1
        elif sent_to_event[doc.sids[ind+1]] != sent_to_event[doc.sids[ind]] and sent_to_event[doc.sids[ind]] == 'Distant_Anecdotal':
            sent_to_topic[doc.sids[ind]] = 1
    # print(sent_to_topic)
    # print(sent_to_event)
    assert(len(sent_to_event) == len(doc.sids))
    doc.sent_to_topic = sent_to_topic

    return doc

#process_doc('../data/kbp/NYT_ENG_20130912.0240.txt')

