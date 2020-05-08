import os
from process_file import process_doc
import random
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
from full_model import Classifier
import torch.nn as nn
import torch.optim as optim
import time


def get_batch(doc, ref_type='headline'):
    sent, ls, out, sids = [], [], [], []
    sent.append(doc.headline)
    ls.append(len(doc.headline))
    for sid in doc.sentences:
        if SPEECH:
            out.append(out_map[doc.sent_to_speech.get(sid, 'NA')])
        else:
            out.append(out_map[doc.sent_to_event.get(sid)])
        sent.append(doc.sentences[sid])
        ls.append(len(doc.sentences[sid]))
        sids.append(sid)
    ls = torch.LongTensor(ls)
    out = torch.LongTensor(out)
    return sent, ls, out, sids


def train(epoch, data):
    start_time = time.time()
    total_loss = 0
    global prev_best_macro

    for ind, doc in enumerate(data):
        model.train()
        optimizer.zero_grad()
        sent, ls, out, _ = get_batch(doc)
        if has_cuda:
            ls = ls.cuda()
            out = out.cuda()

        _output, _, _, _ = model.forward(sent, ls)
        loss = criterion(_output, out)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        del sent, ls, out
        if has_cuda:
            torch.cuda.empty_cache()

    print("--Training--\nEpoch: ", epoch, "Loss: ", total_loss, "Time Elapsed: ", time.time()-start_time)
    perf = evaluate(validate_data)
    # print(perf)
    if prev_best_macro < perf:
        prev_best_macro = perf
        print ("-------------------Test start-----------------------")
        _ = evaluate(test_data, True)
        print ("-------------------Test end-----------------------")
        torch.save(model.state_dict(), 'discourse_lstm_model.pt')


def evaluate(data, is_test=False):
    y_true, y_pred = [], []
    model.eval()
    for doc in data:
        sent, ls, out, sids = get_batch(doc)
        if has_cuda:
            ls = ls.cuda()
            #out = out.cuda()

        _output, _, _, _ = model.forward(sent, ls)
        _output = _output.squeeze()
        _, predict = torch.max(_output, 1)
        y_pred += list(predict.cpu().numpy() if has_cuda else predict.numpy())
        temp_true = list(out.numpy())
        y_true += temp_true

    print("MACRO: ", precision_recall_fscore_support(y_true, y_pred, average='macro'))
    print("MICRO: ", precision_recall_fscore_support(y_true, y_pred, average='micro'))
    if is_test:
        print("Classification Report \n", classification_report(y_true, y_pred))
    print("Confusion Matrix \n", confusion_matrix(y_true, y_pred))
    return precision_recall_fscore_support(y_true, y_pred, average='macro')[2]


if __name__ == '__main__':

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--drop', help='DROP', default=6, type=float)
    # parser.add_argument('--learn_rate', help='LEARNING RATE', default=0, type=float)
    # parser.add_argument('--loss_wt', help='LOSS WEIGHTS', default=0, type=str)
    parser.add_argument('--seed', help='SEED', default=0, type=int)

    args = parser.parse_args()

    has_cuda = torch.cuda.is_available()

    SPEECH = 0
    if SPEECH:
        out_map = {'NA':0, 'Speech':1}
    else:
        out_map = {'NA':0,'Main':1,'Main_Consequence':2, 'Cause_Specific':3, 'Cause_General':4, 'Distant_Historical':5,
        'Distant_Anecdotal':6, 'Distant_Evaluation':7, 'Distant_Expectations_Consequences':8}

    train_data = []
    validate_data = []
    test_data = []
    for domain in ["Business", "Politics", "Crime", "Disaster", "kbp"]:
        subdir = "../data/train/"+domain
        files = os.listdir(subdir)
        for file in files:
            if '.txt' in file:
                doc = process_doc(os.path.join(subdir, file), domain) #'../data/Business/nyt_corpus_data_2007_04_27_1843240.txt'
                #print(doc.sent_to_event)
                train_data.append(doc)
        subdir = "../data/test/"+domain
        files = os.listdir(subdir)
        for file in files:
            if '.txt' in file:
                doc = process_doc(os.path.join(subdir, file), domain) #'../data/Business/nyt_corpus_data_2007_04_27_1843240.txt'
                #print(doc.sent_to_event)
                test_data.append(doc)


    subdir = "../data/validation"
    files = os.listdir(subdir)
    for file in files:
        if '.txt' in file:
            doc = process_doc(os.path.join(subdir, file), 'VAL') #'../data/Business/nyt_corpus_data_2007_04_27_1843240.txt'
            #print(doc.sent_to_event)
            validate_data.append(doc)
    print(len(train_data), len(validate_data), len(test_data))

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if has_cuda:
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    prev_best_macro = 0.

    model = Classifier({'num_layers': 1, 'hidden_dim': 512, 'bidirectional': True, 'embedding_dim': 1024,
                        'dropout': 0.5, 'out_dim': len(out_map)})

    if has_cuda:
        model = model.cuda()
    model.init_weights()

    criterion = nn.CrossEntropyLoss()

    print("Model Created")

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params, lr=5e-5, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)

    try:
        for epoch in range(15):
            print("---------------------------Started Training Epoch = {0}--------------------------".format(epoch+1))
            train(epoch, train_data)

    except KeyboardInterrupt:
        print ("----------------- INTERRUPTED -----------------")
        evaluate(validate_data)
        evaluate(test_data)
