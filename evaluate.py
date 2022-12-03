import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def isfinite(x, name):
    print(name)
    try:
        if np.isfinite(x).all():
            print("Finite")
        else:
            print("Not finite")
    except:
        if torch.isfinite(x).all():
            print("Finite")
        else:
            print("Not finite")
    print(x)

@torch.no_grad()
def evaluate_fewshot(
    encoder, loader, n_way=5, n_shots=[1,5], n_query=15, classifier='LR', power_norm=False):

    encoder.eval()

    accs = {}
    for n_shot in n_shots:
        accs[f'{n_shot}-shot'] = []

    for idx, (images, _) in enumerate(loader):
        print("Loading batch: ", idx)
        images = images.cuda(non_blocking=True)
        f = encoder(images)
        del images
        # isfinite(f, "f before norm")
        f = f/f.norm(dim=-1, keepdim=True)

        # isfinite(f, "f before power norm")
        if power_norm:
            f = f ** 0.5
        
        # isfinite(f, "f after power norm")

        max_n_shot = max(n_shots)
        test_batch_size = int(f.shape[0]/n_way/(n_query+max_n_shot))
        sup_f, qry_f = torch.split(f.view(test_batch_size, n_way, max_n_shot+n_query, -1), [max_n_shot, n_query], dim=2)
        qry_f = qry_f.reshape(test_batch_size, n_way*n_query, -1).detach().cpu().numpy()
        qry_label = torch.arange(n_way).unsqueeze(1).expand(n_way, n_query).reshape(-1).numpy()

        del f
        
        for tb in range(test_batch_size):
            # print(f"Batch {tb} of {test_batch_size}")
            for n_shot in n_shots:
                # print(f"n_shot: {n_shot}")
                cur_sup_f = sup_f[tb, :, :n_shot, :].reshape(n_way*n_shot, -1).detach().cpu().numpy()
                # check if cur_sup_f is all finite numbers, if not print it and check for sup_f after the reshape, if it is all finite numbers then check for f
                # isfinite(cur_sup_f, "cur_sup_f")
                # isfinite(sup_f[tb, :, :n_shot, :].reshape(n_way*n_shot, -1).detach().cpu(), "sup_f[tb, :, :n_shot, :].reshape(n_way*n_shot, -1).detach().cpu()")
                # isfinite(sup_f[tb, :, :n_shot, :].reshape(n_way*n_shot, -1).detach(), "sup_f[tb, :, :n_shot, :].reshape(n_way*n_shot, -1).detach()")
                # isfinite(sup_f[tb, :, :n_shot, :].reshape(n_way*n_shot, -1), "sup_f[tb, :, :n_shot, :].reshape(n_way*n_shot, -1)")
                # isfinite(f, "f")
            
                # print("cur_sup_f fine")
                cur_sup_y = torch.arange(n_way).unsqueeze(1).expand(n_way, n_shot).reshape(-1).numpy()
                # print("cur_sup_y fine")
                cur_qry_f = qry_f[tb]
                # print("cur_qry_f fine")
                cur_qry_y = qry_label
                # print("cur_qry_y fine")

                if classifier == 'LR':
                    clf = LogisticRegression(penalty='l2',
                                            random_state=0,
                                            C=1.0,
                                            solver='lbfgs',
                                            max_iter=1000,
                                            multi_class='multinomial')
                elif classifier == 'SVM':
                    clf = LinearSVC(C=1.0)
                # print("clf fine")

                clf.fit(cur_sup_f, cur_sup_y)
                # print("clf fit fine")
                cur_qry_pred = clf.predict(cur_qry_f)
                # print("clf predict fine")
                acc = metrics.accuracy_score(cur_qry_y, cur_qry_pred)
                # print("metrics fine")

                accs[f'{n_shot}-shot'].append(acc)
                # print("append fine")
            print(f"n_shot: {n_shot}", accs[f'{n_shot}-shot'][-1])
        
    for n_shot in n_shots:
        acc = np.array(accs[f'{n_shot}-shot'])
        mean = acc.mean()
        std = acc.std()
        c95 = 1.96*std/math.sqrt(acc.shape[0])
        print('classifier: {}, power_norm: {}, {}-way {}-shot acc: {:.2f}+{:.2f}'.format(
            classifier, power_norm, n_way, n_shot, mean*100, c95*100))
    return 
    
