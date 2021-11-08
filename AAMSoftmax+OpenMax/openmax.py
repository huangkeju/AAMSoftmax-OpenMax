import numpy as np
import scipy.spatial.distance as spd
import torch
import torch.nn.functional as F

import libmr


def calc_distance(query_score, mcv, eu_weight, distance_type='eucos'):
    if distance_type == 'eucos':
        query_distance = spd.euclidean(mcv, query_score) * eu_weight + \
            spd.cosine(mcv, query_score)
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(mcv, query_score)
    elif distance_type == 'cosine':
        query_distance = spd.cosine(mcv, query_score)
    else:
        print("distance type not known: enter either of eucos, euclidean or cosine")
    return query_distance


def fit_weibull(means, dists, categories, tailratio=0.01, distance_type='eucos'):
    """
    Input:
        means (C, channel, C)
        dists (N_c, channel, C) * C
    Output:
        weibull_model : Perform EVT based analysis using tails of distances and save
                        weibull model parameters for re-adjusting softmax scores
    """
    weibull_model = {}
    for mean, dist, category_name in zip(means, dists, categories):
        # print(dist[distance_type])
        # print(np.max(dist[distance_type], axis=1))
        weibull_model[category_name] = {}
        weibull_model[category_name]['distances_{}'.format(distance_type)] = dist[distance_type]
        weibull_model[category_name]['mean_vec'] = mean
        weibull_model[category_name]['weibull_model'] = []
        tailsize = int(tailratio*len(dist[distance_type][0]))

        for channel in range(mean.shape[0]):
            mr = libmr.MR()
            # tailtofit = np.sort(dist[distance_type][channel, :])[-tailsize:]
            # print(tailtofit)
            # mr.fit_high(tailtofit, len(tailtofit))
            mr.fit_high(dist[distance_type][channel, :], tailsize)

            weibull_model[category_name]['weibull_model'].append(mr)

    return weibull_model

def fit_weibull_adapt(means, dists, categories, tailratio=1.0, distance_type='eucos'):
    """
    Input:
        means (C, channel, C)
        dists (N_c, channel, C) * C
    Output:
        weibull_model : Perform EVT based analysis using tails of distances and save
                        weibull model parameters for re-adjusting softmax scores
    """
    weibull_model = {}
    for mean, dist, category_name in zip(means, dists, categories):
        # print(dist[distance_type])
        # print(np.max(dist[distance_type], axis=1))
        weibull_model[category_name] = {}
        weibull_model[category_name]['distances_{}'.format(distance_type)] = dist[distance_type]
        weibull_model[category_name]['mean_vec'] = mean
        weibull_model[category_name]['weibull_model'] = []
        # tailsize = int(tailratio*len(dist[distance_type][0]))

        for channel in range(mean.shape[0]):
            
            tr = tailratio
            cr = 0.

            while cr < 0.99 and tr >= 0.02:
                tailsize = int(tr*len(dist[distance_type][0]))
                mr = libmr.MR()
                # tailtofit = np.sort(dist[distance_type][channel, :])[-tailsize:]
                # print(tailtofit)
                # mr.fit_high(tailtofit, len(tailtofit))
                mr.fit_high(dist[distance_type][channel, :], tailsize)
                tr -= 0.01
                w_scores = np.array([mr.w_score(d) for d in dist[distance_type][channel, :]])
                cr = float(np.sum(w_scores<0.5))/len(w_scores)

            # print(tr)
            weibull_model[category_name]['weibull_model'].append(mr)

    return weibull_model


def query_weibull(category_name, weibull_model, distance_type='eucos'):
    return [weibull_model[category_name]['mean_vec'],
            weibull_model[category_name]['distances_{}'.format(distance_type)],
            weibull_model[category_name]['weibull_model']]


def compute_openmax_prob(scores, scores_u):
    prob_scores, prob_unknowns = [], []
    for s, su in zip(scores, scores_u):
        # channel_scores = np.exp(s)
        # channel_unknown = np.exp(np.sum(su))

        channel_scores = s
        channel_unknown = np.sum(su)

        total_denom = np.sum(channel_scores) + channel_unknown
        prob_scores.append(channel_scores / total_denom)
        prob_unknowns.append(channel_unknown / total_denom)

    # Take channel mean
    scores = np.mean(prob_scores, axis=0)
    unknowns = np.mean(prob_unknowns, axis=0)
    modified_scores = scores.tolist() + [unknowns]
    return modified_scores


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def openmax(weibull_model, categories, input_score, cscore, eu_weight, alpha=10, distance_type='eucos'):
    """Re-calibrate scores via OpenMax layer
    Output:
        openmax probability and softmax probability
    """
    nb_classes = len(categories)

    ranked_list = cscore.argsort().ravel()[::-1][:alpha]
    alpha_weights = [((alpha + 1) - i) / float(alpha) for i in range(1, alpha + 1)]
    # alpha_weights = [1. for i in range(1, alpha + 1)]
    omega = np.zeros(nb_classes)
    omega[ranked_list] = alpha_weights

    scores, scores_u = [], []
    for channel, input_score_channel in enumerate(input_score):
        score_channel, score_channel_u = [], []
        for c, category_name in enumerate(categories):
            mav, dist, model = query_weibull(category_name, weibull_model, distance_type)
            channel_dist = calc_distance(input_score_channel, mav[channel], eu_weight, distance_type)
            wscore = model[channel].w_score(channel_dist)
            # print(category_name, channel_dist, wscore)
            modified_score = cscore[channel][c] * (1 - wscore * omega[c])
            score_channel.append(modified_score)
            score_channel_u.append(cscore[channel][c] - modified_score)

        # print(score_channel)
        # print(score_channel_u)
        scores.append(score_channel)
        scores_u.append(score_channel_u)

    scores = np.asarray(scores)
    scores_u = np.asarray(scores_u)

    openmax_prob = np.array(compute_openmax_prob(scores, scores_u))
    softmax_prob = softmax(np.array(cscore.ravel()))
    return openmax_prob, softmax_prob


def compute_channel_distances(mavs, features, eu_weight=0.5):
    """
    Input:
        mavs (channel, C)
        features: (N, channel, C)
    Output:
        channel_distances: dict of distance distribution from MAV for each channel.
    """
    eucos_dists, eu_dists, cos_dists = [], [], []
    for channel, mcv in enumerate(mavs):  # Compute channel specific distances
        eu_dists.append([spd.euclidean(mcv, feat[channel]) for feat in features])
        cos_dists.append([spd.cosine(mcv, feat[channel]) for feat in features])
        eucos_dists.append([spd.euclidean(mcv, feat[channel]) * eu_weight +
                            spd.cosine(mcv, feat[channel]) for feat in features])
 
    return {'eucos': np.array(eucos_dists), 'cosine': np.array(cos_dists), 'euclidean': np.array(eu_dists)}


def compute_train_score_and_mavs_and_dists(train_class_num,trainloader,device,net,arcnet):
    scores = [[] for _ in range(train_class_num)]
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # this must cause error for cifar
            feats = net(inputs)
            outputs = F.linear(F.normalize(feats), F.normalize(arcnet.weight))
            for score, cscore, t in zip(feats, outputs, targets):
                # print(f"torch.argmax(score) is {torch.argmax(score)}, t is {t}")
                if torch.argmax(cscore) == t:
                    scores[t].append(score.unsqueeze(dim=0).unsqueeze(dim=0))
    scores = [torch.cat(x).cpu().numpy() for x in scores]  # (N_c, 1, F) * C
    mavs = np.array([np.mean(x, axis=0) for x in scores])  # (C, 1, F)
    # mavs = F.normalize(arcnet.weight).data.cpu().numpy()
    # mavs = mavs[:, np.newaxis, :]  # (C, 1, F)
    dists = [compute_channel_distances(mcv, score) for mcv, score in zip(mavs, scores)]
    return scores, mavs, dists
