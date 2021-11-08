import random
import torch.backends.cudnn as cudnn
import torch.utils.data
from data_loader import get_sigloader_fromlist
import net
import numpy as np
from evaluation import Evaluation
from openmax import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F
from metrics import ArcMarginProduct

device = torch.device('cuda:0')

data_path = '../adsb-107loaded_2.mat'
save_path0 = 'results'
model_root0 = 'weights'
cudnn.benchmark = True
class_num = 43
feat_dim = 64
batch_size = 64
test_batch_size = 250
plot_num = 10000
drop_out= 0.5
repeatn = 10

weibull_tail = 0.1
weibull_alpha = 1
weibull_threshold = 0.5

def test(trainloader, testloader):
    my_net.eval()
    lossf.eval()

    _, mavs, dists = compute_train_score_and_mavs_and_dists(class_num, trainloader, device, my_net, lossf)
    categories = list(range(0, class_num))
    weibull_model = fit_weibull_adapt(mavs, dists, categories, weibull_tail, "cosine")

    cdists = [dists[i]['cosine'][0] for i in range(class_num)]
    train_labels = [[i]*len(dists[i]['cosine'][0]) for i in range(class_num)]
    cdists = np.concatenate(cdists, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    scores, distances, cscores, labels = [], [], [], []
    weight = torch.tensor(np.squeeze(mavs)).to(device)
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            feats = my_net(inputs)
            dist = F.linear(F.normalize(feats), F.normalize(weight))
            outputs = F.softmax(dist*lossf.s, dim=1)
            scores.append(feats)
            distances.append(dist)
            cscores.append(outputs)
            labels.append(targets)

    scores = torch.cat(scores,dim=0).cpu().numpy()
    distances = (1.-torch.cat(distances,dim=0).cpu().numpy())/2.
    cscores = torch.cat(cscores,dim=0).cpu().numpy()
    labels = torch.cat(labels,dim=0).cpu().numpy()
    scores = np.array(scores)[:, np.newaxis, :]
    cscores = np.array(cscores)[:, np.newaxis, :]
    labels = np.array(labels)

    pred_softmax, pred_softmax_threshold, pred_openmax = [], [], []
    score_softmax, score_openmax = [], []
    for score, cscore in zip(scores, cscores):
        so, ss = openmax(weibull_model, categories, score, cscore,
                         1.0, weibull_alpha, "cosine")
        pred_softmax.append(np.argmax(ss))
        pred_softmax_threshold.append(np.argmax(ss) if np.max(ss) >= weibull_threshold else class_num)
        pred_openmax.append(np.argmax(so))

        score_softmax.append(ss)
        score_openmax.append(so)

    eval_softmax = Evaluation(pred_softmax, labels)
    eval_softmax_threshold = Evaluation(pred_softmax_threshold, labels)
    eval_openmax = Evaluation(pred_openmax, labels)

    return eval_softmax, eval_softmax_threshold, eval_openmax, np.squeeze(scores), np.squeeze(mavs), cdists, train_labels, np.squeeze(distances), labels

if __name__ == '__main__':

    for rpi in range(repeatn):

        save_path = str(rpi)+'/'+save_path0
        model_root = str(rpi)+'/'+model_root0

        manual_seed = random.randint(1, 10000)
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)

        dataset_train, dataset_test = get_sigloader_fromlist(data_path, class_num, save_path)

        dataloader_train = torch.utils.data.DataLoader(
            dataset = dataset_train,
            batch_size = batch_size,
            shuffle = True,
            drop_last = False,
            num_workers = 0,
            # pin_memory = True
        )

        dataloader_test = torch.utils.data.DataLoader(
            dataset = dataset_test,
            batch_size = test_batch_size,
            shuffle = False,
            drop_last = False,
            num_workers = 0,
            # pin_memory = True
        )

        print('loading model...')

        my_net = net.Net(class_num=class_num, feat_dim=feat_dim, dropout=drop_out).to(device)
        my_net.load_state_dict(torch.load(model_root+'/model_epoch_99.pth'))
        lossf = ArcMarginProduct(feat_dim, class_num, s=3., m=1.0).to(device)
        lossf.load_state_dict(torch.load(model_root+'/arcnet_epoch_99.pth'))
        
        test_softmax, test_softmax_thres, test_openmax, feats, means, train_dists, train_labels, dists, labels = test(dataloader_train, dataloader_test)
        print('softmax f1: {:.4f} softmax_thres f1: {:.4f} openmax f1: {:.4f}'.format(test_softmax.f1_macro, 
            test_softmax_thres.f1_macro, test_openmax.f1_macro))

        torch.save(test_softmax, save_path+'/test_softmax.pkl')
        torch.save(test_softmax_thres, save_path+'/test_softmax_thres.pkl')
        torch.save(test_openmax, save_path+'/test_openmax.pkl')
        np.savetxt(save_path+'/train_dists.txt', train_dists)
        np.savetxt(save_path+'/test_dists.txt', dists)
        np.savetxt(save_path+'/train_labels.txt', train_labels)
        np.savetxt(save_path+'/test_labels.txt', labels)

        idx = np.random.permutation(len(labels))
        idx = idx[:plot_num]
        feats = feats[idx,:]
        labels = labels[idx]
        feats = np.concatenate([means, feats], axis=0)

        print('TSNE begins ...')
        feat2 = TSNE(n_components=2, learning_rate=100, metric='cosine').fit_transform(feats)
        means2 = feat2[:class_num,:]
        feat2 = feat2[class_num:,:]
        means2labels = np.arange(class_num)

        cmap = plt.get_cmap('brg')

        cn = float(class_num)
        plt.figure(figsize=[5,5], dpi=600)
        plt.scatter(feat2[labels!=class_num,0], feat2[labels!=class_num,1], c=cmap(labels[labels!=class_num]/cn), s=1)
        plt.scatter(feat2[labels==class_num,0], feat2[labels==class_num,1], c='black', s=1)
        plt.savefig(save_path+'/feat_true.svg', format='svg')
        plt.clf()

        # plt.figure(figsize=[5,5], dpi=500)
        predict = np.array(test_softmax.predict)
        predict = predict[idx]
        plt.scatter(feat2[predict!=class_num,0], feat2[predict!=class_num,1], c=cmap(predict[predict!=class_num]/cn), s=1)
        plt.scatter(feat2[predict==class_num,0], feat2[predict==class_num,1], c='black', s=1)
        plt.savefig(save_path+'/feat_softmax.svg', format='svg')
        plt.clf()

        # plt.figure(figsize=[5,5], dpi=500)
        predict = np.array(test_softmax_thres.predict)
        predict = predict[idx]
        plt.scatter(feat2[predict!=class_num,0], feat2[predict!=class_num,1], c=cmap(predict[predict!=class_num]/cn), s=1)
        plt.scatter(feat2[predict==class_num,0], feat2[predict==class_num,1], c='black', s=1)
        plt.savefig(save_path+'/feat_softmax_thres.svg', format='svg')
        plt.clf()

        # plt.figure(figsize=[5,5], dpi=500)
        predict = np.array(test_openmax.predict)
        predict = predict[idx]
        plt.scatter(feat2[predict!=class_num,0], feat2[predict!=class_num,1], c=cmap(predict[predict!=class_num]/cn), s=1)
        plt.scatter(feat2[predict==class_num,0], feat2[predict==class_num,1], c='black', s=1)
        plt.savefig(save_path+'/feat_openmax.svg', format='svg')
        plt.clf()
    
    print('done')