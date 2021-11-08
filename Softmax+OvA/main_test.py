import random
import torch.backends.cudnn as cudnn
import torch.utils.data
from data_loader import get_sigloader_fromlist
import net
import numpy as np
from evaluation import Evaluation
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F

device = torch.device('cuda:0')

data_path = '../adsb-107loaded_2.mat'
save_path0 = 'results'
model_root0 = 'weights'
cudnn.benchmark = True
class_num = 54
feat_dim = 64
test_batch_size = 250
plot_num = 10000
drop_out= 0.5
repeatn = 10

threshold = 0.9

def test(testloader):
    my_net.eval()

    scores, labels, feats = [], [], []
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            feat, outputs = my_net(inputs)
            feats.append(feat)
            scores.append(torch.sigmoid(outputs))
            labels.append(targets)

    feats = torch.cat(feats,dim=0).cpu().numpy()
    scores = torch.cat(scores,dim=0).cpu().numpy()
    labels = torch.cat(labels,dim=0).cpu().numpy()
    scores = np.array(scores)
    labels = np.array(labels)

    score0, pred = [], []
    for score in scores:
        pred.append(np.argmax(score) if np.max(score) >= threshold else class_num)
        score0.append(score)

    eval0 = Evaluation(pred, labels)

    return eval0, feats, pred, labels

if __name__ == '__main__':

    for rpi in range(repeatn):

        save_path = str(rpi)+'/'+save_path0
        model_root = str(rpi)+'/'+model_root0

        manual_seed = random.randint(1, 10000)
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)

        _, dataset_test = get_sigloader_fromlist(data_path, class_num, save_path)

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
        
        test_ova, feats, preds, labels = test(dataloader_test)
        print('ova f1: {:.4f}'.format(test_ova.f1_macro))

        torch.save(test_ova, save_path+'/test_ova.pkl')
        np.savetxt(save_path+'/preds.txt', preds)
        np.savetxt(save_path+'/labels.txt', labels)

        idx = np.random.permutation(len(labels))
        idx = idx[:plot_num]
        feats = feats[idx,:]
        labels = labels[idx]

        # print('TSNE begins ...')
        # feat2 = TSNE(n_components=2, learning_rate=100, metric='cosine').fit_transform(feats)

        # cmap = plt.get_cmap('brg')

        # cn = float(class_num)
        # plt.figure(figsize=[5,5], dpi=600)
        # plt.scatter(feat2[labels!=class_num,0], feat2[labels!=class_num,1], c=cmap(labels[labels!=class_num]/cn), s=1)
        # plt.scatter(feat2[labels==class_num,0], feat2[labels==class_num,1], c='black', s=1)
        # plt.savefig(save_path+'/feat_true.svg', format='svg')
        # plt.clf()

        # # plt.figure(figsize=[5,5], dpi=500)
        # predict = np.array(test_ova.predict)
        # predict = predict[idx]
        # plt.scatter(feat2[predict!=class_num,0], feat2[predict!=class_num,1], c=cmap(predict[predict!=class_num]/cn), s=1)
        # plt.scatter(feat2[predict==class_num,0], feat2[predict==class_num,1], c='black', s=1)
        # plt.savefig(save_path+'/feat_ova.svg', format='svg')
        # plt.clf()
    
    print('done')