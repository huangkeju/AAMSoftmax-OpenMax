import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from data_loader import get_sigloader, get_sigloader_fromlist
from net import Net
import numpy as np
from openmax import *
from metrics import ArcMarginProduct
from evaluation import Evaluation

device = torch.device('cuda:0')

data_path = '../adsb-107loaded_2.mat'
save_path0 = 'results'
model_root0 = 'weights'
cudnn.benchmark = True
class_num = 21
feat_dim = 64
lr = 1e-4
batch_size = 128
test_batch_size = 250
n_epoch = 100
save_epoch = 100
log_step = 100
repeatn = 5
drop_out = 0.5

weibull_tail = 0.1
weibull_alpha = 1
weibull_threshold = 0.5

s_list = np.arange(5., 6., 1.0)
m_list = np.arange(1.3, 1.4, 0.1)

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

    for s in s_list:
        for m in m_list:

            head_dir = 's='+"{:.2f}".format(s)+'_m='+"{:.2f}".format(m)+'/'

            for rpi in range(repeatn):

                print('Experiment '+str(rpi)+' start:')

                save_path = head_dir+str(rpi)+'/'+save_path0
                model_root = head_dir+str(rpi)+'/'+model_root0

                if not os.path.exists(head_dir):
                    os.mkdir(head_dir)
                if not os.path.exists(head_dir+str(rpi)):
                    os.mkdir(head_dir+str(rpi))
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                if not os.path.exists(model_root):
                    os.mkdir(model_root)

                manual_seed = random.randint(1, 10000)
                random.seed(manual_seed)
                torch.manual_seed(manual_seed)

                dataset_train, dataset_test = get_sigloader(data_path, class_num, save_path)

                dataloader_train = torch.utils.data.DataLoader(
                    dataset = dataset_train,
                    batch_size = batch_size,
                    shuffle = True,
                    drop_last = True,
                    num_workers = 0,
                )

                my_net = Net(class_num, feat_dim, drop_out).to(device)
                # print(my_net)
                trainable_num = sum(p.numel() for p in my_net.parameters() if p.requires_grad)
                # print('number of trainable parameters:', trainable_num)

                lossf = ArcMarginProduct(feat_dim, class_num, s=s, m=m).to(device)
                loss_class = torch.nn.CrossEntropyLoss().to(device)

                optimizer = optim.Adam([{'params': my_net.parameters()}, {'params': lossf.parameters()}], lr=lr)

                acc_train = np.zeros([n_epoch,])
                loss_train = np.zeros([n_epoch,])

                for epoch in range(n_epoch):

                    train_iter = iter(dataloader_train)

                    n_train_correct = 0
                    loss_train_sum = 0.

                    my_net.train()
                    lossf.train()

                    i = 0
                    while i < len(dataloader_train):


                        img, label = train_iter.next()
                        img, label = img.float().to(device), label.to(device)

                        batch_size = len(label)

                        feat = my_net(img)
                        class_output = lossf(feat, label)
                        err_label = loss_class(class_output, label)

                        optimizer.zero_grad()
                        err_label.backward()
                        optimizer.step()

                        pred = class_output.data.max(1, keepdim=True)[1]
                        n_train_correct += pred.eq(label.data.view_as(pred)).cpu().sum()
                        loss_train_sum += float(err_label)

                        i += 1
                        if (i % log_step == 0):
                            print('Epoch [{}/{}] Step [{}/{}]: loss class={:.5f}'.format(
                                epoch, n_epoch, i, len(dataloader_train), err_label))

                    acc_train[epoch] = float(n_train_correct)/float(len(dataset_train))
                    loss_train[epoch] = loss_train_sum/float(len(dataloader_train))

                    np.savetxt(save_path+'/acc_train.txt', acc_train)
                    np.savetxt(save_path+'/loss_train.txt', loss_train)

                    if ((epoch+1) % save_epoch == 0):
                        torch.save(my_net.state_dict(), '{0}/model_epoch_{1}.pth'.format(model_root, epoch))
                        torch.save(lossf.state_dict(), '{0}/arcnet_epoch_{1}.pth'.format(model_root, epoch))
                        
                    print('epoch: {} train acc: {:.4f} train loss: {:.4f}'.format(epoch, acc_train[epoch], loss_train[epoch]))
                
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

                my_net = Net(class_num=class_num, feat_dim=feat_dim, dropout=drop_out).to(device)
                my_net.load_state_dict(torch.load(model_root+'/model_epoch_99.pth'))
                lossf = ArcMarginProduct(feat_dim, class_num, s=s, m=m).to(device)
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
            
    print('done')