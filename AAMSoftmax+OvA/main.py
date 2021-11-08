import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from data_loader import get_sigloader
from net import Net
import numpy as np
from metrics import ArcMarginProduct
from loss import BCEOvA

device = torch.device('cuda:0')

data_path = '../adsb-107loaded_2.mat'
save_path0 = 'results'
model_root0 = 'weights'
cudnn.benchmark = True
class_num = 43
feat_dim = 64
lr = 1e-4
batch_size = 128
n_epoch = 100
save_epoch = 100
log_step = 100
repeatn = 10
drop_out = 0.5

if __name__ == '__main__':


    for rpi in range(repeatn):

        print('Experiment '+str(rpi)+' start:')

        save_path = str(rpi)+'/'+save_path0
        model_root = str(rpi)+'/'+model_root0

        if not os.path.exists(str(rpi)):
            os.mkdir(str(rpi))
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

        lossf = ArcMarginProduct(feat_dim, class_num, s=3., m=1.0).to(device)
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

                feat, pred = my_net(img)
                class_output = lossf(feat, label)
                err_label = loss_class(class_output, label) + BCEOvA(pred, label)

                optimizer.zero_grad()
                err_label.backward()
                optimizer.step()

                pred = pred.data.max(1, keepdim=True)[1]
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
        
    print('done')