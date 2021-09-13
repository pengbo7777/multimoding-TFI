from __future__ import print_function

import matplotlib.pyplot as plot
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from option import Options
from utils import *
import sys
import os
# from tensorboardX import SummaryWriter
import numpy as np
# from save_log import *

# global variable
best_pred = 100.0
errlist_train = []
errlist_val = []


def adjust_learning_rate(optimizer, args, epoch, best_pred):
    lr = args.lr * (0.5 ** ((epoch - 1) // args.lr_decay))
    if (epoch - 1) % args.lr_decay == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    # init the args
    global best_pred, errlist_train, errlist_val
    args = Options().parse()
    # writer = SummaryWriter(log_dir='./log/' + str(args.dataset) + "/" + args.boarddir,
    #                        comment="Fabric_regression")
    # make_print_to_file('./log/' + args.boarddir)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args.cuda)
    torch.manual_seed(args.seed)
    # plot 
    if args.plot:
        print('=>Enabling matplotlib for display:')
        plot.ion()
        plot.show()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # init dataloader
    dataset = importlib.import_module('dataloader.' + args.dataset)
    Dataloder = dataset.Dataloder
    train_loader, test_loader = Dataloder(args).getloader()
    # init the model
    models = importlib.import_module('model.' + args.model)
    model = models.Net(47)
    # model = nn.DataParallel(model, device_ids=[1])
    model = torch.nn.DataParallel(model)

    if args.pretrain is not None:
        # checkpoint = torch.load(args.pretrain)

        # model = torch.nn.DataParallel(model)
        model.load_state_dict({k: v for k, v in
                               torch.load(
                                   args.pretrain)[
                                   'state_dict'].items()})
        # load params
        # model.load_state_dict(new_state_dict)
        # model = nn.DataParallel(model, device_ids=[0, 1])
        # model.load_state_dict(checkpoint['state_dict'])
        # model = model.load_state_dict(torch.load(args.pretrain))
    print(model)
    # criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    # criterion = MultiFocalLoss(3, [21 / 164, 118 / 164, 25 / 164])
    # criterion = FocalLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2, last_epoch=-1)
    if args.cuda:
        model.cuda()
        # Please use CUDA_VISIBLE_DEVICES to control the number of gpus
        # model = torch.nn.DataParallel(model)
    """
    optim.SGD(model.parameters(), lr=args.lr, momentum=
            args.momentum, weight_decay=args.weight_decay)
    """
    # check point
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            best_pred = checkpoint['best_pred']
            errlist_train = checkpoint['errlist_train']
            errlist_val = checkpoint['errlist_val']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no resume checkpoint found at '{}'". \
                  format(args.resume))

    def train(epoch):
        model.train()
        global best_pred, errlist_train
        train_loss, correct, total = 0, 0, 0
        adjust_learning_rate(optimizer, args, epoch, best_pred)
        for batch_idx, (data, target) in enumerate(train_loader):

            # scheduler(optimizer, batch_idx, epoch, best_pred)
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output, _ = model(data)
            target = target.long()
            output = output.float()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()
            total += target.size(0)
            err = 100 - 100. * correct / total
            progress_bar(batch_idx, len(train_loader),
                         'Loss: %.3f | Err: %.3f%% (%d/%d)' % \
                         (train_loss / (batch_idx + 1),
                          err, total - correct, total))
        errlist_train += [err]
        # print('Train Epoch: {}\t Loss: {:.6f}'.format(epoch, train_loss))
        print('\n' + args.model + ' Train set, Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(dataset.train_data),
            100. * correct / len(dataset.train_data)))
        writer.add_scalar('train_loss', train_loss / (batch_idx + 1), epoch)

    def save_res(pred_list, target_list, epoch, dictionary=None):
        base_path = "/home/pengbo/workspace/ResNeSt_DEP/result/" + str(args.dataset) + "/" + args.boarddir
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        file = open(base_path + "/" + str(epoch) + '.txt', 'w')
        file.write(str(pred_list) + "\n")
        file.write(str(target_list) + "\n")
        if dictionary != None:
            file.write(str(dictionary) + "\n")
        file.close()

    def test(epoch):
        model.eval()
        global best_pred, errlist_train, errlist_val
        test_loss, correct, total = 0, 0, 0
        is_best = False
        pred_list = []
        target_list = []
        dictionary_list = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                # output = model(data)
                output, _ = model(data)
                target = target.long()
                test_loss += criterion(output, target).item()
                # get the index of the max log-probability
                pred = output.data.max(1)[1]
                # 记录结果
                pred_list += pred.cpu().numpy().tolist()
                target_list += target.cpu().numpy().tolist()
                # if 'DEP' in args.model:
                #     dictionary_list += _.cpu().numpy().tolist()
                # end
                correct += pred.eq(target.data).cpu().sum()
                total += target.size(0)

                err = 100 - 100. * correct / total
                progress_bar(batch_idx, len(test_loader),
                             'Loss: %.3f | Err: %.3f%% (%d/%d)' % \
                             (test_loss / (batch_idx + 1),
                              err, total - correct, total))
            writer.add_scalar('test_loss', (test_loss / (batch_idx + 1)), epoch)
            writer.add_scalar('test_accuracy', 100. * correct / len(dataset.test_data), epoch)
            save_res(pred_list, target_list, epoch, dictionary_list)

        if args.eval:
            print('Error rate is %.3f' % err)
            return
        # save checkpoint
        errlist_val += [err]
        if err < best_pred:
            best_pred = err
            is_best = True
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_pred': best_pred,
            'errlist_train': errlist_train,
            'errlist_val': errlist_val,
        }, args=args, is_best=is_best)

    # def show_dictionary():
    #     model.eval()
    #     global best_pred, errlist_train, errlist_val
    #     test_loss, correct, total = 0, 0, 0
    #     is_best = False
    #     pred_list = []
    #     target_list = []
    #     with torch.no_grad():
    #         for batch_idx, (data, target) in enumerate(test_loader):
    #             if args.cuda:
    #                 data, target = data.cuda(), target.cuda()
    #             data, target = Variable(data), Variable(target)
    #             # output = model(data)
    #             output, dictionary = model(data)
    #             target = target.long()
    #             test_loss += criterion(output, target).item()
    #             # get the index of the max log-probability
    #             pred = output.data.max(1)[1]
    #             dictionary.cpu().numpy()

    if args.eval:
        test(args.start_epoch)
        return

    for epoch in range(args.start_epoch, args.epochs + 1):
        print('Epoch:', epoch)
        train(epoch)
        test(epoch)

        # save train_val curve to a file
        if args.plot:
            plot.clf()
            plot.xlabel('Epoches: ')
            plot.ylabel('Error Rate: %')
            plot.plot(errlist_train, label='train')
            plot.plot(errlist_val, label='val')
            plot.legend(loc='upper left')
            plot.savefig("runs/%s/%s/%s" % (args.dataset, args.model, args.checkname)
                         + 'train_val_l2_1e-2.jpg')
            plot.draw()
            plot.pause(0.001)

    def __test():
        test_loss, correct, total = 0, 0, 0
        is_best = False
        pred_list = []
        target_list = []
        dictionary_list = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                # output = model(data)
                output, _ = model(data)
                target = target.long()
                test_loss += criterion(output, target).item()
                # get the index of the max log-probability
                pred = output.data.max(1)[1]
                # 记录结果
                pred_list += pred.cpu().numpy().tolist()
                target_list += target.cpu().numpy().tolist()
                # if 'DEP' in args.model:
                dictionary_list += _.cpu().numpy().tolist()
                # end
                correct += pred.eq(target.data).cpu().sum()
                total += target.size(0)

                err = 100 - 100. * correct / total
                progress_bar(batch_idx, len(test_loader),
                             'Loss: %.3f | Err: %.3f%% (%d/%d)' % \
                             (test_loss / (batch_idx + 1),
                              err, total - correct, total))
            # writer.add_scalar('test_loss', (test_loss / (batch_idx + 1)), epoch)
            # writer.add_scalar('test_accuracy', 100. * correct / len(dataset.test_data), epoch)
            save_res(pred_list, target_list, epoch, dictionary_list)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class MultiFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


if __name__ == "__main__":
    main()
