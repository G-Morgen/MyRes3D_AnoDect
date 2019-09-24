import torch
from torch.autograd import Variable
import time
import os
import sys

from utils import *


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))
    train_loader_a  = data_loader[0]
    train_loader_n = data_loader[1]

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    aloss1 = AverageMeter()
    aloss2 = AverageMeter()

    end_time = time.time()
    for i, ((inputs1, targets1),(inputs2,targets2)) in enumerate(zip(train_loader_a,train_loader_n)):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets1 = targets1.cuda()
            targets2 = targets2.cuda()
            inputs1  = inputs1.cuda()
            inputs2  = inputs2.cuda()
        inputs1 = Variable(inputs1)
        inputs2 = Variable(inputs2)

        targets1 = Variable(targets1)
        targets2 = Variable(targets2)


        rec = model(inputs1, score=False)
        outputs = model(torch.cat((inputs1, inputs2), 0), score=True)
        targets = torch.cat((targets1, targets2), 0)

        loss1 = criterion[0](outputs, targets) 
        loss2 = criterion[1](rec, inputs1) 

        alpha = 1
        beta  = 0.04 #0.04

        loss = loss1 * alpha + loss2 * beta 
        
        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,2))

        losses.update(loss.data, inputs1.size(0))
        aloss1.update(loss1.data, inputs1.size(0))
        aloss2.update(loss2.data, inputs1.size(0))
        top1.update(prec1, inputs1.size(0))
        top5.update(prec5, inputs1.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(train_loader_a) + (i + 1),
            'loss': losses.val.item(),
            'loss1': aloss1.val.item(),
            'loss2': aloss2.val.item(),
            'prec1': top1.val.item(),
            #'prec5': top5.val.item(),
            'lr': optimizer.param_groups[0]['lr']
        })

        if i % 10 ==0:
            print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t'
                  'Loss2 {loss2.val:.4f} ({loss2.avg:.4f})\t'
                  'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                  #'Prec@5 {top5.val:.1f} ({top5.avg:.1f})'
                  .format(
                      epoch,
                      i,
                      len(train_loader_a),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      loss1=aloss1,
                      loss2=aloss2,
                      top1=top1,
                      #top5=top5,
                      lr=optimizer.param_groups[0]['lr']))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg.item(),
        'loss1': aloss1.val.item(),
        'loss2': aloss2.val.item(),
        'prec1': top1.avg.item(),
        #'prec5': top5.avg.item(),
        'lr': optimizer.param_groups[0]['lr']
    })

    #if epoch % opt.checkpoint == 0:
    #    save_file_path = os.path.join(opt.result_path,
    #                                  'save_{}.pth'.format(epoch))
    #    states = {
    #        'epoch': epoch + 1,
    #        'arch': opt.arch,
    #        'state_dict': model.state_dict(),
    #        'optimizer': optimizer.state_dict(),
    #    }
    #    torch.save(states, save_file_path)
