import os, sys
import time, random
import torch
import torch.nn as nn
import argparse
import importlib
import numpy as np
import models, dataloaders
from torch.utils.data import DataLoader
from utils.metric import accuracy, AverageMeter, Timer

'''
python -u test_generator.py --gpuid 1 --gen_model_type generator --gen_model_name CIFAR_GEN --task_id 1 \
                            --dataset CIFAR100 --optimizer SGD --lr 0.1 --momentum 0.9 --weight_decay 0.0002 \
                            --schedule 30 50 90 100 --schedule_type decay --batch_size 128 \
                            --seed 31 --train_aug --model_type resnet --model_name resnet32  \
                            --log_dir '../outputs/DreamingCL/DFCIL-fourtask/debug-max-task-1/CIFAR100' \
                            --first_split_size 25 --other_split_size 25
'''

def create_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                         help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--log_dir', type=str, default="../outputs/DreamingCL/DFCIL-fivetask/CIFAR100",
                         help="Save experiments results in dir for future plotting!")
    parser.add_argument('--dataroot', type=str, default='data', help="The root folder of dataset or downloaded data")
    parser.add_argument('--first_split_size', type=int, default=20)
    parser.add_argument('--other_split_size', type=int, default=20)   
    parser.add_argument('--gen_model_type', type=str, default='mlp', help="The type (mlp|lenet|vgg|resnet) of generator network")
    parser.add_argument('--gen_model_name', type=str, default='MLP', help="The name of actual model for the generator")
    parser.add_argument('--model_type', type=str, default='mlp', help="The type (mlp|lenet|vgg|resnet) of backbone network")
    parser.add_argument('--model_name', type=str, default='MLP', help="The name of actual model for the backbone")
    parser.add_argument('--task_id', type=int, default=1)
    parser.add_argument('--repeat_id', type=int, default=1)
    parser.add_argument('--train_aug', dest='train_aug', default=False, action='store_true',
                        help="Allow data augmentation during training")
    parser.add_argument('--validation', default=False, action='store_true', help='Evaluate on fold of training dataset rather than testing data')

    
    parser.add_argument('--dataset', type=str, default='MNIST', help="CIFAR10|MNIST")
    parser.add_argument('--optimizer', type=str, default='SGD', help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--schedule', nargs="+", type=int, default=[2],
                        help="The list of epoch numbers to reduce learning rate by factor of 0.1. Last number is the end epoch")
    parser.add_argument('--schedule_type', type=str, default='decay',
                        help="decay")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ReBN', default=False, action='store_true',
                        help="Replace norm BN with ReBN")
    return parser

def load_model(model, filename):
    model.load_state_dict(torch.load(filename))
    print('=> Load Done')
    model = model.cuda()
    model.eval()
    for param in model.parameters(): param.requires_grad = False

def accumulate_acc(output, target, task, meter, topk):
    meter.update(accuracy(output, target, topk), len(target))
    return meter

def learn_batch(args, model, generator, pretrained_model, train_loader, val_loader=None):
    # trains
    # Reset optimizer before learning each task 
    optimizer_arg = {'params':model.parameters(),
                    'lr':args.lr,
                    'weight_decay':args.weight_decay}
    optimizer = torch.optim.__dict__[args.optimizer](**optimizer_arg)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)

    # Evaluate the performance of current task
    print('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=0,total=args.schedule[-1]))
    if val_loader is not None:
        validation(val_loader, model)

    losses = AverageMeter()
    acc = AverageMeter()
    batch_time = AverageMeter()
    batch_timer = Timer()
    class_idx = np.arange(20*args.task_id)
    val_losses = []
    val_accs = []
    for epoch in range(args.schedule[-1]):
        
        if epoch > 0: scheduler.step()
        for param_group in optimizer.param_groups:
            print('LR:', param_group['lr'])
        batch_timer.tic()
        for x,y,task  in train_loader:

            # verify in train mode
            model.train()

            # get data from generator
            x_replay = generator.sample(len(x))
            with torch.no_grad():
                y_hat = pretrained_model.forward(x_replay)
            y_hat = y_hat[:, class_idx]

            # get predicted class-labels (indexed according to each class' position in [self.class_idx]!)
            _, y_replay = torch.max(y_hat, dim=1)
            
            # model update
            logits = model.forward(x_replay)[:, :20*args.task_id]
            loss = nn.CrossEntropyLoss()(logits, y_replay.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(batch_timer.toc())  
            batch_timer.tic()
            
            # measure accuracy and record loss
            y_replay = y_replay.detach()
            accumulate_acc(logits, y_replay, task, acc, topk=(1,))
            losses.update(loss,  y_replay.size(0)) 
            batch_timer.tic()

        # eval update
        print('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=epoch+1,total=args.schedule[-1]))
        print(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses,acc=acc))
        val_losses.append(losses.avg)
        val_accs.append(acc.avg)
        # Evaluate the performance of current task
        if val_loader is not None:
            validation(val_loader, model)

        # reset
        losses = AverageMeter()
        acc = AverageMeter()
            
    model.eval()

    # Compare to acc of pretrained_model
    validation(val_loader, pretrained_model)

    try:
        return batch_time.avg, val_losses, val_accs
    except:
        return None

def validation(dataloader, model, task_in = None,  verbal = True):

    # This function doesn't distinguish tasks.
    batch_timer = Timer()
    acc = AverageMeter()
    batch_timer.tic()

    orig_mode = model.training
    model.eval()

    for i, (input, target, task) in enumerate(dataloader):
        with torch.no_grad():
            input = input.cuda()
            target = target.cuda()
        if task_in is None:
            output = model.forward(input)[:, :20*args.task_id]
            acc = accumulate_acc(output, target, task, acc, topk=(1,))
        else:
            mask = target >= task_in[0]
            mask_ind = mask.nonzero().view(-1) 
            input, target = input[mask_ind], target[mask_ind]

            mask = target < task_in[-1]
            mask_ind = mask.nonzero().view(-1) 
            input, target = input[mask_ind], target[mask_ind]
            
            if len(target) > 1:
                output = model.forward(input)[:, task_in]
                acc = accumulate_acc(output, target-task_in[0], task, acc, topk=(1,))
        
    model.train(orig_mode)

    if verbal:
        print(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                .format(acc=acc, time=batch_timer.toc()))
    return acc.avg

if __name__ == '__main__':
    parser=create_args()
    argv = sys.argv[1:]
    args = parser.parse_args(argv)

    torch.backends.cudnn.deterministic=True
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # create dataset to test according to task-id and repeat-id
    if args.dataset == 'CIFAR100':
        Dataset = dataloaders.iCIFAR100
        num_classes = 100
        dataset_size = [32,32,3]
    class_order = np.arange(num_classes).tolist()
    print('=============================================')
    print('Shuffling....')
    print('pre-shuffle:' + str(class_order))
    random.seed(seed)
    random.shuffle(class_order)
    print('post-shuffle:' + str(class_order))
    print('=============================================')
    tasks = []
    p = 0
    while p < num_classes:
        inc = args.other_split_size if p > 0 else args.first_split_size
        tasks.append(class_order[p:p+inc])
        p += inc

    train_transform = dataloaders.utils.get_transform(dataset=args.dataset, phase='train', aug=args.train_aug, dgr=False)
    test_transform  = dataloaders.utils.get_transform(dataset=args.dataset, phase='test', aug=args.train_aug, dgr=False)
    train_dataset = Dataset(args.dataroot, train=True, tasks=tasks,
                        download_flag=True, transform=train_transform, 
                        seed=seed, validation=args.validation)
    test_dataset  = Dataset(args.dataroot, train=False, tasks=tasks,
                            download_flag=False, transform=test_transform, 
                            seed=seed, validation=args.validation)

    if args.ReBN: method = 'abd-rebn'
    else: method = 'abd-no-rebn'
    saved_models_folder = args.log_dir + '/' + method + '/models/repeat-' + str(args.seed+1) + '/task-' + str(args.task_id) 
    saved_path_loss = 'outputs/loss-' + method + '-' + args.dataset + args.first_split_size + '-' + args.other_split_size + '-' + str(seed) + '-' + args.model_name + '.txt'
    saved_path_acc  = 'outputs/acc-' + method + '-' + args.dataset + args.first_split_size + '-' + args.other_split_size + '-' + str(seed) + '-' + args.model_name + '.txt'
    

    # Generator
    generator = models.__dict__[args.gen_model_type].__dict__[args.gen_model_name]()
    load_model(generator,  saved_models_folder + '/generator.pth')
    print('Generator: ', sum(p.numel() for p in generator.parameters()))
    print(generator)

    # Pretrained model
    pretrained_model = models.__dict__[args.model_type].__dict__[args.model_name](out_dim=num_classes, ReBN=args.ReBN)
    load_model(pretrained_model, saved_models_folder + '/class.pth')
    print('Model:', sum(p.numel() for p in pretrained_model.parameters()))
    print(pretrained_model)

    # New model     
    model = models.__dict__[args.model_type].__dict__[args.model_name](out_dim=num_classes, ReBN=args.ReBN)
    model = model.cuda()

    # Train
    train_dataset.load_dataset(args.task_id-1, train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
    test_dataset.load_dataset(args.task_id-1, train=False)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=2)
    avg_train_time, val_losses, val_accs = learn_batch(args, model, generator, pretrained_model, train_loader, test_loader)

    # write to file
    with open(saved_path_loss, 'w') as f:
        f.write("\n".join([str(i) for i in val_losses]))
    with open(saved_path_acc, 'w') as f:
        f.write("\n".join([str(i) for i in val_accs]))

    








