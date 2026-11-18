import random
from pathlib import Path
from data_utils import TASK_CLASSES, TrainPreprocessor
from ANMdataloder import ANM
from ADNIdataloder import ADNI
import copy
from torch.utils.data import DataLoader
from models import AVClassifier
from sklearn import metrics
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import re
from min_norm_solvers import MinNormSolver
import numpy as np
from tqdm import tqdm
import argparse
import os
import pickle
from operator import mod
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from losses import SupConLoss

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=os.environ.get("CAPPM_DATA_ROOT", str(Path(__file__).resolve().parents[1] / "processed_data")), help="External directory containing ANM/overlap, ADNI/overlap, etc.")
    parser.add_argument('--dataset', required=True, type=str,
                        help='ANM, ADNI,ANM_early,ADNI_early')
    parser.add_argument('--model', default='model', type=str)
    parser.add_argument('--n_classes', default=None, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--optimizer', default='adam',
                        type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.002, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=30, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1,
                        type=float, help='decay coefficient')
    parser.add_argument('--ckpt_path', default='log_cd',
                        type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true',
                        help='turn on train mode')
    parser.add_argument('--clip_grad', action='store_true',
                        help='turn on train mode')
    parser.add_argument('--use_tensorboard', default=True,
                        type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', default='log_cd',
                        type=str, help='path to save tensorboard logs')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='0',
                        type=str, help='GPU ids')
    parser.add_argument('--use_conflict', action='store_true',
                         help='whether to conflict')
    parser.add_argument('--use_coatt', action='store_true',
                         help='whether to conflict')
    parser.add_argument('--gamma', default=1.5, type=float)
    parser.add_argument('--use_healnet', action='store_true',
                         help='whether to conflict')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0,1,2,3,4])
    parser.add_argument('--lr_schedule', choices=['constant','step'], default='constant')
    args = parser.parse_args()
    if args.dataset not in TASK_CLASSES:
        parser.error('Choose one of: ' + ', '.join(TASK_CLASSES))
    expected = TASK_CLASSES[args.dataset]
    if args.n_classes is not None and args.n_classes != expected:
        parser.error('n_classes does not match dataset')
    args.n_classes = expected
    if args.use_coatt:
        parser.error('use_coatt is not implemented in the supplied model; use use_healnet for CAPPM')
    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler,class_weights, writer=None):
    criterion = nn.CrossEntropyLoss(class_weights)

    model.train()

    _loss = 0
    correct_mm = 0
    correct_a = 0
    correct_v = 0
    total = 0

    loss_value_mm=[]
    loss_value_a=[]
    loss_value_v=[]

    record_names_audio = []
    record_names_visual = []
    for name, param in model.named_parameters():
        if 'head' in name: 
            continue
        if ('audio' in name):
            record_names_audio.append((name, param))
            continue
        if ('visual' in name):
            record_names_visual.append((name, param))
            continue


    this_cos_audio_train=[]
    this_cos_visual_train=[]
    for step, (spec, images, label) in enumerate(dataloader):

        optimizer.zero_grad()
        images = images.to(device)
        spec = spec.to(device)
        label = label.to(device)
        out,out_a,out_v,_,_, = model(spec.float(), images.float())

        loss_mm = criterion(out, label)

        loss_a=criterion(out_a,label)

        loss_v=criterion(out_v,label)


        loss_value_mm.append(loss_mm.item())
        loss_value_a.append(loss_a.item())
        loss_value_v.append(loss_v.item())

        _, predicted_mm = torch.max(out, 1)
        _, predicted_a = torch.max(out_a, 1)
        _, predicted_v = torch.max(out_v, 1)
        
        total += label.size(0)
        correct_mm += (predicted_mm == label).sum().item()
        correct_a += (predicted_a == label).sum().item()
        correct_v += (predicted_v == label).sum().item()


        losses=[loss_mm,loss_a,loss_v]
        all_loss = ['both','audio', 'visual']


        grads_audio = {}
        grads_visual={}


        for idx, loss_type in enumerate(all_loss):
            loss = losses[idx]
            loss.backward(retain_graph=True)

            if(loss_type=='visual'):
                for tensor_name, param in record_names_visual:
                    if loss_type not in grads_visual.keys():
                        grads_visual[loss_type] = {}
                    grads_visual[loss_type][tensor_name] = param.grad.data.clone()
                grads_visual[loss_type]["concat"] = torch.cat([grads_visual[loss_type][tensor_name].flatten()  for tensor_name, _ in record_names_visual])
            elif(loss_type=='audio'):
                for tensor_name, param in record_names_audio:
                    if loss_type not in grads_audio.keys():
                        grads_audio[loss_type] = {}
                    grads_audio[loss_type][tensor_name] = param.grad.data.clone()
                grads_audio[loss_type]["concat"] = torch.cat([grads_audio[loss_type][tensor_name].flatten()  for tensor_name, _ in record_names_audio])
            else:
                for tensor_name, param in record_names_audio:
                    if loss_type not in grads_audio.keys():
                        grads_audio[loss_type] = {}
                    grads_audio[loss_type][tensor_name] = param.grad.data.clone()
                grads_audio[loss_type]["concat"] = torch.cat([grads_audio[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_audio])
                for tensor_name, param in record_names_visual:
                    if loss_type not in grads_visual.keys():
                        grads_visual[loss_type] = {}
                    grads_visual[loss_type][tensor_name] = param.grad.data.clone()
                grads_visual[loss_type]["concat"] = torch.cat([grads_visual[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_visual])

            optimizer.zero_grad()
        
        this_cos_audio=F.cosine_similarity(grads_audio['both']["concat"],grads_audio['audio']["concat"],dim=0)
        this_cos_visual=F.cosine_similarity(grads_visual['both']["concat"],grads_visual['visual']["concat"],dim=0)
        this_cos_audio_train.append(this_cos_audio.item())
        this_cos_visual_train.append(this_cos_visual.item())

        audio_task=['both','audio']
        visual_task=['both','visual']


        if args.use_conflict:

            audio_k=[0,0]
            visual_k=[0,0]

            if(this_cos_audio>0):
                audio_k[0]=0.5
                audio_k[1]=0.5
            else:
                audio_k, min_norm = MinNormSolver.find_min_norm_element([list(grads_audio[t].values()) for t in audio_task])
            if(this_cos_visual>0):
                visual_k[0]=0.5
                visual_k[1]=0.5
            else:
                visual_k, min_norm = MinNormSolver.find_min_norm_element([list(grads_visual[t].values()) for t in visual_task])



            gamma=args.gamma

            loss=loss_mm+loss_a+loss_v
            loss.backward()


            for name, param in model.named_parameters():
                if param.grad is not None:
                    layer = re.split('[_.]',str(name))
                    if('head' in layer):
                        continue
                    if('audio' in layer):
                        three_norm=torch.norm(param.grad.data.clone())
                        new_grad=2*audio_k[0]*grads_audio['both'][name]+2*audio_k[1]*grads_audio['audio'][name]
                        new_norm=torch.norm(new_grad)
                        diff=three_norm/new_norm
                        if(diff>1):
                            param.grad=diff*new_grad*gamma
                        else:
                            param.grad=new_grad*gamma

                    if('visual' in layer):
                        three_norm=torch.norm(param.grad.data.clone())
                        new_grad=2*visual_k[0]*grads_visual['both'][name]+2*visual_k[1]*grads_visual['visual'][name]
                        new_norm=torch.norm(new_grad)
                        diff=three_norm/new_norm
                        if(diff>1):
                            param.grad=diff*new_grad*gamma
                        else:
                            param.grad=new_grad*gamma
            
        else:
            loss=loss_mm+loss_a+loss_v
            loss.backward()


        optimizer.step()
        _loss += loss.item()

    acc_mm = correct_mm / total
    acc_a = correct_a / total
    acc_v = correct_v / total

    
    return _loss / len(dataloader),acc_mm, acc_a, acc_v,this_cos_audio_train,this_cos_visual_train


def valid(args, model, device, dataloader):

    n_classes = args.n_classes

    predicted_label=[]
    true_label=[]
    with torch.no_grad():
        model.eval()
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a= [0.0 for _ in range(n_classes)]
        acc_v= [0.0 for _ in range(n_classes)]

        for step, (spec, images, label) in enumerate(dataloader):



            spec = spec.to(device)
            images = images.to(device)
            label = label.to(device)

            prediction_all = model(spec.float(), images.float())


            prediction=prediction_all[0]
            prediction_audio=prediction_all[1]
            prediction_visual=prediction_all[2]

            _, predicted = prediction.max(1)
            predicted_label.extend(predicted.tolist())
            true_label.extend(label.tolist())

            for i, item in enumerate(label):

                ma = prediction[i].cpu().data.numpy()
                index_ma = np.argmax(ma)
                num[label[i]] += 1.0
                if index_ma == label[i]:
                    acc[label[i]] += 1.0
                
                ma_audio=prediction_audio[i].cpu().data.numpy()
                index_ma_audio = np.argmax(ma_audio)
                if index_ma_audio == label[i]:
                    acc_a[label[i]] += 1.0


                ma_visual=prediction_visual[i].cpu().data.numpy()
                index_ma_visual = np.argmax(ma_visual)
                if index_ma_visual == label[i]:
                    acc_v[label[i]] += 1.0
    val_f1_score=f1_score(true_label,predicted_label,average='macro')


    return sum(acc) / sum(num), sum(acc_a) / sum(num),sum(acc_v) / sum(num),val_f1_score


def main(seeds):

    args = get_arguments()
    print(args)
    print(args.use_conflict)

    setup_seed(seeds)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0' if (args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available())) else 'cpu')
    if device.type == 'cpu':
        gpu_ids = []

    dataset_cls = ADNI if args.dataset.startswith('ADNI') else ANM
    data_path = Path(args.data_root) / args.dataset / 'overlap'
    train_dataset = dataset_cls(split='train', datapath=data_path)
    val_dataset = dataset_cls(split='val', datapath=data_path, preprocessor=train_dataset.preprocessor)
    train_label = train_dataset.get_true_label()
    if set(train_label) != set(range(args.n_classes)):
        raise ValueError('Training split is missing a class')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers, pin_memory=False)

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers)
    
    class_weights = compute_class_weight('balanced', classes=torch.unique(torch.tensor(train_label)).numpy(), y=train_label)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    
    model = AVClassifier(args,train_dataset.get_csvshape())
    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.to(device)


    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08)

    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)


    train_acc_mm_history = []
    train_acc_a_history = []
    train_acc_v_history = []

    val_acc_mm_history = []
    val_acc_a_history = []
    val_acc_v_history = []

    this_cos_audio_history=[]
    this_cos_visual_history=[]


    if args.train:
        best_acc = -1
        best_f1=-1

        for epoch in range(args.epochs):

            batch_loss,train_acc_mm, train_acc_a, train_acc_v,this_cos_audio_train,this_cos_visual_train = train_epoch(
                args, epoch, model, device, train_dataloader, optimizer, scheduler,class_weights, None)
            if args.lr_schedule == 'step':
                scheduler.step()
            
            train_acc_mm_history.append(train_acc_mm)
            train_acc_a_history.append(train_acc_a)
            train_acc_v_history.append(train_acc_v)
            this_cos_audio_history.extend(this_cos_audio_train)
            this_cos_visual_history.extend(this_cos_visual_train)

            acc, acc_a,acc_v,val_f1 = valid(args, model, device, val_dataloader)

            val_acc_mm_history.append(acc)
            val_acc_a_history.append(acc_a)
            val_acc_v_history.append(acc_v)




            if val_f1 > best_f1:
                best_f1 = float(val_f1)

                os.makedirs(os.path.join(args.dataset, args.ckpt_path), exist_ok=True)
                if args.use_conflict:
                    if args.use_healnet:
                        model_name = '{}_MMPareto_healnet_{}.pth'.format(args.dataset,seeds)
                    else:
                        model_name = '{}_MMPareto_{}.pth'.format(args.dataset,seeds)
                elif args.use_coatt:
                    model_name = '{}_crossattention_{}.pth'.format(args.dataset,seeds)
                else:
                    if args.use_healnet:
                        model_name = '{}_healnet_depth5_{}.pth'.format(args.dataset,seeds)
                    else:
                        model_name = '{}_{}.pth'.format(args.dataset,seeds)

                saved_dict = {'saved_epoch': epoch,
                                'acc': acc,
                                'f1': val_f1,
                                'preprocessing': train_dataset.preprocessor.state,
                                'train_fingerprint': train_dataset.fingerprint,
                                'task': args.dataset,
                                'args': vars(args),
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict()}

                save_dir = os.path.join(args.ckpt_path, model_name)

                torch.save(saved_dict, os.path.join(args.dataset,save_dir))

                print("Saved best validation F1: {:.4f}".format(best_f1))
            print("Epoch: {}: Train Loss: {:.4f}, Train Acc: {:.4f}, Train Acc_a: {:.4f}, Train Acc_v: {:.4f}, Val Acc: {:.4f}, Val Acc_a: {:.4f}, Val Acc_v: {:.4f}".format(
                epoch,batch_loss,train_acc_mm, train_acc_a, train_acc_v, acc, acc_a,acc_v))
            

















                

if __name__ == "__main__":
    for seeds in get_arguments().seeds:
        main(seeds)
