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
from sklearn.metrics import classification_report,precision_recall_curve,roc_auc_score,average_precision_score
from sklearn.preprocessing import label_binarize
import pandas as pd


def calc_confusion_matrix(result, test_label,n_classes):
    result = F.one_hot(result,num_classes=n_classes)

    test_label = F.one_hot(test_label,num_classes=n_classes)

    true_label= np.argmax(test_label, axis =1)

    predicted_label= np.argmax(result, axis =1)
    
    precision = dict()
    recall = dict()
    thres = dict()
    for i in range(n_classes):
        precision[i], recall[i], thres[i] = precision_recall_curve(test_label[:, i],
                                                            result[:, i])


    print (classification_report(true_label, predicted_label,digits=3))
    cr = classification_report(true_label, predicted_label, output_dict=True,digits=3)
    return cr

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=os.environ.get("CAPPM_DATA_ROOT", str(Path(__file__).resolve().parents[1] / "processed_data")), help="External directory containing ANM/overlap, ADNI/overlap, etc.")
    parser.add_argument('--dataset', required=True, type=str,
                        help='ANM, ADNI')
    parser.add_argument('--model', default='model', type=str)
    parser.add_argument('--n_classes', default=None, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--optimizer', default='sgd',
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
    parser.add_argument('--split', default='test',
                        type=str, help='[train,val,test]')
    parser.add_argument('--use_conflict', action='store_true',
                         help='whether to conflict')
    parser.add_argument('--use_coatt', action='store_true',
                         help='whether to conflict')
    parser.add_argument('--use_healnet', action='store_true',
                         help='whether to conflict')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0,1,2,3,4])
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


def valid(args, model, device, dataloader):

    n_classes = args.n_classes

    all_a_emb = []
    all_v_emb = []
    predicted_label=[]
    true_label=[]
    out=[]
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
            a_emb = prediction_all[3]
            v_emb = prediction_all[4]

            _, predicted = prediction.max(1)
            predicted_label.extend(predicted.tolist())
            out.extend(prediction.tolist())
            true_label.extend(label.tolist())

            all_a_emb.append(a_emb.detach().cpu())
            all_v_emb.append(v_emb.detach().cpu())

            a_emb_combined = torch.cat(all_a_emb, dim=0)
            v_emb_combined = torch.cat(all_v_emb, dim=0)

            emv_combined = torch.cat((a_emb_combined,v_emb_combined), dim=1)

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

        cr=calc_confusion_matrix(torch.tensor(predicted_label), torch.tensor(true_label),n_classes)
        true_label_binarized = label_binarize(true_label, classes=list(range(n_classes)))

        auc_score = roc_auc_score(true_label_binarized, torch.softmax(torch.tensor(out), dim=1).numpy(), average='macro', multi_class='ovr')
        aupr_score = average_precision_score(true_label_binarized, torch.softmax(torch.tensor(out), dim=1).numpy(), average='macro')


    return sum(acc) / sum(num), sum(acc_a) / sum(num),sum(acc_v) / sum(num),a_emb_combined,v_emb_combined,emv_combined,auc_score,aupr_score,cr


def main(seeds):

    args = get_arguments()

    setup_seed(seeds)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0' if (args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available())) else 'cpu')
    if device.type == 'cpu':
        gpu_ids = []

    if args.use_conflict:
        suffix = '_MMPareto_healnet_' if args.use_healnet else '_MMPareto_'
    elif args.use_coatt:
        suffix = '_crossattention_'
    else:
        suffix = '_healnet_depth5_' if args.use_healnet else '_'
    checkpoint_path = Path(args.dataset) / args.ckpt_path / (args.dataset + suffix + str(seeds) + '.pth')
    # Only load checkpoints you created or otherwise trust.
    saved_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'preprocessing' not in saved_dict or saved_dict.get('task') != args.dataset:
        raise ValueError('Checkpoint lacks training preprocessing/task metadata; retrain with this package')
    for key in ('use_conflict', 'use_healnet'):
        if saved_dict.get('args', {}).get(key) != getattr(args, key):
            raise ValueError('Evaluation model flags differ from checkpoint: ' + key)
    preprocessing = TrainPreprocessor(state=saved_dict['preprocessing'])
    dataset_cls = ADNI if args.dataset.startswith('ADNI') else ANM
    the_dataset = dataset_cls(split=args.split, datapath=Path(args.data_root) / args.dataset / 'overlap', preprocessor=preprocessing)

    the_dataloader = DataLoader(the_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers)
    
    
    model = AVClassifier(args,the_dataset.get_csvshape())
    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.to(device)

    if args.use_conflict:
        if args.use_healnet:
            model_name = str(args.dataset)+'_MMPareto'+'_healnet'+'_'+str(seeds)+'.pth'
        else:
            model_name = str(args.dataset)+'_MMPareto'+'_'+str(seeds)+'.pth'
    elif args.use_coatt:
            model_name = str(args.dataset)+'_crossattention'+'_'+str(seeds)+'.pth'
    else:
        if args.use_healnet:
            model_name = str(args.dataset)+'_healnet_depth5'+'_'+str(seeds)+'.pth'
        else:
            model_name = str(args.dataset)+'_'+str(seeds)+'.pth'
    print(model_name)

    save_dir = os.path.join(args.ckpt_path, model_name)


    model.load_state_dict(saved_dict['model'])
    acc, acc_a,acc_v,a_emb,v_emb,all_emb,auc_score,aupr_score,cr = valid(args, model, device, the_dataloader)

    print("F1: {:.4f}, acc: {:.4f}, pre: {:.4f}, recal: {:.4f}, auroc:{:.4f},aupr:{:.4f}".format(cr['macro avg']['f1-score'],cr['accuracy'],cr['macro avg']['precision'],cr['macro avg']['recall'],auc_score,aupr_score))









    return cr,auc_score,aupr_score
                

if __name__ == "__main__":
    F1_list=[]
    acc_list=[]
    pre_list=[]
    recall_list=[]
    auc_list=[]
    aupr_list=[]
    for seeds in get_arguments().seeds:
        cr,auc,aupr=main(seeds)
        F1_list.append(cr['macro avg']['f1-score'])
        acc_list.append(cr['accuracy'])
        pre_list.append(cr['macro avg']['precision'])
        recall_list.append(cr['macro avg']['recall'])
        auc_list.append(auc)
        aupr_list.append(aupr)
    print(sum(F1_list)/len(F1_list),sum(acc_list)/len(F1_list),sum(pre_list)/len(F1_list),sum(recall_list)/len(F1_list),sum(auc_list)/len(F1_list),sum(aupr_list)/len(F1_list))
    print(np.std(F1_list),np.std(acc_list),np.std(pre_list),np.std(recall_list),np.std(auc_list),np.std(aupr_list))
