import argparse
import transformations as ts
import opt_tc as tc
import numpy as np
from data_loader import Data_Loader
import torch 
import os 
from collections import defaultdict
import opt_tc as tc
from wideresnet import WideResNet
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import pandas as pd 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def roc_curve_plot(y_test , pred_proba_c1):
    # 임곗값에 따른 FPR, TPR 값을 반환 받음. 
    fprs , tprs , thresholds = roc_curve(y_test ,pred_proba_c1)

    # ROC Curve를 plot 곡선으로 그림. 
    plt.plot(fprs , tprs, label='ROC')
    # 가운데 대각선 직선을 그림. 
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
  
    # FPR X 축의 Scale을 0.1 단위로 변경, X,Y 축명 설정등   
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('FPR( 1 - Sensitivity )')
    plt.ylabel('TPR( Recall )')
    plt.legend()
    plt.savefig('./result.png')

def vis_result(df, normal_threshold):

    plt.figure(figsize=(8,6))
    plt.scatter(df[df.Label == 0].index, df[df.Label == 0].Scores, c='indigo', marker='o', label='normal')
    plt.scatter(df[df.Label == 1].index, df[df.Label == 1].Scores, alpha=0.5,c='gold', marker='o', label='abnormal')
    
    plt.title('normal and abnormal DeepSVDD scores', fontsize = 12)
    plt.xlabel('Length', fontsize = 10)
    plt.ylabel('Scores', fontsize = 10)

    
    plt.legend()
    plt.axhline(normal_threshold, 0, len(df), color='blue', linestyle='--', linewidth=2)
    plt.savefig('./scores.png')

def transform_data(data, trans):
    trans_inds = np.tile(np.arange(trans.n_transforms), len(data))
    trans_data = trans.transform_batch(np.repeat(np.array(data), trans.n_transforms, axis=0), trans_inds)
    return trans_data, trans_inds

# 데이터 로드
def load_trans_data(args, trans):
    dl = Data_Loader()
    x_train, x_vaild, y_vaild, x_test, y_test, class_name = dl.get_dataset1(args.dataset, true_label=args.class_ind)
    x_train_trans, labels = transform_data(x_train, trans)  # (5000, 32, 32, 3) -> (360000, 32, 32, 3)
    x_vaild, _ = transform_data(x_vaild, trans)
    x_test, _ = transform_data(x_test, trans)
    
    x_train_trans = x_train_trans.transpose(0, 3, 1, 2)
    
    x_vaild_trans = x_vaild.transpose(0, 3, 1, 2)
    y_vaild = (np.array(y_vaild) == args.class_ind)     # target normal 번호는 True 나머지 abnormal들은 False로 바꿔줌
    
    x_test_trans = x_test.transpose(0, 3, 1, 2)
    y_test = (np.array(y_test) == args.class_ind)     # target normal 번호는 True 나머지 abnormal들은 False로 바꿔줌
    
    print(len(x_train_trans), len(x_vaild_trans), len(x_test_trans))
    return class_name, x_train_trans, x_vaild_trans, y_vaild, x_test_trans, y_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wide Residual Networks')
    # Model options
    parser.add_argument('--depth', default=10, type=int)
    parser.add_argument('--widen-factor', default=4, type=int)

    # Training options
    parser.add_argument('--batch_size', default=288*3, type=int)  # 288     # 8 또는 72의 배수 
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float)
    parser.add_argument('--epochs', default=21, type=int)    # 16

    # Trans options
    parser.add_argument('--type_trans', default='simple', type=str)    # complicated or simple

    # CT options
    parser.add_argument('--lmbda', default=0.1, type=float)
    parser.add_argument('--m', default=0.1, type=float)               # tc_loss 계산때 사용, cifar 이미지 사용때는 0.1, tabular 데이터는 1
    parser.add_argument('--reg', default=True, type=bool)
    parser.add_argument('--eps', default=0, type=float)

    # Exp options
    parser.add_argument('--class_ind', default=1, type=int)
    parser.add_argument('--dataset', default='cifar10', type=str)
    
    # save model 
    parser.add_argument('--save_path', default='./result', type=str)
    args = parser.parse_args()
    
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    transformer = ts.get_transformer(args.type_trans)
    class_name, x_train, x_vaild, y_vaild, x_test, y_test = load_trans_data(args, transformer)
    
    load_model = './result/{}_best_model.pth'.format(classes[args.class_ind])
    print(load_model)

    num_trans = 8
    widen_factor = 4
    netWRN = WideResNet(args.depth, num_trans, widen_factor).to(device)
    netWRN.load_state_dict(torch.load(load_model)['net_dict'])
    normal_means = torch.load(load_model)['normal']
    print('normal_means:', normal_means.shape)
    
    
    netWRN.eval()
    batch_size = 16
    n_rots = num_trans
    n_rots_test = n_rots
    ndf = 256
    
    
    val_probs_rots = np.zeros((len(y_test), n_rots_test))
    with torch.no_grad():
        for i in tqdm(range(0, len(x_test), batch_size), desc='Test'):
            batch_range = min(batch_size, len(x_test) - i)
            idx = np.arange(batch_range) + i
            xs = torch.from_numpy(x_test[idx]).float().to(device) 

            zs, fs = netWRN(xs)
            zs = torch.reshape(zs, (batch_range // n_rots_test, n_rots_test, ndf))    # ([576, 256])

            diffs = ((zs.unsqueeze(2) - normal_means) ** 2).sum(-1)
            diffs_eps = args.eps * torch.ones_like(diffs)
            diffs = torch.max(diffs, diffs_eps)
            logp_sz = torch.nn.functional.log_softmax(-diffs, dim=2)
            # logp_sz = torch.nn.functional.log_softmax(-diffs, dim=0)

            zs_reidx = np.arange(batch_range // n_rots_test) + i // n_rots_test
            val_probs_rots[zs_reidx] = -torch.diagonal(logp_sz, 0, 1, 2).cpu().data.numpy() # (576, 1)
            
        val_probs_rots = val_probs_rots.sum(1)
        roc_auc = roc_auc_score(y_test, -val_probs_rots)
        
    print('{:.2f}%'.format(roc_auc*100))
    
    roc_curve_plot(y_test , -val_probs_rots)
    
    
    # normal_scores = val_probs_rots[y_test]
    # abnormal_scores = val_probs_rots[[not y for y in y_test]]
    
    scores_df = pd.Series(val_probs_rots, name='Scores').astype(float)
    labels_df = pd.Series(y_test, name='Label')
    df = pd.concat([scores_df, labels_df], axis=1)
    df = df.sample(frac=1,random_state=0).reset_index(drop = True)
    normal_threshold = df.loc[df['Label']==True]['Scores'].mean()
    
    vis_result(df, normal_threshold)
     
    
    
    