import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve
# 'blue', 'green', 'purple', 'black', 'cyan', 'mistyrose', 'pink', 'gold', 'magenta', 'darkorange', 'dimgray', 'lime'
def _prepare_data(pred: np.ndarray, gt: np.ndarray) -> tuple:
    gt = gt > 128
    pred = pred / 255
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    return pred, gt


def _get_adaptive_threshold(matrix: np.ndarray, max_value: float = 1) -> float:
    return min(2 * matrix.mean(), max_value)

class FNR(object):
    def __init__(self, beta: float = 0.3):
        self.beta = beta
        self.precisions = []
        self.recalls = []
        self.fnrs = []


    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred, gt)

        precisions, recalls, changeable_fms = self.cal_pr(pred=pred, gt=gt)
        fnr = 1 - recalls

        self.precisions.append(precisions)
        self.recalls.append(recalls)
        self.fnrs.append(fnr)

    def cal_pr(self, pred: np.ndarray, gt: np.ndarray) -> tuple:
        pred = (pred * 255).astype(np.uint8)
        bins = np.linspace(0, 256, 257)
        fg_hist, _ = np.histogram(pred[gt], bins=bins)
        bg_hist, _ = np.histogram(pred[~gt], bins=bins)
        fg_w_thrs = np.cumsum(np.flip(fg_hist), axis=0)
        bg_w_thrs = np.cumsum(np.flip(bg_hist), axis=0)
        TPs = fg_w_thrs
        Ps = fg_w_thrs + bg_w_thrs
        Ps[Ps == 0] = 1
        T = max(np.count_nonzero(gt), 1)
        precisions = TPs / Ps
        recalls = TPs / T
        numerator = (1 + self.beta) * precisions * recalls
        denominator = np.where(numerator == 0, 1, self.beta * precisions + recalls)
        changeable_fms = numerator / denominator
        return precisions, recalls, changeable_fms

    def get_results(self) -> dict:
        precisions = np.mean(self.precisions, axis=0)
        recalls = np.mean(self.recalls, axis=0)
        F_measure = 1.3*precisions*recalls/(0.3*precisions+recalls)
        return dict(precision=precisions, recall=recalls, F_measure=F_measure)

datasets = ['NJU2K', 'NLPR', 'STERE', 'DES', 'SSD', 'LFSD', 'SIP']
# dataset = ['NJU2K']
i = 0
algorithm = ['PICA', 'BBSNet', 'BTS-Net', 'DCMF','DMRA','SPSN', 'HiDANet','Ours']
# algorithm = ['1']
colors = ['blue', 'green', 'purple', 'gold', 'black', 'cyan','green', 'purple', 'gold', 'black','red']
linestyles = ['--', '--', '--', '--', '--', '-', '-', '-', '-', '-', '-', '-']
linewidths = [1, 1, 1, 1, 1, 1, 1, 1, 1.8]
# axisPR = [[0.0, 1.0, 0.7, 1.0], [0.0, 1.0, 0.65, 0.95], [0.0, 1.0, 0.675, 0.975], [0.0, 1.0, 0.7, 1.0],
#           [0.0, 1.0, 0.6, 0.9]]
# axisFT = [[0.0, 255.0, 0.75, 0.96], [0.0, 255.0, 0.70, 0.89], [0.0, 255.0, 0.6, 0.9], [0.0, 255.0, 0.7, 0.95],
#           [0.0, 255.0, 0.60, 0.84]]

axisPR = [[0.0, 1.0, 0.7, 1.0], [0.0, 1.0, 0.65, 1.0], [0.0, 1.0, 0.675, 1.0], [0.0, 1.0, 0.7, 1.0],
          [0.0, 1.0, 0.6, 1.0],[0.0, 1.0, 0.65, 1.0], [0.0, 1.0, 0.675, 1.0], [0.0, 1.0, 0.7, 1.0]]

axisFT = [[0.0, 255.0, 0.75, 0.95], [0.0, 255.0, 0.70, 0.95], [0.0, 255.0, 0.6, 0.95], [0.0, 255.0, 0.7, 0.93],
          [0.0, 255.0, 0.60, 0.95], [0.0, 255.0, 0.70, 0.93], [0.0, 255.0, 0.6, 1.0], [0.0, 255.0, 0.7, 1.0]]


# axisPR = [[0.0, 1.05, 0.1, 1.05], [0.0, 1.05, 0.1, 1.05], [0.0, 1.05, 0.1, 1.05], [0.0, 1.05, 0.1, 1.05],
#           [0.0, 1.05, 0.1, 1.05], [0.0, 1.05, 0.1, 1.05] ,[0.0, 1.05, 0.1, 1.05], [0.0, 1.05, 0.1, 1.05]]
# axisFT = [[0.0, 1.05, 0.1, 1.05], [0.0, 1.05, 0.1, 1.05], [0.0, 1.05, 0.1, 1.05], [0.0, 1.05, 0.1, 1.05],
#           [0.0, 1.05, 0.1, 1.05], [0.0, 1.05, 0.1, 1.05] ,[0.0, 1.05, 0.1, 1.05], [0.0, 1.05, 0.1, 1.05]]
plt.figure(figsize=(4, 10))
plt.subplots_adjust(top=0.95, left=0.3, bottom=0.06, right=0.97, wspace=0.2, hspace=0.3)
flag = True
gt_folder = 'F:/BaiduNetdiskDownload/paperresult/datasets/'
pr_folder = 'C:/Users/Administrator/Desktop/OKdatasets/'

for i, data in enumerate(datasets):
    plt.subplot(2, 1, 1)
    for j, algo in enumerate(algorithm):
        # gt_folder = 'C:/Users/SpongeBob/Desktop/BBstest1/' + data + '/' + 'GT'
        # pr_folder = 'C:/Users/SpongeBob/Desktop/论文数据集/数据集1/' + algo + '/'

        print(data)
        PR_folder = pr_folder + algo + '/' + data
        GT_folder = gt_folder + data + '/Test/GT'
        print('pred_root:', PR_folder, '        gnd_truth:', GT_folder)

        fnr = FNR()
        a = 0

        for pr_file in tqdm(os.listdir(PR_folder)):
            if pr_file.endswith('.png') or pr_file.endswith('.jpg'):
                pr_path = os.path.join(PR_folder, pr_file)
                gt_path = os.path.join(GT_folder, pr_file)
                if os.path.isfile(gt_path):
                    # Load predicted and ground truth images as numpy arrays
                    pr_img = np.array(Image.open(pr_path).convert('L'))
                    gt_img = np.array(Image.open(gt_path).convert('L'))
                    if pr_img.shape != gt_img.shape:
                        # Resize predicted image to match ground truth image shape
                        pr_img = cv2.resize(pr_img, gt_img.shape[::-1])
                    # Compute precision and recall values using FNR class
                    fnr.step(pr_img, gt_img)

        # Get the precision and recall values
        results = fnr.get_results()
        precisions = results['precision']
        recalls = results['recall']
        if algo == 'Ours':
            plt.plot(recalls, precisions, color=colors[-1], linestyle=linestyles[j], linewidth=linewidths[j], label=algo)
        else:
            plt.plot(recalls, precisions, color=colors[j], linestyle=linestyles[j], linewidth=linewidths[j], label=algo)

        plt.axis(axisPR[i])
        # plt.savefig('C:/Users/SpongeBob/Desktop/result/' + data + '.svg', format='svg', dpi=200, transparent=True,
        #             bbox_inches='tight')
    # plt.title(data, fontsize=12)
    plt.xlabel('Recall', fontsize=14)
    if data == 'NJU2K':
        plt.ylabel('Precision', fontsize=20)

    plt.grid(ls='--')
    plt.legend(loc="lower center", ncol=2, fontsize=10, frameon=True, framealpha=0.2)
    plt.savefig('C:/Users/Administrator/Desktop/OKdatasets/' + data + '.svg', format='svg', dpi=200, transparent=True,
                bbox_inches='tight')
    plt.clf()


# 调用图例一次

#之前的注释
# for i, data in enumerate(datasets):
#     plt.subplot(2, 1, 2)
#     for j, algo in enumerate(algorithm):
#         gt_folder = 'F:/BaiduNetdiskDownload/paperresult/datasets'
#         pr_folder = 'C:/Users/Administrator/Desktop/OKdatasets'
#         # gt_folder = 'C:/Users/SpongeBob/Desktop/BBstest1/' + data + '/' + 'GT'
#         # pr_folder = 'C:/Users/SpongeBob/Desktop/论文数据集/数据集1/' + algo + '/' + data
#         print('pred_root:', pr_folder, '        gnd_truth:', gt_folder)
#         fnr = FNR()
#         a = 0
#         for pr_file in os.listdir(pr_folder):
#             a += 1
#             print(a)
#             if pr_file.endswith('.png') or pr_file.endswith('.jpg'):
#                 pr_path = os.path.join(pr_folder, pr_file)
#                 gt_path = os.path.join(gt_folder, pr_file)
#                 if os.path.isfile(gt_path):
#                     # Load predicted and ground truth images as numpy arrays
#                     pr_img = np.array(Image.open(pr_path).convert('L'))
#                     gt_img = np.array(Image.open(gt_path).convert('L'))
#                     if pr_img.shape != gt_img.shape:
#                         # Resize predicted image to match ground truth image shape
#                         pr_img = cv2.resize(pr_img, gt_img.shape[::-1])
#                     # Compute precision and recall values using FNR class
#                     fnr.step(pr_img, gt_img)
#
#         # Get the precision and recall values
#         results = fnr.get_results()
#         precisions = results['precision']
#         recalls = results['recall']
#         F_measure = results['F_measure']
#         plt.plot(np.linspace(0, 256, 256), F_measure, color=colors[j], linestyle=linestyles[j], linewidth=linewidths[j],
#                  label=algo)
#         plt.axis(axisFT[i])
#
#     # plt.title(data, fontsize=12)
#     plt.xlabel('Threshod', fontsize=14)
#     if data == 'NJU2K':
#         plt.ylabel('F-measure', fontsize=20)
#     plt.legend(loc="lower center", ncol=2, fontsize=10, frameon=True,framealpha=0.2)

    # plt.grid(ls='--')
    # plt.savefig('C:/Users/SpongeBob/Desktop/result/' + data + '.svg', format='svg', dpi=200,transparent=True, bbox_inches='tight')
plt.show()

