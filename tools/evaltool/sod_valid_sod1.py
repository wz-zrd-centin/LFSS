"""
Code: https://github.com/mczhuge/ICON
Author: mczhuge
Desc: Core code for validation
"""

import os
import sys
import cv2
from tqdm import tqdm
import metrics as M
import json
import argparse

def main():
    args = parser.parse_args()
    # SOD
    method_list = args.method

    print(os.listdir(method_list))
    # 修改计算范围

    method_list_dir = ['Ours','PICA', 'BBSNet', 'BTS-Net', 'DCMF','DMRA','SPSN', 'HiDANet']

    for method in method_list_dir:
        datasets = ['NJU2K','NLPR','STERE','DES','LFSD','SSD','SIP']
        for dataset in datasets:
            for i in range(0, 1):
                FM = M.Fmeasure_and_FNR()
                WFM = M.WeightedFmeasure()
                SM = M.Smeasure()
                EM = M.Emeasure()
                MAE = M.MAE()
                #FNR = M.FNR()
                #attr = args.attr

                args.dataset = dataset
                dataset = args.dataset

                gt_root = os.path.join('../../datasets/' + dataset + '/Test/', 'GT')

                pred_root = os.path.join(method_list + '/' + method + '/', dataset)

                print(method)
                print(dataset)
                print('pred_root:', pred_root, '        gnd_truth:', gt_root)

                gt_name_list = sorted(os.listdir(pred_root))
                for gt_name in tqdm(gt_name_list, total=len(gt_name_list)):
                    #print(gt_name)
                    gt_path = os.path.join(gt_root, gt_name)
                    pred_path = os.path.join(pred_root, gt_name)
                    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

                    # print(gt.shape)
                    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                    # print(gt.shape, pred.shape)
                    if gt.shape != pred.shape:
                        gt = cv2.resize(gt, pred.shape)

                    # pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                    # print(gt.shape, pred.shape)
                    FM.step(pred=pred, gt=gt)
                    WFM.step(pred=pred, gt=gt)
                    SM.step(pred=pred, gt=gt)
                    EM.step(pred=pred, gt=gt)
                    MAE.step(pred=pred, gt=gt)
                    #FNR.step(pred=pred, gt=gt)

                fm = FM.get_results()[0]['fm']
                wfm = WFM.get_results()['wfm']
                sm = SM.get_results()['sm']
                em = EM.get_results()['em']
                mae = MAE.get_results()['mae']
                fnr = FM.get_results()[1]

                Method_r = str(args.method)
                Dataset_r = str(args.dataset)
                Smeasure_r = str(sm.round(3))
                Wmeasure_r = str(wfm.round(3))
                MAE_r = str(mae.round(3))
                adpEm_r = str(em['adp'].round(3))
                meanEm_r = str('-' if em['curve'] is None else em['curve'].mean().round(3))
                maxEm_r = str('-' if em['curve'] is None else em['curve'].max().round(3))
                adpFm_r = str(fm['adp'].round(3))
                meanFm_r = str(fm['curve'].mean().round(3))
                maxFm_r = str(fm['curve'].max().round(3))
                fnr_r = str(fnr.round(3))


                eval_record = str(
                    # 'Method:'+ Method_r + ','+
                    #'Dataset:'+ Dataset_r + ' || '+
                    method +"--"+ dataset +' :\t' +
                    'Smeasure\t'+ Smeasure_r + '\t'+
                    'maxFm\t' + maxFm_r + '\t' +
                    'maxEm\t' + maxEm_r + '\t' +
                    'MAE\t'+ MAE_r + '\t'
                     )


                print(eval_record)
                print('-' * 100)
                print()


                txt = 'Prediction/result.txt'
                f = open(txt, 'a')
                f.write(eval_record)
                f.write("\n")
                f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default='C:/Users/Administrator/Desktop/OKdatasets')
    parser.add_argument("--dataset", default='HiDANet')
    #parser.add_argument("--attr", default='SOC-AC')
    main()
