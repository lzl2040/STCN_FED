import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from davis2017.davis import DAVIS
from davis2017.metrics import db_eval_boundary, db_eval_iou
from davis2017 import utils
from davis2017.results import Results
from scipy.optimize import linear_sum_assignment


class DAVISEvaluation(object):
    def __init__(self, davis_root, task, gt_set, sequences='all', codalab=False, show=False):
        """
        Class to evaluate DAVIS sequences from a certain set and for a certain task
        :param davis_root: Path to the DAVIS folder that contains JPEGImages, Annotations, etc. folders.
        :param task: Task to compute the evaluation, chose between semi-supervised or unsupervised.
        :param gt_set: Set to compute the evaluation
        :param sequences: Sequences to consider for the evaluation, 'all' to use all the sequences in a set.
        """
        self.davis_root = davis_root
        self.task = task
        self.dataset = DAVIS(root=davis_root, task=task, subset=gt_set, sequences=sequences, codalab=codalab) # ['paragliding-launch', 'shooting', 'soapbox']
        self.show = show

    @staticmethod
    def _evaluate_semisupervised(all_gt_masks, all_res_masks, all_void_masks, metric):
        print(all_res_masks.shape[0])
        print(all_gt_masks.shape[0])
        print("all" + str(all_gt_masks.shape))
        if all_res_masks.shape[0] > all_gt_masks.shape[0]:
            sys.stdout.write("\nIn your PNG files there is an index higher than the number of objects in the sequence!")
            sys.exit()
        elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
            zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
            all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
        j_metrics_res, f_metrics_res = np.zeros(all_gt_masks.shape[:2]), np.zeros(all_gt_masks.shape[:2])
        for ii in range(all_gt_masks.shape[0]):
            # print(ii)
            if 'J' in metric:
                # print("J")
                j_metrics_res[ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
            if 'F' in metric:
                # print("F")
                f_metrics_res[ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
        return j_metrics_res, f_metrics_res

    @staticmethod
    def _evaluate_unsupervised(all_gt_masks, all_res_masks, all_void_masks, metric, max_n_proposals=20):
        if all_res_masks.shape[0] > max_n_proposals:
            sys.stdout.write(f"\nIn your PNG files there is an index higher than the maximum number ({max_n_proposals}) of proposals allowed!")
            sys.exit()
        elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
            zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
            all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
        j_metrics_res = np.zeros((all_res_masks.shape[0], all_gt_masks.shape[0], all_gt_masks.shape[1]))
        f_metrics_res = np.zeros((all_res_masks.shape[0], all_gt_masks.shape[0], all_gt_masks.shape[1]))
        for ii in range(all_gt_masks.shape[0]):
            for jj in range(all_res_masks.shape[0]):
                if 'J' in metric:
                    j_metrics_res[jj, ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[jj, ...], all_void_masks)
                if 'F' in metric:
                    f_metrics_res[jj, ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[jj, ...], all_void_masks)
        if 'J' in metric and 'F' in metric:
            all_metrics = (np.mean(j_metrics_res, axis=2) + np.mean(f_metrics_res, axis=2)) / 2
        else:
            all_metrics = np.mean(j_metrics_res, axis=2) if 'J' in metric else np.mean(f_metrics_res, axis=2)
        row_ind, col_ind = linear_sum_assignment(-all_metrics)
        return j_metrics_res[row_ind, col_ind, :], f_metrics_res[row_ind, col_ind, :]

    def evaluate(self, res_path, metric=('J', 'F'), debug=False):
        metric = metric if isinstance(metric, tuple) or isinstance(metric, list) else [metric]
        if 'T' in metric:
            raise ValueError('Temporal metric not supported!')
        if 'J' not in metric and 'F' not in metric:
            raise ValueError('Metric possible values are J for IoU or F for Boundary')

        # Containers
        metrics_res = {}
        if 'J' in metric:
            metrics_res['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
        if 'F' in metric:
            metrics_res['F'] = {"M": [], "R": [], "D": [], "M_per_object": {}}

        # Sweep all sequences
        # print("Sweep all sequences")
        # print(self.show)
        results = Results(root_dir=res_path)
        # print("get results")
        seqs = self.dataset.get_sequences()
        for seq in tqdm(list(seqs)):  # ['bike-packing', 'blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout', 'car-shadow', 'cows', 'dance-twirl', 'dog', 'dogs-jump', 'drift-chicane', 'drift-straight', 'goat', 'gold-fish', 'horsejump-high', 'india', 'judo', 'kite-surf', 'lab-coat', 'libby', 'loading', 'mbike-trick', 'motocross-jump', 'paragliding-launch', 'parkour', 'pigs', 'scooter-black', 'shooting', 'soapbox']
            print('seq:' + seq)
            all_gt_masks, all_void_masks, all_masks_id = self.dataset.get_all_masks(seq, True)
            # print(all_masks_id)
            # print(all_gt_masks.shape[0])
            if self.task == 'semi-supervised':
                len = all_gt_masks.shape[1]
                all_gt_masks, all_masks_id = all_gt_masks[:, 1:len+1, :, :], all_masks_id[1:len+1]
            all_res_masks = results.read_masks(seq, all_masks_id,n = all_gt_masks.shape[0])
            # print("read")
            if self.task == 'unsupervised':
                j_metrics_res, f_metrics_res = self._evaluate_unsupervised(all_gt_masks, all_res_masks, all_void_masks, metric)
            elif self.task == 'semi-supervised':
                # print("start")
                j_metrics_res, f_metrics_res = self._evaluate_semisupervised(all_gt_masks, all_res_masks, None, metric)
                # print("end")
            for ii in range(all_gt_masks.shape[0]): # all_gt_masks (2, 67, 480, 910)
                seq_name = f'{seq}_{ii+1}'
                # print(seq_name)
                if 'J' in metric:
                    [JM, JR, JD] = utils.db_statistics(j_metrics_res[ii])
                    metrics_res['J']["M"].append(JM)
                    metrics_res['J']["R"].append(JR)
                    metrics_res['J']["D"].append(JD)
                    metrics_res['J']["M_per_object"][seq_name] = JM
                if 'F' in metric:
                    [FM, FR, FD] = utils.db_statistics(f_metrics_res[ii])
                    metrics_res['F']["M"].append(FM)
                    metrics_res['F']["R"].append(FR)
                    metrics_res['F']["D"].append(FD)
                    metrics_res['F']["M_per_object"][seq_name] = FM
                if self.show:
                    jandf = (f_metrics_res[ii] + j_metrics_res[ii])/2
                    import matplotlib.pylab as plt
                    plt.figure()
                    plt.plot(jandf)
                    plt.plot(jandf, 'ro')
                    plt.savefig("{}_{}.tiff".format(seq, ii))
            print("over")
            # Show progress
            if debug:
                sys.stdout.write(seq + '\n')
                sys.stdout.flush()
        return metrics_res
