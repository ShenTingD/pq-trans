# shenting
# save detection result into json file

import numpy as np
import torch
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

DUMP_CONF_THRESH = 0.8  # Dump boxes with obj prob larger than that.


def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs


def dump_result_json(end_points, config, pred_sem_cls):
    ''' Dump results.

    Args:
        end_points: dict
            {..., pred_mask}
            pred_mask is a binary mask array of size (batch_size, num_proposal) computed by running NMS and empty box removal
    Returns:
        None
    '''

    # INPUT
    point_clouds = end_points['point_clouds'].cpu().numpy()
    batch_size = point_clouds.shape[0]

    # NETWORK OUTPUTS
    # seed_xyz = end_points['seed_xyz'].detach().cpu().numpy() # (B,num_seed,3)
    # if 'vote_xyz' in end_points:
    #     aggregated_vote_xyz = end_points['aggregated_vote_xyz'].detach().cpu().numpy()
    #     vote_xyz = end_points['vote_xyz'].detach().cpu().numpy() # (B,num_seed,3)
    #     aggregated_vote_xyz = end_points['aggregated_vote_xyz'].detach().cpu().numpy()
    objectness_scores = end_points['last_objectness_scores'].detach().cpu().numpy()  # (B,K,2)
    pred_center = end_points['last_center'].detach().cpu().numpy()  # (B,K,3)
    pred_heading_class = torch.argmax(end_points['last_heading_scores'], -1)  # B,num_proposal
    pred_heading_residual = torch.gather(end_points['last_heading_residuals'], 2,
                                         pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
    pred_heading_class = pred_heading_class.detach().cpu().numpy()  # B,num_proposal
    pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy()  # B,num_proposal
    pred_size_class = torch.argmax(end_points['last_size_scores'], -1)  # B,num_proposal
    pred_size_residual = torch.gather(end_points['last_size_residuals'], 2,
                                      pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1,
                                                                                         3))  # B,num_proposal,1,3
    pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy()  # B,num_proposal,3
    aggregated_vote_xyz = end_points['fp2_xyz'].detach().cpu().numpy()

    scan_names = end_points['scan_name']  # .detach().cpu() #B,scan_name
    # OTHERS
    pred_mask = end_points['pred_mask']  # B,num_proposal
    idx_beg = 0

    result = dict()
    for i in range(batch_size):
        result[scan_names[i]] = dict()
        if scan_names[i] == 'scene0647_00':
            debug = 0
        pc = point_clouds[i, :, :]
        objectness_prob = softmax(objectness_scores[i, :, :])[:, 1]  # (K,)

        # Dump various point clouds
        result[scan_names[i]]['pc'] = pc.tolist()

        # Dump predicted bounding boxes

        if np.sum(objectness_prob > DUMP_CONF_THRESH) > 0:
            num_proposal = pred_center.shape[1]
            obbs = []
            for j in range(num_proposal):
                obb = config.param2obb(pred_center[i, j, 0:3], pred_heading_class[i, j], pred_heading_residual[i, j],
                                       pred_size_class[i, j], pred_size_residual[i, j])
                obb = np.append(obb, pred_sem_cls[i, j])
                obbs.append(obb)
                # if ((obb[0]>0.6) and (obb[1]<-0.3)) or (obb[1]<-0.5):
                #     pred_mask[i,j] = 0
            if len(obbs) > 0:
                obbs = np.vstack(tuple(obbs))  # (num_proposal, 7)
                result[scan_names[i]]['pred_conf_bbox']= obbs[objectness_prob > DUMP_CONF_THRESH, :].tolist()
                result[scan_names[i]]['pred_conf_nms_bbox']= obbs[np.logical_and(objectness_prob > DUMP_CONF_THRESH, pred_mask[i, :] == 1), :].tolist()
                result[scan_names[i]]['pred_nms_bbox']= obbs[pred_mask[i, :] == 1, :].tolist()
    return result