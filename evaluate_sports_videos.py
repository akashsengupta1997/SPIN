import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import cv2

import config
import constants
from models import hmr, SMPL
from utils.pose_utils import compute_similarity_transform_batch, \
    scale_and_translation_transform_batch
from utils.geometry import orthographic_project_torch, undo_keypoint_normalisation
from datasets.sports_videos_eval_dataset import SportsVideosEvalDataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def evaluate_single_in_multitasknet_sports_videos(model,
                                                  eval_dataset,
                                                  metrics,
                                                  device,
                                                  save_path,
                                                  num_workers=4,
                                                  pin_memory=True,
                                                  vis_every_n_batches=1):

    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 drop_last=True,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory)

    smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1)
    smpl.to(device)
    smpl_male = SMPL(config.SMPL_MODEL_DIR, batch_size=1, gender='male')
    smpl_male.to(device)
    smpl_female = SMPL(config.SMPL_MODEL_DIR, batch_size=1, gender='female')
    smpl_female.to(device)

    if 'pve' in metrics:
        pve_sum = 0.0
        pve_per_frame = []

    if 'pve_scale_corrected' in metrics:
        pve_scale_corrected_sum = 0.0
        pve_scale_corrected_per_frame = []

    if 'pve_pa' in metrics:
        pve_pa_sum = 0.0
        pve_pa_per_frame = []

    if 'pve-t' in metrics:
        pvet_sum = 0.0
        pvet_per_frame = []

    if 'pve-t_scale_corrected' in metrics:
        pvet_scale_corrected_sum = 0.0
        pvet_scale_corrected_per_frame = []

    frame_path_per_frame = []
    pose_per_frame = []
    shape_per_frame = []
    cam_per_frame = []
    num_samples = 0
    num_vertices = 6890

    model.eval()
    for batch_num, samples_batch in enumerate(tqdm(eval_dataloader)):
        # ------------------------------- TARGETS and INPUTS -------------------------------
        input = samples_batch['input']
        input = input.to(device)

        target_shape = samples_batch['shape']
        target_shape = target_shape.to(device)
        target_vertices = samples_batch['vertices']

        target_gender = samples_batch['gender'][0]
        if target_gender == 'm':
            target_reposed_smpl_output = smpl_male(betas=target_shape)
        elif target_gender == 'f':
            target_reposed_smpl_output = smpl_female(betas=target_shape)
        target_reposed_vertices = target_reposed_smpl_output.vertices

        # ------------------------------- PREDICTIONS -------------------------------
        pred_rotmat, pred_betas, pred_camera = model(input)
        pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                           global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_vertices_projected2d = orthographic_project_torch(pred_vertices, pred_camera)
        pred_vertices_projected2d = undo_keypoint_normalisation(pred_vertices_projected2d, input.shape[-1])
        pred_reposed_smpl_output = smpl(betas=pred_betas)
        pred_reposed_vertices = pred_reposed_smpl_output.vertices

        # Numpy-fying
        target_vertices = target_vertices.cpu().detach().numpy()
        target_reposed_vertices = target_reposed_vertices.cpu().detach().numpy()

        pred_vertices = pred_vertices.cpu().detach().numpy()
        pred_vertices_projected2d = pred_vertices_projected2d.cpu().detach().numpy()
        pred_reposed_vertices = pred_reposed_vertices.cpu().detach().numpy()
        pred_rotmat = pred_rotmat.cpu().detach().numpy()
        pred_betas = pred_betas.cpu().detach().numpy()
        pred_camera = pred_camera.cpu().detach().numpy()

        # ------------------------------- METRICS -------------------------------
        if 'pve' in metrics:
            pve_batch = np.linalg.norm(pred_vertices - target_vertices, axis=-1)  # (1, 6890)
            pve_sum += np.sum(pve_batch)  # scalar
            pve_per_frame.append(np.mean(pve_batch, axis=-1))

        # Scale and translation correction
        if 'pve_scale_corrected' in metrics:
            pred_vertices_scale_corrected = scale_and_translation_transform_batch(
                pred_vertices,
                target_vertices)
            pve_scale_corrected_batch = np.linalg.norm(
                pred_vertices_scale_corrected - target_vertices,
                axis=-1)  # (bs, 6890)
            pve_scale_corrected_sum += np.sum(pve_scale_corrected_batch)  # scalar
            pve_scale_corrected_per_frame.append(
                np.mean(pve_scale_corrected_batch, axis=-1))

        # Procrustes analysis
        if 'pve_pa' in metrics:
            pred_vertices_pa = compute_similarity_transform_batch(pred_vertices, target_vertices)
            pve_pa_batch = np.linalg.norm(pred_vertices_pa - target_vertices, axis=-1)  # (1, 6890)
            pve_pa_sum += np.sum(pve_pa_batch)  # scalar
            pve_pa_per_frame.append(np.mean(pve_pa_batch, axis=-1))

        if 'pve-t' in metrics:
            pvet_batch = np.linalg.norm(pred_reposed_vertices - target_reposed_vertices, axis=-1)
            pvet_sum += np.sum(pvet_batch)
            pvet_per_frame.append(np.mean(pvet_batch, axis=-1))

            # Scale and translation correction
            if 'pve-t_scale_corrected' in metrics:
                pred_reposed_vertices_sc = scale_and_translation_transform_batch(
                    pred_reposed_vertices,
                    target_reposed_vertices)
                pvet_scale_corrected_batch = np.linalg.norm(
                    pred_reposed_vertices_sc - target_reposed_vertices,
                    axis=-1)  # (bs, 6890)
                pvet_scale_corrected_sum += np.sum(pvet_scale_corrected_batch)  # scalar
                pvet_scale_corrected_per_frame.append(
                    np.mean(pvet_scale_corrected_batch, axis=-1))

        num_samples += target_shape.shape[0]

        frame_path = samples_batch['frame_path']
        frame_path_per_frame.append(frame_path)
        pose_per_frame.append(pred_rotmat)
        shape_per_frame.append(pred_betas)
        cam_per_frame.append(pred_camera)

        # ------------------------------- VISUALISE -------------------------------
        if batch_num % vis_every_n_batches == 0:
            vis_imgs = samples_batch['vis_img'].numpy()
            vis_imgs = np.transpose(vis_imgs, [0, 2, 3, 1])

            plt.figure(figsize=(12, 8))
            plt.subplot(231)
            plt.imshow(vis_imgs[0])

            plt.subplot(232)
            plt.imshow(vis_imgs[0])
            plt.scatter(pred_vertices_projected2d[0, :, 0], pred_vertices_projected2d[0, :, 1], s=0.1, c='r')

            plt.subplot(233)
            plt.scatter(target_vertices[0, :, 0], target_vertices[0, :, 1], s=0.1, c='b')
            plt.scatter(pred_vertices[0, :, 0], pred_vertices[0, :, 1], s=0.05, c='r')
            plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal', adjustable='box')

            plt.subplot(234)
            plt.scatter(target_vertices[0, :, 0], target_vertices[0, :, 1], s=0.1, c='b')
            plt.scatter(pred_vertices_pa[0, :, 0], pred_vertices_pa[0, :, 1], s=0.05, c='r')
            plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal', adjustable='box')

            plt.subplot(235)
            plt.scatter(target_reposed_vertices[0, :, 0], target_reposed_vertices[0, :, 1], s=0.1, c='b')
            plt.scatter(pred_reposed_vertices[0, :, 0], pred_reposed_vertices[0, :, 1], s=0.05, c='r')
            plt.gca().set_aspect('equal', adjustable='box')

            plt.subplot(236)
            plt.scatter(target_reposed_vertices[0, :, 0], target_reposed_vertices[0, :, 1], s=0.1, c='b')
            plt.scatter(pred_reposed_vertices_sc[0, :, 0], pred_reposed_vertices_sc[0, :, 1], s=0.05, c='r')
            plt.gca().set_aspect('equal', adjustable='box')

            # plt.show()
            split_path = frame_path[0].split('/')
            clip_name = split_path[-3]
            frame_num = split_path[-1]
            save_fig_path = os.path.join(save_path, clip_name + '_' + frame_num)
            plt.savefig(save_fig_path, bbox_inches='tight')
            plt.close()

    # ------------------------------- DISPLAY METRICS AND SAVE PER-FRAME METRICS -------------------------------
    frame_path_per_frame = np.concatenate(frame_path_per_frame, axis=0)
    np.save(os.path.join(save_path, 'fname_per_frame.npy'), frame_path_per_frame)

    pose_per_frame = np.concatenate(pose_per_frame, axis=0)
    np.save(os.path.join(save_path, 'pose_per_frame.npy'), pose_per_frame)

    shape_per_frame = np.concatenate(shape_per_frame, axis=0)
    np.save(os.path.join(save_path, 'shape_per_frame.npy'), shape_per_frame)

    cam_per_frame = np.concatenate(cam_per_frame, axis=0)
    np.save(os.path.join(save_path, 'cam_per_frame.npy'), cam_per_frame)

    if 'pve' in metrics:
        pve = pve_sum / (num_samples * num_vertices)
        pve_per_frame = np.concatenate(pve_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pve_per_frame.npy'), pve_per_frame)
        print('PVE: {:.5f}'.format(pve))

    if 'pve_scale_corrected' in metrics:
        pve_scale_corrected = pve_scale_corrected_sum / (num_samples * num_vertices)
        pve_scale_corrected_per_frame = np.concatenate(pve_scale_corrected_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pve_scale_corrected_per_frame.npy'),
                pve_scale_corrected_per_frame)
        print('PVE SC: {:.5f}'.format(pve_scale_corrected))

    if 'pve_pa' in metrics:
        pve_pa = pve_pa_sum / (num_samples * num_vertices)
        pve_pa_per_frame = np.concatenate(pve_pa_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pve_pa_per_frame.npy'), pve_pa_per_frame)
        print('PVE PA: {:.5f}'.format(pve_pa))

    if 'pve-t' in metrics:
        pvet = pvet_sum / (num_samples * num_vertices)
        pvet_per_frame = np.concatenate(pvet_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pvet_per_frame.npy'), pvet_per_frame)
        print('PVE-T: {:.5f}'.format(pvet))

    if 'pve-t_scale_corrected' in metrics:
        pvet_scale_corrected = pvet_scale_corrected_sum / (num_samples * num_vertices)
        pvet_scale_corrected_per_frame = np.concatenate(pvet_scale_corrected_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pvet_scale_corrected_per_frame.npy'),
                pvet_scale_corrected_per_frame)
        print('PVE-T SC: {:.5f}'.format(pvet_scale_corrected))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None, help='Path to network checkpoint')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of processes for data loading')
    args = parser.parse_args()

    # Device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = hmr(config.SMPL_MEAN_PARAMS).to(device)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    # Setup evaluation dataset
    dataset_path = '/scratch2/as2562/datasets/sports_videos_smpl/final_dataset'
    dataset = SportsVideosEvalDataset(dataset_path, img_wh=constants.IMG_RES)
    print("Eval examples found:", len(dataset))

    # Metrics
    metrics = ['pve', 'pve_scale_corrected', 'pve-t', 'pve_pa', 'pve-t_scale_corrected']

    save_path = '/data/cvfs/as2562/SPIN/evaluations/sports_videos_final_dataset'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Run evaluation
    evaluate_single_in_multitasknet_sports_videos(model,
                                                  dataset,
                                                  metrics,
                                                  device,
                                                  save_path,
                                                  num_workers=args.num_workers,
                                                  pin_memory=True,
                                                  vis_every_n_batches=1)




