import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import cv2

import config
import constants
from models import hmr, SMPL
from utils.pose_utils import compute_similarity_transform_batch
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

    if 'pve_pa' in metrics:
        pve_pa_sum = 0.0

    if 'pve-t' in metrics:
        pvet_sum = 0.0

    if 'pve-t_pa' in metrics:
        pvet_pa_sum = 0.0

    num_samples = 0
    num_vertices = 6890

    all_frame_paths = []
    pve_per_frame = []
    pve_pa_per_frame = []
    pvet_per_frame = []
    pvet_pa_per_frame = []

    model.eval()
    for batch_num, samples_batch in enumerate(tqdm(eval_dataloader)):
        input = samples_batch['input']
        input = input.to(device)

        target_shape = samples_batch['shape']
        target_shape = target_shape.to(device)
        target_vertices = samples_batch['vertices']

        pred_rotmat, pred_betas, pred_camera = model(input)
        pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                           global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_vertices_projected2d = orthographic_project_torch(pred_vertices, pred_camera)
        pred_vertices_projected2d = undo_keypoint_normalisation(pred_vertices_projected2d, input.shape[-1])
        pred_reposed_smpl_output = smpl(betas=pred_betas)
        pred_reposed_vertices = pred_reposed_smpl_output.vertices

        target_gender = samples_batch['gender'][0]
        if target_gender == 'm':
            target_reposed_smpl_output = smpl_male(betas=target_shape)
        elif target_gender == 'f':
            target_reposed_smpl_output = smpl_female(betas=target_shape)
        target_reposed_vertices = target_reposed_smpl_output.vertices

        # Numpy-fying
        target_vertices = target_vertices.cpu().detach().numpy()
        target_reposed_vertices = target_reposed_vertices.cpu().detach().numpy()

        pred_vertices = pred_vertices.cpu().detach().numpy()
        pred_vertices_projected2d = pred_vertices_projected2d.cpu().detach().numpy()
        pred_reposed_vertices = pred_reposed_vertices.cpu().detach().numpy()

        if 'pve' in metrics:
            pve_batch = np.linalg.norm(pred_vertices - target_vertices,
                                       axis=-1)  # (1, 6890)
            pve_sum += np.sum(pve_batch)  # scalar
            pve_per_frame.append(np.mean(pve_batch))

        # Procrustes analysis
        if 'pve_pa' in metrics:
            pred_vertices_pa = compute_similarity_transform_batch(pred_vertices, target_vertices)
            pve_pa_batch = np.linalg.norm(pred_vertices_pa - target_vertices, axis=-1)  # (1, 6890)
            pve_pa_sum += np.sum(pve_pa_batch)  # scalar
            pve_pa_per_frame.append(np.mean(pve_pa_batch))

        if 'pve-t' in metrics:
            pvet_batch = np.linalg.norm(pred_reposed_vertices - target_reposed_vertices, axis=-1)
            pvet_sum += np.sum(pvet_batch)
            pvet_per_frame.append(np.mean(pvet_batch))

        # Procrustes analysis
        if 'pve-t_pa' in metrics:
            pred_reposed_vertices_pa = compute_similarity_transform_batch(pred_reposed_vertices,
                                                                          target_reposed_vertices)
            pvet_pa_batch = np.linalg.norm(pred_reposed_vertices_pa - target_reposed_vertices, axis=-1)  # (1, 6890)
            pvet_pa_sum += np.sum(pvet_pa_batch)  # scalar
            pvet_pa_per_frame.append(np.mean(pvet_pa_batch))

        num_samples += target_shape.shape[0]

        frame_path = samples_batch['frame_path']
        all_frame_paths.append(frame_path)

        # Visualise
        if batch_num % vis_every_n_batches == 0:
            vis_imgs = samples_batch['vis_img'].numpy()
            vis_imgs = np.transpose(vis_imgs, [0, 2, 3, 1])

            for i in range(1):
                plt.figure(figsize=(12, 8))
                plt.subplot(231)
                plt.imshow(vis_imgs[i])

                plt.subplot(232)
                plt.imshow(vis_imgs[i])
                plt.scatter(pred_vertices_projected2d[i, :, 0], pred_vertices_projected2d[i, :, 1], s=0.1, c='r')

                plt.subplot(233)
                plt.scatter(target_vertices[i, :, 0], target_vertices[i, :, 1], s=0.1, c='b')
                plt.scatter(pred_vertices[i, :, 0], pred_vertices[i, :, 1], s=0.05, c='r')
                plt.gca().invert_yaxis()
                plt.gca().set_aspect('equal', adjustable='box')

                plt.subplot(234)
                plt.scatter(target_vertices[i, :, 0], target_vertices[i, :, 1], s=0.1, c='b')
                plt.scatter(pred_vertices_pa[i, :, 0], pred_vertices_pa[i, :, 1], s=0.05, c='r')
                plt.gca().invert_yaxis()
                plt.gca().set_aspect('equal', adjustable='box')

                plt.subplot(235)
                plt.scatter(target_reposed_vertices[i, :, 0], target_reposed_vertices[i, :, 1], s=0.1, c='b')
                plt.scatter(pred_reposed_vertices[i, :, 0], pred_reposed_vertices[i, :, 1], s=0.05, c='r')
                plt.gca().set_aspect('equal', adjustable='box')

                plt.subplot(236)
                plt.scatter(target_reposed_vertices[i, :, 0], target_reposed_vertices[i, :, 1], s=0.1, c='b')
                plt.scatter(pred_reposed_vertices_pa[i, :, 0], pred_reposed_vertices_pa[i, :, 1], s=0.05, c='r')
                plt.gca().set_aspect('equal', adjustable='box')

                # plt.show()
                split_path = frame_path[i].split('/')
                clip_name = split_path[-3]
                frame_num = split_path[-1]
                save_fig_path = os.path.join(save_path, clip_name + '_' + frame_num)
                plt.savefig(save_fig_path, bbox_inches='tight')
                plt.close()

    if 'pve' in metrics:
        pve = pve_sum / (num_samples * num_vertices)
        print('PVE: {:.5f}'.format(pve))

    if 'pve_pa' in metrics:
        pve_pa = pve_pa_sum / (num_samples * num_vertices)
        print('PVE PA: {:.5f}'.format(pve_pa))

    if 'pve-t' in metrics:
        pvet = pvet_sum / (num_samples * num_vertices)
        print('PVE-T: {:.5f}'.format(pvet))

    if 'pve-t_pa' in metrics:
        pvet_pa = pvet_pa_sum / (num_samples * num_vertices)
        print('PVE-T PA: {:.5f}'.format(pvet_pa))

        # Save per frame metrics
    metrics_save_file = os.path.join(save_path, 'metrics.txt')
    with open(metrics_save_file, 'w') as f_metrics:
        for i in range(len(all_frame_paths)):
            f_metrics.write('{} PVE: {:.5f} PVE PA: {:.5f} PVE-T: {:.5f} PVE-T PA: {:.5f}\n'.format(all_frame_paths[i],
                                                                                                    pve_per_frame[i],
                                                                                                    pve_pa_per_frame[i],
                                                                                                    pvet_per_frame[i],
                                                                                                    pvet_pa_per_frame[
                                                                                                        i]))
        f_metrics.write('Full dataset metrics: '
                        'PVE: {:.5f} PVE PA: {:.5f} PVE-T: {:.5f} PVE-T PA: {:.5f}\n'.format(pve,
                                                                                             pve_pa,
                                                                                             pvet,
                                                                                             pvet_pa))


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
    dataset_path = '/scratch2/as2562/datasets/sports_videos_smpl/draft_dataset1'
    dataset = SportsVideosEvalDataset(dataset_path, img_wh=constants.IMG_RES)
    print("Eval examples found:", len(dataset))

    # Metrics
    metrics = ['pve', 'pve-t', 'pve_pa', 'pve-t_pa']

    save_path = '/data/cvfs/as2562/SPIN/evaluations/sports_videos_draft_dataset1'
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




