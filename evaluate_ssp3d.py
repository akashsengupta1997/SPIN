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
from models import hmr, SMPL, NMRRenderer
from utils.pose_utils import compute_similarity_transform_batch, \
    scale_and_translation_transform_batch
from utils.geometry import undo_keypoint_normalisation
from utils.label_conversions import convert_multiclass_to_binary_labels_torch
from utils.cam_utils import get_intrinsics_matrix, \
    batch_convert_weak_perspective_to_camera_translation, orthographic_project_torch
from datasets.ssp3d_eval_dataset import SSP3DEvalDataset


def evaluate_single_in_multitasknet_ssp3d(model,
                                          eval_dataset,
                                          metrics,
                                          device,
                                          save_path,
                                          num_workers=4,
                                          pin_memory=True,
                                          vis_every_n_batches=1,
                                          output_img_wh=256):

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

    if 'pves' in metrics:
        pve_sum = 0.0
        pve_per_frame = []

    if 'pves_sc' in metrics:
        pve_scale_corrected_sum = 0.0
        pve_scale_corrected_per_frame = []

    if 'pves_pa' in metrics:
        pve_pa_sum = 0.0
        pve_pa_per_frame = []

    if 'pve-ts' in metrics:
        pvet_sum = 0.0
        pvet_per_frame = []

    if 'pve-ts_sc' in metrics:
        pvet_scale_corrected_sum = 0.0
        pvet_scale_corrected_per_frame = []

    if 'silhouette_ious' in metrics:
        # Set-up NMR renderer to render silhouettes from predicted vertex meshes.
        # Assuming camera rotation is identity (since it is dealt with by global_orients)
        cam_K = get_intrinsics_matrix(output_img_wh, output_img_wh,
                                      constants.FOCAL_LENGTH)
        cam_K = torch.from_numpy(cam_K.astype(np.float32)).to(device)
        cam_R = torch.eye(3).to(device)
        cam_K = cam_K[None, :, :]
        cam_R = cam_R[None, :, :]
        nmr_parts_renderer = NMRRenderer(1,
                                         cam_K,
                                         cam_R,
                                         output_img_wh,
                                         rend_parts_seg=True).to(device)
        num_true_positives = 0.0
        num_false_positives = 0.0
        num_true_negatives = 0.0
        num_false_negatives = 0.0
        silhouette_iou_per_frame = []

    if 'joints2D_l2es' in metrics:
        j2d_l2e_sum = 0.0
        j2d_l2e_per_frame = []

    fname_per_frame = []
    pose_per_frame = []
    shape_per_frame = []
    cam_per_frame = []
    num_samples = 0
    num_vertices = 6890
    num_joints2d = 17

    model.eval()
    for batch_num, samples_batch in enumerate(tqdm(eval_dataloader)):
        # ------------------------------- TARGETS and INPUTS -------------------------------
        input = samples_batch['input']
        input = input.to(device)

        target_shape = samples_batch['shape'].to(device)
        target_pose = samples_batch['pose'].to(device)
        target_silhouette = samples_batch['silhouette']
        target_joints2d_coco = samples_batch['keypoints']
        target_gender = samples_batch['gender'][0]

        if target_gender == 'm':
            target_smpl_output = smpl_male(body_pose=target_pose[:, 3:],
                                           global_orient=target_pose[:, :3],
                                           betas=target_shape)
            target_reposed_smpl_output = smpl_male(betas=target_shape)
        elif target_gender == 'f':
            target_smpl_output = smpl_female(body_pose=target_pose[:, 3:],
                                             global_orient=target_pose[:, :3],
                                             betas=target_shape)
            target_reposed_smpl_output = smpl_female(betas=target_shape)
        target_vertices = target_smpl_output.vertices
        target_reposed_vertices = target_reposed_smpl_output.vertices

        # ------------------------------- PREDICTIONS -------------------------------
        pred_rotmat, pred_betas, pred_camera = model(input)
        pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                           global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_vertices_projected2d = orthographic_project_torch(pred_vertices, pred_camera)
        pred_vertices_projected2d = undo_keypoint_normalisation(pred_vertices_projected2d, input.shape[-1])

        pred_joints_all = pred_output.joints
        pred_joints_coco = pred_joints_all[:, 49:, :]
        pred_joints2d_coco = orthographic_project_torch(pred_joints_coco, pred_camera)
        pred_joints2d_coco = undo_keypoint_normalisation(pred_joints2d_coco, output_img_wh)

        pred_reposed_smpl_output = smpl(betas=pred_betas)
        pred_reposed_vertices = pred_reposed_smpl_output.vertices

        pred_camera = pred_camera.cpu().detach().numpy()
        if 'silhouette_ious' in metrics:
            pred_cam_ts = batch_convert_weak_perspective_to_camera_translation(pred_camera,
                                                                               constants.FOCAL_LENGTH,
                                                                               output_img_wh)
            pred_cam_ts = torch.from_numpy(pred_cam_ts).float().to(device)
            part_seg = nmr_parts_renderer(pred_vertices, pred_cam_ts.unsqueeze(0))
            pred_silhouette = convert_multiclass_to_binary_labels_torch(part_seg)
            pred_silhouette = pred_silhouette.cpu().detach().numpy()

        # Numpy-fying
        target_vertices = target_vertices.cpu().detach().numpy()
        target_reposed_vertices = target_reposed_vertices.cpu().detach().numpy()
        target_silhouette = target_silhouette.numpy()
        target_joints2d_coco = target_joints2d_coco.numpy()

        pred_vertices = pred_vertices.cpu().detach().numpy()
        pred_vertices_projected2d = pred_vertices_projected2d.cpu().detach().numpy()
        pred_reposed_vertices = pred_reposed_vertices.cpu().detach().numpy()
        pred_rotmat = pred_rotmat.cpu().detach().numpy()
        pred_betas = pred_betas.cpu().detach().numpy()
        pred_joints2d_coco = pred_joints2d_coco.cpu().detach().numpy()

        # ------------------------------- METRICS -------------------------------
        if 'pves' in metrics:
            pve_batch = np.linalg.norm(pred_vertices - target_vertices, axis=-1)  # (1, 6890)
            pve_sum += np.sum(pve_batch)  # scalar
            pve_per_frame.append(np.mean(pve_batch, axis=-1))

        # Scale and translation correction
        if 'pves_sc' in metrics:
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

        if 'pve-ts' in metrics:
            pvet_batch = np.linalg.norm(pred_reposed_vertices - target_reposed_vertices, axis=-1)
            pvet_sum += np.sum(pvet_batch)
            pvet_per_frame.append(np.mean(pvet_batch, axis=-1))

        # Scale and translation correction
        if 'pve-ts_sc' in metrics:
            pred_reposed_vertices_sc = scale_and_translation_transform_batch(
                pred_reposed_vertices,
                target_reposed_vertices)
            pvet_scale_corrected_batch = np.linalg.norm(
                pred_reposed_vertices_sc - target_reposed_vertices,
                axis=-1)  # (bs, 6890)
            pvet_scale_corrected_sum += np.sum(pvet_scale_corrected_batch)  # scalar
            pvet_scale_corrected_per_frame.append(
                np.mean(pvet_scale_corrected_batch, axis=-1))

        if 'silhouette_ious' in metrics:
            pred_silhouette = np.round(pred_silhouette).astype(np.bool)
            target_silhouette = np.round(target_silhouette).astype(np.bool)

            true_positive = np.logical_and(pred_silhouette, target_silhouette)
            false_positive = np.logical_and(pred_silhouette,
                                            np.logical_not(target_silhouette))
            true_negative = np.logical_and(np.logical_not(pred_silhouette),
                                           np.logical_not(target_silhouette))
            false_negative = np.logical_and(np.logical_not(pred_silhouette),
                                            target_silhouette)

            num_tp = int(np.sum(true_positive))
            num_fp = int(np.sum(false_positive))
            num_tn = int(np.sum(true_negative))
            num_fn = int(np.sum(false_negative))

            num_true_positives += num_tp
            num_false_positives += num_fp
            num_true_negatives += num_tn
            num_false_negatives += num_fn

            silhouette_iou_per_frame.append(num_tp/(num_tp + num_fp + num_fn))

        if 'joints2D_l2es' in metrics:
            j2d_l2e_batch = np.linalg.norm(pred_joints2d_coco - target_joints2d_coco,
                                           axis=-1)  # (bs, 17)
            j2d_l2e_sum += np.sum(j2d_l2e_batch)  # scalar
            j2d_l2e_per_frame.append(np.mean(j2d_l2e_batch, axis=-1))

        num_samples += target_shape.shape[0]

        fname = samples_batch['fname']
        fname_per_frame.append(fname)
        pose_per_frame.append(pred_rotmat)
        shape_per_frame.append(pred_betas)
        cam_per_frame.append(pred_camera)

        # ------------------------------- VISUALISE -------------------------------
        if batch_num % vis_every_n_batches == 0:
            vis_imgs = samples_batch['vis_img'].numpy()
            vis_imgs = np.transpose(vis_imgs, [0, 2, 3, 1])

            plt.figure(figsize=(12, 8))
            plt.subplot(241)
            plt.imshow(vis_imgs[0])

            plt.subplot(242)
            plt.imshow(vis_imgs[0])
            plt.scatter(pred_vertices_projected2d[0, :, 0], pred_vertices_projected2d[0, :, 1], s=0.1, c='r')

            if 'silhouette_iou' in metrics:
                plt.subplot(243)
                plt.imshow(pred_silhouette[0].astype(np.int16) -
                           target_silhouette[0].astype(np.int16))
                if 'j2d_l2e' in metrics:
                    for j in range(target_joints2d_coco.shape[1]):
                        plt.scatter(target_joints2d_coco[0, j, 0],
                                    target_joints2d_coco[0, j, 1],
                                    c='b', s=10.0)
                        plt.text(target_joints2d_coco[0, j, 0], target_joints2d_coco[0, j, 1],
                                 str(j))
                        plt.scatter(pred_joints2d_coco[0, j, 0],
                                    pred_joints2d_coco[0, j, 1],
                                    c='r', s=10.0)
                        plt.text(pred_joints2d_coco[0, j, 0], pred_joints2d_coco[0, j, 1],
                                 str(j))

            plt.subplot(244)
            plt.scatter(target_vertices[0, :, 0], target_vertices[0, :, 1], s=0.1, c='b')
            plt.scatter(pred_vertices[0, :, 0], pred_vertices[0, :, 1], s=0.05, c='r')
            plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal', adjustable='box')

            plt.subplot(245)
            plt.scatter(target_vertices[0, :, 0], target_vertices[0, :, 1], s=0.1, c='b')
            plt.scatter(pred_vertices_pa[0, :, 0], pred_vertices_pa[0, :, 1], s=0.05, c='r')
            plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal', adjustable='box')

            plt.subplot(246)
            plt.scatter(target_reposed_vertices[0, :, 0], target_reposed_vertices[0, :, 1], s=0.1, c='b')
            plt.scatter(pred_reposed_vertices[0, :, 0], pred_reposed_vertices[0, :, 1], s=0.05, c='r')
            plt.gca().set_aspect('equal', adjustable='box')

            plt.subplot(247)
            plt.scatter(target_reposed_vertices[0, :, 0], target_reposed_vertices[0, :, 1], s=0.1, c='b')
            plt.scatter(pred_reposed_vertices_sc[0, :, 0], pred_reposed_vertices_sc[0, :, 1], s=0.05, c='r')
            plt.gca().set_aspect('equal', adjustable='box')

            save_fig_path = os.path.join(save_path, fname[0])
            plt.savefig(save_fig_path, bbox_inches='tight')
            plt.close()

    # ------------------------------- DISPLAY METRICS AND SAVE PER-FRAME METRICS -------------------------------
    fname_per_frame = np.concatenate(fname_per_frame, axis=0)
    np.save(os.path.join(save_path, 'fname_per_frame.npy'), fname_per_frame)

    pose_per_frame = np.concatenate(pose_per_frame, axis=0)
    np.save(os.path.join(save_path, 'pose_per_frame.npy'), pose_per_frame)

    shape_per_frame = np.concatenate(shape_per_frame, axis=0)
    np.save(os.path.join(save_path, 'shape_per_frame.npy'), shape_per_frame)

    cam_per_frame = np.concatenate(cam_per_frame, axis=0)
    np.save(os.path.join(save_path, 'cam_per_frame.npy'), cam_per_frame)

    if 'pves' in metrics:
        pve = pve_sum / (num_samples * num_vertices)
        pve_per_frame = np.concatenate(pve_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pve_per_frame.npy'), pve_per_frame)
        print('PVE: {:.5f}'.format(pve))

    if 'pves_sc' in metrics:
        pve_scale_corrected = pve_scale_corrected_sum / (num_samples * num_vertices)
        pve_scale_corrected_per_frame = np.concatenate(pve_scale_corrected_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pve_scale_corrected_per_frame.npy'),
                pve_scale_corrected_per_frame)
        print('PVE SC: {:.5f}'.format(pve_scale_corrected))

    if 'pves_pa' in metrics:
        pve_pa = pve_pa_sum / (num_samples * num_vertices)
        pve_pa_per_frame = np.concatenate(pve_pa_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pve_pa_per_frame.npy'), pve_pa_per_frame)
        print('PVE PA: {:.5f}'.format(pve_pa))

    if 'pve-ts' in metrics:
        pvet = pvet_sum / (num_samples * num_vertices)
        pvet_per_frame = np.concatenate(pvet_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pvet_per_frame.npy'), pvet_per_frame)
        print('PVE-T: {:.5f}'.format(pvet))

    if 'pve-ts_sc' in metrics:
        pvet_scale_corrected = pvet_scale_corrected_sum / (num_samples * num_vertices)
        pvet_scale_corrected_per_frame = np.concatenate(pvet_scale_corrected_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pvet_scale_corrected_per_frame.npy'),
                pvet_scale_corrected_per_frame)
        print('PVE-T SC: {:.5f}'.format(pvet_scale_corrected))

    if 'silhouette_ious' in metrics:
        mean_iou = num_true_positives / (
                num_true_positives + num_false_negatives + num_false_positives)
        global_acc = (num_true_positives + num_true_negatives) / (
                num_true_positives + num_true_negatives + num_false_negatives + num_false_positives)
        np.save(os.path.join(save_path, 'silhouette_iou_per_frame.npy'),
                silhouette_iou_per_frame)
        print('Mean IOU: {:.3f}'.format(mean_iou))
        print('Global Acc: {:.3f}'.format(global_acc))

    if 'joints2D_l2es' in metrics:
        j2d_l2e = j2d_l2e_sum / (num_samples * num_joints2d)
        j2d_l2e_per_frame = np.concatenate(j2d_l2e_per_frame, axis=0)
        np.save(os.path.join(save_path, 'j2d_l2e_per_frame_per_frame.npy'),
                j2d_l2e_per_frame)
        print('J2D L2 Error: {:.5f}'.format(j2d_l2e))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='data/model_checkpoint.pt', help='Path to network checkpoint')
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--num_workers', default=4, type=int, help='Number of processes for data loading')
    args = parser.parse_args()

    # Device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = hmr(config.SMPL_MEAN_PARAMS).to(device)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    # Setup evaluation dataset
    dataset_path = '/scratch/as2562/datasets/ssp_3d'
    dataset = SSP3DEvalDataset(dataset_path, img_wh=constants.IMG_RES)
    print("Eval examples found:", len(dataset))

    # Metrics
    metrics = ['pves', 'pves_sc', 'pves_pa', 'pve-ts', 'pve-ts_sc',
               'silhouette_ious', 'joints2D_l2es']

    save_path = '/data/cvfs/as2562/SPIN/evaluations/ssp3d'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Run evaluation
    evaluate_single_in_multitasknet_ssp3d(model,
                                          dataset,
                                          metrics,
                                          device,
                                          save_path,
                                          num_workers=args.num_workers,
                                          pin_memory=True,
                                          vis_every_n_batches=1)




