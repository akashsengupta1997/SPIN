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
from utils.geometry import undo_keypoint_normalisation
from utils.cam_utils import orthographic_project_torch
from datasets.my_h36m_eval_dataset import H36MEvalDataset

import subsets

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def evaluate_single_in_multitasknet_h36m(model,
                                         eval_dataset,
                                         batch_size,
                                         metrics,
                                         device,
                                         save_path,
                                         num_workers=4,
                                         pin_memory=True,
                                         vis_every_n_batches=200):

    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 drop_last=True,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory)

    smpl_model = SMPL(config.SMPL_MODEL_DIR, batch_size=batch_size)
    smpl_model.to(device)

    J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
    J_regressor_batch = J_regressor[None, :].expand(batch_size, -1, -1).to(device)

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

    if 'mpjpe' in metrics:
        mpjpe_sum = 0.0
        mpjpe_per_frame = []

    if 'mpjpe_scale_corrected' in metrics:
        mpjpe_scale_corrected_sum = 0.0
        mpjpe_scale_corrected_per_frame = []

    if 'j3d_rec_err' in metrics:
        j3d_rec_err_sum = 0.0
        j3d_rec_err_per_frame = []

    if 'pve_2d' in metrics:
        pve_2d_sum = 0.0
        pve_2d_per_frame = []

    if 'pve_2d_scale_corrected' in metrics:
        pve_2d_scale_corrected_sum = 0.0
        pve_2d_scale_corrected_per_frame = []

    if 'pve_2d_pa' in metrics:
        pve_2d_pa_sum = 0.0
        pve_2d_pa_per_frame = []

    fname_per_frame = []
    pose_per_frame = []
    shape_per_frame = []
    cam_per_frame = []
    num_samples = 0
    num_vertices = 6890
    num_joints3d = 14

    model.eval()
    for batch_num, samples_batch in enumerate(tqdm(eval_dataloader)):
        # ------------------------------- TARGETS and INPUTS -------------------------------
        input = samples_batch['input']
        input = input.to(device)
        target_joints_h36m = samples_batch['target_j3d']
        target_pose = samples_batch['pose'].to(device)
        target_shape = samples_batch['shape'].to(device)

        target_smpl_output = smpl_model(body_pose=target_pose[:, 3:],
                                        global_orient=target_pose[:, :3],
                                        betas=target_shape)
        target_vertices = target_smpl_output.vertices
        target_reposed_smpl_output = smpl_model(betas=target_shape)
        target_reposed_vertices = target_reposed_smpl_output.vertices
        target_joints_h36mlsp = target_joints_h36m[:, constants.H36M_TO_J14, :]

        # ------------------------------- PREDICTIONS -------------------------------
        pred_rotmat, pred_betas, pred_camera = model(input)
        pred_output = smpl_model(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                 global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_vertices_projected2d = orthographic_project_torch(pred_vertices, pred_camera)
        pred_vertices_projected2d = undo_keypoint_normalisation(pred_vertices_projected2d, input.shape[-1])
        pred_reposed_smpl_output = smpl_model(betas=pred_betas)
        pred_reposed_vertices = pred_reposed_smpl_output.vertices

        pred_joints_h36m = torch.matmul(J_regressor_batch, pred_vertices)
        pred_pelvis = pred_joints_h36m[:, [0], :].clone()
        pred_joints_h36mlsp = pred_joints_h36m[:, constants.H36M_TO_J14, :] - pred_pelvis

        # Numpy-fying
        target_vertices = target_vertices.cpu().detach().numpy()
        target_reposed_vertices = target_reposed_vertices.cpu().detach().numpy()
        target_joints_h36mlsp = target_joints_h36mlsp.cpu().detach().numpy()

        pred_vertices = pred_vertices.cpu().detach().numpy()
        pred_vertices_projected2d = pred_vertices_projected2d.cpu().detach().numpy()
        pred_reposed_vertices = pred_reposed_vertices.cpu().detach().numpy()
        pred_joints_h36mlsp = pred_joints_h36mlsp.cpu().detach().numpy()
        pred_rotmat = pred_rotmat.cpu().detach().numpy()
        pred_betas = pred_betas.cpu().detach().numpy()
        pred_camera = pred_camera.cpu().detach().numpy()

        # ------------------------------- METRICS -------------------------------
        if 'pve' in metrics:
            pve_batch = np.linalg.norm(pred_vertices - target_vertices,
                                       axis=-1)  # (bs, 6890)
            pve_sum += np.sum(pve_batch)  # scalar
            pve_per_frame.append(np.mean(pve_batch, axis=-1))

        # Scale and translation correction
        if 'pve_scale_corrected' in metrics:
            pred_vertices_scale_corrected = scale_and_translation_transform_batch(pred_vertices,
                                                                                  target_vertices)
            pve_scale_corrected_batch = np.linalg.norm(pred_vertices_scale_corrected - target_vertices,
                                                       axis=-1)  # (bs, 6890)
            pve_scale_corrected_sum += np.sum(pve_scale_corrected_batch)  # scalar
            pve_scale_corrected_per_frame.append(np.mean(pve_scale_corrected_batch, axis=-1))

        # Procrustes analysis
        if 'pve_pa' in metrics:
            pred_vertices_pa = compute_similarity_transform_batch(pred_vertices, target_vertices)
            pve_pa_batch = np.linalg.norm(pred_vertices_pa - target_vertices, axis=-1)  # (bs, 6890)
            pve_pa_sum += np.sum(pve_pa_batch)  # scalar
            pve_pa_per_frame.append(np.mean(pve_pa_batch, axis=-1))

        if 'pve-t' in metrics:
            pvet_batch = np.linalg.norm(pred_reposed_vertices - target_reposed_vertices, axis=-1)
            pvet_sum += np.sum(pvet_batch)
            pvet_per_frame.append(np.mean(pvet_batch, axis=-1))

        # Scale and translation correction
        if 'pve-t_scale_corrected' in metrics:
            pred_reposed_vertices_sc = scale_and_translation_transform_batch(pred_reposed_vertices,
                                                                             target_reposed_vertices)
            pvet_scale_corrected_batch = np.linalg.norm(pred_reposed_vertices_sc - target_reposed_vertices,
                                                        axis=-1)  # (bs, 6890)
            pvet_scale_corrected_sum += np.sum(pvet_scale_corrected_batch)  # scalar
            pvet_scale_corrected_per_frame.append(np.mean(pvet_scale_corrected_batch, axis=-1))

        if 'mpjpe' in metrics:
            mpjpe_batch = np.linalg.norm(pred_joints_h36mlsp - target_joints_h36mlsp, axis=-1)  # (bs, 14)
            mpjpe_sum += np.sum(mpjpe_batch)
            mpjpe_per_frame.append(np.mean(mpjpe_batch, axis=-1))

        # Scale and translation correction
        if 'mpjpe_scale_corrected' in metrics:
            pred_joints_h36mlsp_sc = scale_and_translation_transform_batch(pred_joints_h36mlsp,
                                                                           target_joints_h36mlsp)
            mpjpe_scale_corrected_batch = np.linalg.norm(pred_joints_h36mlsp_sc - target_joints_h36mlsp,
                                                         axis=-1)  # (bs, 6890)
            mpjpe_scale_corrected_sum += np.sum(mpjpe_scale_corrected_batch)  # scalar
            mpjpe_scale_corrected_per_frame.append(np.mean(mpjpe_scale_corrected_batch, axis=-1))

        # Procrustes analysis
        if 'j3d_rec_err' in metrics:
            pred_joints_h36mlsp_pa = compute_similarity_transform_batch(pred_joints_h36mlsp, target_joints_h36mlsp)
            j3d_rec_err_batch = np.linalg.norm(pred_joints_h36mlsp_pa - target_joints_h36mlsp, axis=-1)  # (bs, 14)
            j3d_rec_err_sum += np.sum(j3d_rec_err_batch)
            j3d_rec_err_per_frame.append(np.mean(j3d_rec_err_batch, axis=-1))

        if 'pve_2d' in metrics:
            pred_vertices_2d = pred_vertices[:, :, :2]
            target_vertices_2d = target_vertices[:, :, :2]
            pve_2d_batch = np.linalg.norm(pred_vertices_2d - target_vertices_2d, axis=-1)  # (bs, 6890)
            pve_2d_sum += np.sum(pve_2d_batch)
            pve_2d_per_frame.append(np.mean(pve_2d_batch, axis=-1))

        # Scale and translation correction
        if 'pve_2d_scale_corrected' in metrics:
            pred_vertices_sc = scale_and_translation_transform_batch(pred_vertices,
                                                                     target_vertices)
            pred_vertices_2d_sc = pred_vertices_sc[:, :, :2]
            target_vertices_2d = target_vertices[:, :, :2]
            pve_2d_sc_batch = np.linalg.norm(pred_vertices_2d_sc - target_vertices_2d,
                                             axis=-1)  # (bs, 6890)
            pve_2d_scale_corrected_sum += np.sum(pve_2d_sc_batch)
            pve_2d_scale_corrected_per_frame.append(np.mean(pve_2d_sc_batch, axis=-1))

        # Procrustes analysis
        if 'pve_2d_pa' in metrics:
            pred_vertices_pa = compute_similarity_transform_batch(pred_vertices, target_vertices)
            pred_vertices_2d_pa = pred_vertices_pa[:, :, :2]
            target_vertices_2d = target_vertices[:, :, :2]
            pve_2d_pa_batch = np.linalg.norm(pred_vertices_2d_pa - target_vertices_2d, axis=-1)  # (bs, 6890)
            pve_2d_pa_sum += np.sum(pve_2d_pa_batch)
            pve_2d_pa_per_frame.append(np.mean(pve_2d_pa_batch, axis=-1))

        num_samples += target_pose.shape[0]
        fnames = samples_batch['fname']
        fname_per_frame.append(fnames)
        pose_per_frame.append(pred_rotmat)
        shape_per_frame.append(pred_betas)
        cam_per_frame.append(pred_camera)

        # ------------------------------- VISUALISE -------------------------------
        if vis_every_n_batches is not None:
            if batch_num % vis_every_n_batches == 0:
                vis_imgs = samples_batch['vis_img'].numpy()
                vis_imgs = np.transpose(vis_imgs, [0, 2, 3, 1])

                plt.figure(figsize=(16, 12))
                plt.subplot(341)
                plt.imshow(vis_imgs[0])

                plt.subplot(342)
                plt.imshow(vis_imgs[0])
                plt.scatter(pred_vertices_projected2d[0, :, 0], pred_vertices_projected2d[0, :, 1], s=0.1, c='r')

                plt.subplot(343)
                plt.scatter(target_vertices[0, :, 0], target_vertices[0, :, 1], s=0.2, c='b')
                # plt.scatter(pred_vertices[0, :, 0], pred_vertices[0, :, 1], s=0.1, c='r')
                plt.gca().invert_yaxis()
                plt.gca().set_aspect('equal', adjustable='box')

                plt.subplot(344)
                plt.scatter(target_vertices[0, :, 0], target_vertices[0, :, 1], s=0.2, c='b')
                plt.scatter(pred_vertices[0, :, 0], pred_vertices[0, :, 1], s=0.1, c='r')
                plt.gca().invert_yaxis()
                plt.gca().set_aspect('equal', adjustable='box')
                plt.text(-0.6, -0.8, s='PVE: {:.4f}'.format(pve_per_frame[-1][0]))

                plt.subplot(345)
                plt.scatter(target_vertices[0, :, 0], target_vertices[0, :, 1], s=0.2,
                            c='b')
                plt.scatter(pred_vertices_scale_corrected[0, :, 0],
                            pred_vertices_scale_corrected[0, :, 1], s=0.1,
                            c='r')
                plt.gca().invert_yaxis()
                plt.gca().set_aspect('equal', adjustable='box')
                plt.text(-0.6, -0.8, s='PVE-SC: {:.4f}'.format(pve_scale_corrected_per_frame[-1])[0])

                plt.subplot(346)
                plt.scatter(target_vertices[0, :, 2], target_vertices[0, :, 1], s=0.2,
                            c='b')
                plt.scatter(pred_vertices_scale_corrected[0, :, 2],
                            pred_vertices_scale_corrected[0, :, 1], s=0.1,
                            c='r')
                plt.gca().invert_yaxis()
                plt.gca().set_aspect('equal', adjustable='box')

                plt.subplot(347)
                plt.scatter(target_vertices[0, :, 0], target_vertices[0, :, 1], s=0.1, c='b')
                plt.scatter(pred_vertices_pa[0, :, 0], pred_vertices_pa[0, :, 1], s=0.1, c='r')
                plt.gca().invert_yaxis()
                plt.gca().set_aspect('equal', adjustable='box')
                plt.text(-0.6, -0.8, s='PVE-PA: {:.4f}'.format(pve_pa_per_frame[-1])[0])

                plt.subplot(348)
                plt.scatter(target_vertices[0, :, 2], target_vertices[0, :, 1], s=0.2, c='b')
                plt.scatter(pred_vertices_pa[0, :, 2], pred_vertices_pa[0, :, 1], s=0.1, c='r')
                plt.gca().invert_yaxis()
                plt.gca().set_aspect('equal', adjustable='box')

                plt.subplot(349)
                plt.scatter(target_reposed_vertices[0, :, 0], target_reposed_vertices[0, :, 1], s=0.1, c='b')
                plt.scatter(pred_reposed_vertices_sc[0, :, 0], pred_reposed_vertices_sc[0, :, 1], s=0.1, c='r')
                plt.gca().set_aspect('equal', adjustable='box')

                plt.subplot(3,4,10)
                for j in range(num_joints3d):
                    plt.scatter(pred_joints_h36mlsp[0, j, 0], pred_joints_h36mlsp[0, j, 1], c='r')
                    plt.scatter(target_joints_h36mlsp[0, j, 0], target_joints_h36mlsp[0, j, 1], c='b')
                    plt.text(pred_joints_h36mlsp[0, j, 0], pred_joints_h36mlsp[0, j, 1], s=str(j))
                    plt.text(target_joints_h36mlsp[0, j, 0], target_joints_h36mlsp[0, j, 1], s=str(j))
                plt.gca().invert_yaxis()
                plt.gca().set_aspect('equal', adjustable='box')

                plt.subplot(3, 4, 11)
                for j in range(num_joints3d):
                    plt.scatter(pred_joints_h36mlsp_sc[0, j, 0],
                                pred_joints_h36mlsp_sc[0, j, 1], c='r')
                    plt.scatter(target_joints_h36mlsp[0, j, 0],
                                target_joints_h36mlsp[0, j, 1], c='b')
                    plt.text(pred_joints_h36mlsp_sc[0, j, 0],
                             pred_joints_h36mlsp_sc[0, j, 1], s=str(j))
                    plt.text(target_joints_h36mlsp[0, j, 0],
                             target_joints_h36mlsp[0, j, 1], s=str(j))
                plt.gca().invert_yaxis()
                plt.gca().set_aspect('equal', adjustable='box')

                plt.subplot(3, 4, 12)
                for j in range(num_joints3d):
                    plt.scatter(pred_joints_h36mlsp_pa[0, j, 0], pred_joints_h36mlsp_pa[0, j, 1], c='r')
                    plt.scatter(target_joints_h36mlsp[0, j, 0], target_joints_h36mlsp[0, j, 1], c='b')
                    plt.text(pred_joints_h36mlsp_pa[0, j, 0], pred_joints_h36mlsp_pa[0, j, 1], s=str(j))
                    plt.text(target_joints_h36mlsp[0, j, 0], target_joints_h36mlsp[0, j, 1], s=str(j))
                plt.gca().invert_yaxis()
                plt.gca().set_aspect('equal', adjustable='box')

                # plt.show()
                save_fig_path = os.path.join(save_path, fnames[0])
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
        np.save(os.path.join(save_path, 'pvet_scale_corrected_per_frame.npy'), pvet_scale_corrected_per_frame)
        print('PVE-T SC: {:.5f}'.format(pvet_scale_corrected))

    if 'mpjpe' in metrics:
        mpjpe = mpjpe_sum / (num_samples * num_joints3d)
        mpjpe_per_frame = np.concatenate(mpjpe_per_frame, axis=0)
        np.save(os.path.join(save_path, 'mpjpe_per_frame.npy'), mpjpe_per_frame)
        print('MPJPE: {:.5f}'.format(mpjpe))

    if 'mpjpe_scale_corrected' in metrics:
        mpjpe_scale_corrected = mpjpe_scale_corrected_sum / (num_samples * num_joints3d)
        mpjpe_scale_corrected_per_frame = np.concatenate(mpjpe_scale_corrected_per_frame, axis=0)
        np.save(os.path.join(save_path, 'mpjpe_scale_corrected_per_frame.npy'),
                mpjpe_scale_corrected_per_frame)
        print('MPJPE SC: {:.5f}'.format(mpjpe_scale_corrected))

    if 'j3d_rec_err' in metrics:
        j3d_rec_err = j3d_rec_err_sum / (num_samples * num_joints3d)
        j3d_rec_err_per_frame = np.concatenate(j3d_rec_err_per_frame, axis=0)
        np.save(os.path.join(save_path, 'j3d_rec_err_per_frame.npy'), j3d_rec_err_per_frame)
        print('Rec Err: {:.5f}'.format(j3d_rec_err))

    if 'pve_2d' in metrics:
        pve_2d = pve_2d_sum / (num_samples * num_vertices)
        pve_2d_per_frame = np.concatenate(pve_2d_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pve_2d_per_frame.npy'), pve_2d_per_frame)
        print('PVE 2D: {:.5f}'.format(pve_2d))

    if 'pve_2d_scale_corrected' in metrics:
        pve_2d_scale_corrected = pve_2d_scale_corrected_sum / (num_samples * num_vertices)
        pve_2d_scale_corrected_per_frame = np.concatenate(pve_2d_scale_corrected_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pve_2d_scale_corrected_per_frame.npy'),
                pve_2d_scale_corrected_per_frame)
        print('PVE SC 2D: {:.5f}'.format(pve_2d_scale_corrected))

    if 'pve_2d_pa' in metrics:
        pve_2d_pa = pve_2d_pa_sum / (num_samples * num_vertices)
        pve_2d_pa_per_frame = np.concatenate(pve_2d_pa_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pve_2d_pa_per_frame.npy'), pve_2d_pa_per_frame)
        print('PVE 2D PA: {:.5f}'.format(pve_2d_pa))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None, help='Path to network checkpoint')
    parser.add_argument('--protocol', type=int, choices=[1, 2])
    parser.add_argument('--num_workers', default=4, type=int, help='Number of processes for data loading')
    args = parser.parse_args()

    # Device
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = hmr(config.SMPL_MEAN_PARAMS).to(device)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    # Setup evaluation dataset
    selected_fnames = subsets.H36M_OCCLUDED_JOINTS
    # selected_fnames = None
    print('Selected fnames:', selected_fnames)
    if selected_fnames is not None:
        vis_every_n_batches = 1
        batch_size = 1

    dataset_path = '/scratch2/as2562/datasets/H36M/eval'
    dataset = H36MEvalDataset(dataset_path, protocol=args.protocol, img_wh=constants.IMG_RES,
                              use_subset=False, selected_fnames=selected_fnames)
    print("Eval examples found:", len(dataset))

    # Metrics
    metrics = ['pve', 'pve-t', 'pve_pa', 'pve-t_pa', 'mpjpe', 'j3d_rec_err',
               'pve_2d', 'pve_2d_pa', 'pve_2d_scale_corrected',
               'pve_scale_corrected', 'pve-t_scale_corrected', 'mpjpe_scale_corrected']

    save_path = '/data/cvfs/as2562/SPIN/evaluations/h36m_protocol{}'.format(str(args.protocol))
    if selected_fnames is not None:
        save_path += '_selected_fnames_occluded_joints'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Run evaluation
    evaluate_single_in_multitasknet_h36m(model=model,
                                         eval_dataset=dataset,
                                         batch_size=8,
                                         metrics=metrics,
                                         device=device,
                                         save_path=save_path,
                                         num_workers=args.num_workers,
                                         pin_memory=True,
                                         vis_every_n_batches=100)







