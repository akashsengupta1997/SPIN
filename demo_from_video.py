"""
Runs SPIN on cropped and centred input video frames and save results in one pickle file.
"""
import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
import pickle
import os

from models import hmr, SMPL
from utils.imutils import crop
import config
import constants


def process_image(img_file, input_res=224):
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    img = cv2.imread(img_file)[:, :,
          ::-1].copy()  # PyTorch does not support negative stride at the moment
    # Assume that the person is centerered in the image
    height = img.shape[0]
    width = img.shape[1]
    center = np.array([width // 2, height // 2])
    scale = max(height, width) / 200
    img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2, 0, 1)
    norm_img = normalize_img(img.clone())[None]
    return img, norm_img


def predict_on_frames(args):
    # Load pretrained model
    model = hmr(config.SMPL_MEAN_PARAMS).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)

    # Load SMPL model
    smpl = SMPL(config.SMPL_MODEL_DIR,
                batch_size=1,
                create_transl=False).to(device)
    model.eval()

    image_paths = [os.path.join(args.in_folder, f) for f in sorted(os.listdir(args.in_folder))
                   if f.endswith('.png')]
    print('Predicting on all png images in {}'.format(args.in_folder))

    all_vertices = []
    all_poses = []
    all_betas = []
    all_cams = []

    for image_path in image_paths:
        print("Image: ", image_path)
        # Preprocess input image and generate predictions
        img, norm_img = process_image(image_path, input_res=constants.IMG_RES)
        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera = model(norm_img.to(device))
            pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                               global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices

        pred_vertices = pred_vertices.cpu().numpy()
        pred_rotmat = pred_rotmat.cpu().numpy()
        pred_betas = pred_betas.cpu().numpy()
        pred_camera = pred_camera.cpu().numpy()

        all_vertices.append(pred_vertices)
        all_cams.append(pred_camera)
        all_betas.append(pred_betas)
        all_poses.append(pred_rotmat)

    # Save predictions as pkl
    all_vertices = np.concatenate(all_vertices, axis=0)
    all_cams = np.concatenate(all_cams, axis=0)
    all_poses = np.concatenate(all_poses, axis=0)
    all_betas = np.concatenate(all_betas, axis=0)

    pred_dict = {'verts': all_vertices,
                 'pose': all_poses,
                 'betas': all_betas,
                 'pred_cam': all_cams}
    if args.out_folder == 'dataset':
        out_folder = args.in_folder.replace('cropped_frames', 'spin_results')
    else:
        out_folder = args.out_folder
    print('Saving to', os.path.join(out_folder, 'spin_results.pkl'))
    os.makedirs(out_folder, exist_ok=True)
    for key in pred_dict.keys():
        print(pred_dict[key].shape)
    with open(os.path.join(out_folder, 'spin_results.pkl'), 'wb') as f:
        pickle.dump(pred_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None, help='Path to pretrained checkpoint')
    parser.add_argument('--in_folder', type=str, required=True,
                        help='Path to input frames folder.')
    parser.add_argument('--out_folder', type=str, default=None,
                        help='Folder to save predictions pickle in')
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    predict_on_frames(args)


