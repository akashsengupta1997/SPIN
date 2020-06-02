import cv2
import numpy as np
import os
import pickle
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from utils.imutils import convert_bbox_centre_hw_to_corners
import constants


class SSP3DEvalDataset(Dataset):
    def __init__(self, ssp3d_dir_path, img_wh,
                 bbox_scale_factor=1.2, eval_img_wh=256):
        super(SSP3DEvalDataset, self).__init__()

        # Paths
        self.images_dir = os.path.join(ssp3d_dir_path, 'images')
        self.pointrend_masks_dir = os.path.join(ssp3d_dir_path, 'silhouettes')
        
        # Data
        data = np.load(os.path.join(ssp3d_dir_path, 'labels.npz'))

        self.frame_fnames = data['fnames']
        self.body_shapes = data['shapes']
        self.body_poses = data['poses']
        self.kprcnn_keypoints = data['joints2D']
        self.bbox_centres = data['bbox_centres']  # Tight bounding box centre
        self.bbox_whs = data['bbox_whs']  # Tight bounding box width/height
        self.genders = data['genders']
        assert len(self.frame_fnames) == len(self.body_shapes) == len(self.kprcnn_keypoints) == len(self.genders)

        self.img_wh = img_wh
        self.eval_img_wh = eval_img_wh
        self.bbox_scale_factor = bbox_scale_factor
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)

    def __len__(self):
        return len(self.frame_fnames)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # Inputs
        fname = self.frame_fnames[index]
        frame_path = os.path.join(self.images_dir, fname)

        img = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
        bbox_centre = self.bbox_centres[index]
        bbox_wh = self.bbox_whs[index] * self.bbox_scale_factor
        bbox_corners = convert_bbox_centre_hw_to_corners(bbox_centre, bbox_wh, bbox_wh)
        top_left = bbox_corners[:2].astype(np.int16)
        bottom_right = bbox_corners[2:].astype(np.int16)
        top_left[top_left < 0] = 0
        bottom_right[bottom_right < 0] = 0
        img = img[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1]]
        orig_height, orig_width = img.shape[:2]
        img = cv2.resize(img, (self.img_wh, self.img_wh), interpolation=cv2.INTER_LINEAR)
        img = np.transpose(img, [2, 0, 1]) / 255.0

        # Target 2D Joints
        keypoints = self.kprcnn_keypoints[index]
        keypoints = keypoints[:, :2] - top_left[::-1]
        keypoints = keypoints[:, :2] * np.array([self.eval_img_wh / float(orig_width),
                                                 self.eval_img_wh / float(orig_height)])

        # Target 3D
        shape = self.body_shapes[index]
        pose = self.body_poses[index]
        gender = self.genders[index]

        # Target Silhouette
        pointrend_mask_path = os.path.join(self.pointrend_masks_dir, fname)
        silhouette = cv2.imread(pointrend_mask_path, 0)
        silhouette = silhouette[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1]]
        silhouette = cv2.resize(silhouette, (self.eval_img_wh, self.eval_img_wh),
                                interpolation=cv2.INTER_NEAREST)

        img = torch.from_numpy(img).float()
        pose = torch.from_numpy(pose).float()
        shape = torch.from_numpy(shape).float()

        input = self.normalize_img(img)

        return {'input': input,
                'shape': shape,
                'pose': pose,
                'vis_img': img,
                'silhouette': silhouette,
                'keypoints': keypoints,
                'fname': fname,
                'gender': gender}
