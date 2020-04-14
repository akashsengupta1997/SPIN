import torch
import torch.nn as nn
import numpy as np

import neural_renderer as nr
import config

from models import SMPL


class NMRRenderer(nn.Module):
    def __init__(self,
                 batch_size,
                 cam_K,
                 cam_R,
                 img_wh=256,
                 rend_parts_seg=False):
        """
        """
        super(NMRRenderer, self).__init__()

        faces = SMPL(config.SMPL_MODEL_DIR).faces.astype(np.int32)
        faces = torch.from_numpy(faces.astype(np.int32))
        faces = faces[None, :].expand(batch_size, -1, -1)
        self.register_buffer('faces', faces)

        if rend_parts_seg:
            textures = np.load(config.VERTEX_TEXTURE_FILE)
            textures = torch.from_numpy(textures).float()
            textures = textures.expand(batch_size, -1, -1, -1, -1, -1)
            self.register_buffer('textures', textures)

            cube_parts = np.load(config.CUBE_PARTS_FILE)
            cube_parts = torch.from_numpy(cube_parts).float()
            self.register_buffer('cube_parts', cube_parts)
        else:
            texture_size = 2
            textures = torch.ones(batch_size, self.faces.shape[1], texture_size, texture_size,
                                  texture_size, 3, dtype=torch.float32)
            self.register_buffer('textures', textures)

        # Setup renderer
        renderer = nr.Renderer(camera_mode='projection',
                               K=cam_K,
                               R=cam_R,
                               image_size=img_wh,
                               orig_size=img_wh,
                               light_direction=[0, 0, 1])
        if rend_parts_seg:
            renderer.light_intensity_ambient = 1
            renderer.anti_aliasing = False
            renderer.light_intensity_directional = 0
        self.renderer = renderer

        self.rend_parts_seg = rend_parts_seg

    def forward(self, vertices, cam_ts):
        """
        :param vertices: (B, N, 3)
        :param cam_ts: (B, 1, 3)
        """
        if cam_ts.ndim == 2:
            cam_ts = cam_ts.unsqueeze(1)
        if self.rend_parts_seg:
            parts, _, mask = self.renderer(vertices, self.faces, self.textures,
                                           t=cam_ts)
            parts = self.get_parts(parts, mask)
            return parts
        else:
            rend_image, depth, _ = self.renderer(vertices, self.faces, self.textures,
                                                 t=cam_ts)
            return rend_image, depth

    def get_parts(self, parts, mask):
        """Process renderer part image to get body part indices."""
        bn,c,h,w = parts.shape
        mask = mask.view(-1,1)
        parts_index = torch.floor(100*parts.permute(0,2,3,1).contiguous().view(-1,3)).long()
        parts = self.cube_parts[parts_index[:,0], parts_index[:,1], parts_index[:,2], None]
        parts *= mask
        parts = parts.view(bn,h,w).long()
        return parts