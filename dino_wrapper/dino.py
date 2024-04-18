
import torch
import numpy as np
from sklearn.decomposition import PCA

from PIL import Image

from .masked_correspondences import get_model
from .masked_correspondences import compute_features
from .masked_correspondences import chunk_cosine_sim

class Dino():

    def __init__(self, mode, layer, facet, bin, thresh):
        self.extractor, self.device = get_model()
        self.mode   = mode
        self.layer  = layer
        self.facet  = facet
        self.bin    = bin
        self.thresh = thresh


    def run_dino(self, images, permute=False):

        descriptors   = []
        num_patches   = []
        saliency_maps = []
        fg_masks      = []

        img_size = [ images[0].shape[2], images[0].shape[2] ]

        for el in images:
            el = el.clone()
            if permute:
                el = el.permute(0, 3, 1, 2)
            out = compute_features(self.extractor, self.device, el, img_size, self.layer, self.facet, self.bin, self.thresh)

            descriptors   += [ d.unsqueeze(0).cpu().clone() for d in out[0] ]
            saliency_maps += [ saliency.cpu() for saliency in out[2].reshape(el.shape[0], -1) ]
            fg_masks      += [ fg.cpu() for fg in out[3].reshape(el.shape[0], -1) ]

        num_patches = out[1]

        return descriptors, num_patches, saliency_maps, fg_masks

    def extract_fg_mask(self, descriptors_list, fg_list):
        descriptors = torch.cat(descriptors_list, dim=-2).squeeze()

        pca = PCA(n_components=4).fit(descriptors.numpy())
        pca_descriptors = pca.transform(descriptors.numpy())

        fg_mask_saliency = torch.cat(fg_list, dim=0) # used to select sign

        if pca_descriptors[fg_mask_saliency, 0].mean() > 0.0:
            fg_mask = pca_descriptors[:, 0] > 0
        else:
            fg_mask = pca_descriptors[:, 0] < 0

        return torch.from_numpy(fg_mask).bool()


    def features_to_matches(self, descriptors1, descriptors2, fg_mask1, fg_mask2, num_patches):

        fg_mask = self.extract_fg_mask([descriptors1, descriptors2], [fg_mask1, fg_mask2])

        fg_mask1 = fg_mask[:fg_mask.shape[0] // 2]
        fg_mask2 = fg_mask[fg_mask.shape[0] // 2:]

        descriptors1 = descriptors1.unsqueeze(0).to(self.device)
        descriptors2 = descriptors2.unsqueeze(0).to(self.device)
        fg_mask1 = fg_mask1.to(self.device)
        fg_mask2 = fg_mask2.to(self.device)


        # calculate similarity between image1 and image2 descriptors
        similarities = chunk_cosine_sim(descriptors1, descriptors2, fg_mask1, fg_mask2)

        # calculate best buddies
        image_idxs = torch.arange(num_patches[0] * num_patches[1], device=self.device)
        sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
        sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
        sim_1, nn_1_fg = sim_1[0, 0], nn_1[0, 0]
        sim_2, nn_2_fg = sim_2[0, 0], nn_2[0, 0]

        reidx_1 = torch.arange(descriptors1.shape[2]).to(self.device)[fg_mask1]
        reidx_2 = torch.arange(descriptors2.shape[2]).to(self.device)[fg_mask2]
        nn_2 = reidx_1[nn_2_fg]
        nn_1 = reidx_2[nn_1_fg]


        # get coordinates to show
        if self.mode == 'full':
            img1_indices_to_show = torch.arange(descriptors1.shape[2], device=fg_mask1.device)[fg_mask1]
            img2_indices_to_show = nn_1
            sim = sim_1

        elif self.mode == 'bidirectional':
            # all 1 to 2
            all1to2_img1_indices_to_show = torch.arange(descriptors1.shape[2], device=fg_mask1.device)[fg_mask1]
            all1to2_img2_indices_to_show = nn_1

            # all 2 to 1
            all2to1_img2_indices_to_show = torch.arange(descriptors2.shape[2], device=fg_mask1.device)[fg_mask2]
            all2to1_img1_indices_to_show = nn_2

            # combine everything
            img1_indices_to_show = torch.cat([all1to2_img1_indices_to_show, all2to1_img1_indices_to_show], dim=0)
            img2_indices_to_show = torch.cat([all1to2_img2_indices_to_show, all2to1_img2_indices_to_show], dim=0)
            sim = torch.cat([sim_1, sim_2], dim=0)

        elif self.mode == 'bb':
            bbs_mask_fg = nn_2_fg[nn_1_fg] == torch.arange(nn_1.shape[0], device=nn_1_fg.device)
            bbs_mask = torch.zeros(image_idxs.shape[0], device=nn_1_fg.device).bool()
            bbs_mask[image_idxs[fg_mask1][bbs_mask_fg]] = True

            img1_indices_to_show = nn_2[nn_1_fg[nn_2_fg] == torch.arange(nn_2.shape[0], device=nn_2_fg.device)]
            img2_indices_to_show = nn_1[bbs_mask_fg]# [bb_indices_to_show]
            sim = None

        points1, points2 = self.convert_patches_to_pixels(img1_indices_to_show, img2_indices_to_show, num_patches)

        return points1, points2, sim


    def convert_patches_to_pixels(self, img1_indices_to_show, img2_indices_to_show, num_patches):

        # coordinates in descriptor map's dimensions
        img1_y_to_show = (img1_indices_to_show / num_patches[1]).cpu().numpy()
        img1_x_to_show = (img1_indices_to_show % num_patches[1]).cpu().numpy()
        img2_y_to_show = (img2_indices_to_show / num_patches[1]).cpu().numpy()
        img2_x_to_show = (img2_indices_to_show % num_patches[1]).cpu().numpy()
        points1, points2 = [], []
        for y1, x1, y2, x2 in zip(img1_y_to_show, img1_x_to_show, img2_y_to_show, img2_x_to_show):
            x1_show = (int(x1) - 1) * self.extractor.stride[1] + self.extractor.stride[1] + self.extractor.p[0] // 2
            y1_show = (int(y1) - 1) * self.extractor.stride[0] + self.extractor.stride[0] + self.extractor.p[0] // 2
            x2_show = (int(x2) - 1) * self.extractor.stride[1] + self.extractor.stride[1] + self.extractor.p[0] // 2
            y2_show = (int(y2) - 1) * self.extractor.stride[0] + self.extractor.stride[0] + self.extractor.p[0] // 2
            points1.append((y1_show, x1_show))
            points2.append((y2_show, x2_show))
        return points1, points2


    def to_pil(self, image: torch.Tensor):
        return Image.fromarray((image*255).astype(np.uint8))
