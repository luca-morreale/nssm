
import torch

from PIL import Image
from typing import List, Tuple

from .extractor_fast import ViTExtractor_Fast


def get_model(model_type='dino_vits8', stride=14):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor_Fast(model_type, stride=stride, device=device)

    return extractor, device

def compute_features(extractor, device, images, load_size=224, layer=9, facet='key', bin=False, thresh=0.05):

    if type(images) == list:
        image_batch = []
        for image in images:
            img_tensor = extractor.preprocess(image, load_size)
            image_batch.append(img_tensor)
        image_batch = torch.cat(image_batch, dim=0)
    else:
        image_batch = extractor.preprocess(images)

    image_batch    = image_batch.to(device)

    descriptors    = extractor.extract_descriptors(image_batch, layer, facet, bin)
    num_patches, _ = extractor.num_patches, extractor.load_size

    if type(images) == list:
        saliency_map = [ extractor.extract_saliency_maps(img_b.unsqueeze(0))[0] for img_b in image_batch ]
        saliency_map = torch.cat(saliency_map, dim=0)
    else:
        saliency_map = extractor.extract_saliency_maps(image_batch)

    fg_mask = saliency_map > thresh

    return descriptors.to(device), num_patches, saliency_map, fg_mask.to(device)





def find_correspondences(image_path1: str, image_path2: str, mode='full', load_size: int = 224, layer: int = 9,
                         facet: str = 'key', bin: bool = True, thresh: float = 0.05, model_type: str = 'dino_vits8',
                         stride: int = 4, return_desc=False, extractor=None, device=None) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]],
                                                                              Image.Image, Image.Image, torch.Tensor]:
    """
    finding point correspondences between two images.
    :param image_path1: path to the first image.
    :param image_path2: path to the second image.
    :param load_size: size of the smaller edge of loaded images. If None, does not resize.
    :param layer: layer to extract descriptors from.
    :param facet: facet to extract descriptors from.
    :param bin: if True use a log-binning descriptor.
    :param thresh: threshold of saliency maps to distinguish fg and bg.
    :param model_type: type of model to extract descriptors from.
    :param stride: stride of the model.
    :return: list of points from image_path1, list of corresponding points from image_path2, the processed pil image of
    image_path1, and the processed pil image of image_path2.
    """

    # extracting descriptors for each image
    if extractor is None:
        extractor, device = get_model(model_type, stride)

    descriptors, num_patches, _, fg_masks = compute_features(extractor, device, [image_path1, image_path2], load_size, layer, facet, bin, thresh)
    descriptors1, descriptors2 =  descriptors[0].unsqueeze(0), descriptors[1].unsqueeze(0)
    num_patches1, num_patches2 = num_patches, num_patches
    fg_mask1, fg_mask2         = fg_masks[:fg_masks.shape[0]//2], fg_masks[fg_masks.shape[0]//2:]
    similarities = chunk_cosine_sim(descriptors1, descriptors2, fg_mask1, fg_mask2)

    # calculate best buddies
    image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=device)
    sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
    sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
    sim_1, nn_1_fg = sim_1[0, 0], nn_1[0, 0]
    sim_2, nn_2_fg = sim_2[0, 0], nn_2[0, 0]

    reidx_1 = torch.arange(descriptors1.shape[2]).to(device)[fg_mask1]
    reidx_2 = torch.arange(descriptors2.shape[2]).to(device)[fg_mask2]
    nn_2 = reidx_1[nn_2_fg]
    nn_1 = reidx_2[nn_1_fg]


    # get coordinates to show
    if mode == 'full':
        img1_indices_to_show = torch.arange(descriptors1.shape[2], device=fg_mask1.device)[fg_mask1]
        img2_indices_to_show = nn_1

    elif mode == 'bidirectional':
        # all 1 to 2
        all1to2_img1_indices_to_show = torch.arange(descriptors1.shape[2], device=fg_mask1.device)[fg_mask1]
        all1to2_img2_indices_to_show = nn_1

        # all 2 to 1
        all2to1_img2_indices_to_show = torch.arange(descriptors2.shape[2], device=fg_mask1.device)[fg_mask2]
        all2to1_img1_indices_to_show = nn_2

        # combine everything
        img1_indices_to_show = torch.cat([all1to2_img1_indices_to_show, all2to1_img1_indices_to_show], dim=0)
        img2_indices_to_show = torch.cat([all1to2_img2_indices_to_show, all2to1_img2_indices_to_show], dim=0)

    elif mode == 'bb':
        bbs_mask_fg = nn_2_fg[nn_1_fg] == torch.arange(nn_1.shape[0], device=nn_1_fg.device)
        bbs_mask = torch.zeros(image_idxs.shape[0], device=nn_1_fg.device).bool()
        bbs_mask[image_idxs[fg_mask1][bbs_mask_fg]] = True


        # get coordinates to show
        img1_indices_to_show = nn_2[nn_1_fg[nn_2_fg] == torch.arange(nn_2.shape[0], device=nn_2_fg.device)]
        img2_indices_to_show = nn_1[bbs_mask_fg]# [bb_indices_to_show]

    # coordinates in descriptor map's dimensions
    img1_y_to_show = (img1_indices_to_show / num_patches1[1]).cpu().numpy()
    img1_x_to_show = (img1_indices_to_show % num_patches1[1]).cpu().numpy()
    img2_y_to_show = (img2_indices_to_show / num_patches2[1]).cpu().numpy()
    img2_x_to_show = (img2_indices_to_show % num_patches2[1]).cpu().numpy()
    points1, points2 = [], []
    for y1, x1, y2, x2 in zip(img1_y_to_show, img1_x_to_show, img2_y_to_show, img2_x_to_show):
        x1_show = (int(x1) - 1) * extractor.stride[1] + extractor.stride[1] + extractor.p // 2
        y1_show = (int(y1) - 1) * extractor.stride[0] + extractor.stride[0] + extractor.p // 2
        x2_show = (int(x2) - 1) * extractor.stride[1] + extractor.stride[1] + extractor.p // 2
        y2_show = (int(y2) - 1) * extractor.stride[0] + extractor.stride[0] + extractor.p // 2
        points1.append((y1_show, x1_show))
        points2.append((y2_show, x2_show))

    out_elements = [points1, points2]
    if mode == 'bb':
        out_elements.append(sim_1[bbs_mask_fg])
    elif mode == 'full':
        out_elements.append(sim_1)
    elif mode == 'bidirectional':
        out_elements.append(torch.cat([sim_1, sim_2], dim=0))
    if return_desc:
        out_elements.append(descriptors1)
        out_elements.append(descriptors2)
    return out_elements


def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor, mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y) """

    masked_x = x[:, :, mask1]
    masked_y = y[:, :, mask2]

    result_list = []
    num_token_x = masked_x.shape[2]

    for token_idx in range(num_token_x):
        token = masked_x[:, :, token_idx].unsqueeze(dim=2)  # Bx1x1xd'
        result = torch.nn.CosineSimilarity(dim=3)(token, masked_y)
        result_list.append(result)  # Bx1xt

    loop_similarity = torch.stack(result_list, dim=0)

    return loop_similarity.squeeze().unsqueeze(0).unsqueeze(0)
