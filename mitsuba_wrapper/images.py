
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import List, Tuple


def convert_img(img, return_tensor=False):
    img_1 = img[..., :3].torch() ** (1.0 / 2.2)
    img_2 = img_1 * 255 / img_1.max()
    if return_tensor:
        return img_2 / 255
    img_3 = img_2.numpy().astype(np.uint8)
    return img_3


def draw_correspondences(points1: List[Tuple[float, float]], points2: List[Tuple[float, float]],
                         image1: Image.Image, image2: Image.Image, show_originals=True) -> Tuple[plt.Figure, plt.Figure]:
    """
    draw point correspondences on images.
    :param points1: a list of (y, x) coordinates of image1, corresponding to points2.
    :param points2: a list of (y, x) coordinates of image2, corresponding to points1.
    :param image1: a PIL image.
    :param image2: a PIL image.
    :return: two figures of images with marked points.
    """
    assert len(points1) == len(points2), f"points lengths are incompatible: {len(points1)} != {len(points2)}."
    num_points = len(points1)

    image1_size = image1.size
    if len(points1) > 0 and show_originals:
        new_image = Image.new('RGB',(2*image1_size[0], 2*image1_size[1]), (250,250,250))
    else:
        new_image = Image.new('RGB',(2*image1_size[0], image1_size[1]), (250,250,250))
    new_image.paste(image1,(0,0))
    new_image.paste(image2,(image1_size[0],0))
    if len(points1) > 0:
        new_image.paste(image1,(0,image1_size[1]))
        new_image.paste(image2,(image1_size[0], image1_size[1]))

    fig1, ax1 = plt.subplots()
    ax1.axis('off')
    ax1.imshow(new_image, aspect='equal')

    if num_points > 15:
        cmap = plt.get_cmap('tab10')
    else:
        cmap = ListedColormap(["red", "yellow", "blue", "lime", "magenta", "indigo", "orange", "cyan", "darkgreen",
                               "maroon", "black", "white", "chocolate", "gray", "blueviolet"])
    colors = np.array([cmap(x) for x in range(num_points)])
    radius1, radius2 = 8, 1
    # radius1, radius2 = 80, 10
    for point1, point2, color in zip(points1, points2, colors):
        y1, x1 = point1

        circ1_1 = plt.Circle((x1, y1), radius1, facecolor=color, edgecolor='white', alpha=0.5)
        circ1_2 = plt.Circle((x1, y1), radius2, facecolor=color, edgecolor='white')
        if show_originals:
            circ1_1 = plt.Circle((x1, y1+image1_size[1]), radius1, facecolor=color, edgecolor='white', alpha=0.5)
            circ1_2 = plt.Circle((x1, y1+image1_size[1]), radius2, facecolor=color, edgecolor='white')
        ax1.add_patch(circ1_1)
        ax1.add_patch(circ1_2)

        y2, x2 = point2
        circ2_1 = plt.Circle((x2+image1_size[0], y2), radius1, facecolor=color, edgecolor='white', alpha=0.5)
        circ2_2 = plt.Circle((x2+image1_size[0], y2), radius2, facecolor=color, edgecolor='white')
        if show_originals:
            circ2_1 = plt.Circle((x2+image1_size[0], y2+image1_size[1]), radius1, facecolor=color, edgecolor='white', alpha=0.5)
            circ2_2 = plt.Circle((x2+image1_size[0], y2+image1_size[1]), radius2, facecolor=color, edgecolor='white')
        ax1.add_patch(circ2_1)
        ax1.add_patch(circ2_2)
    return fig1

