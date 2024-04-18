
import os
import mitsuba as mi
import numpy as np
import torch
import drjit as dr
from PIL import Image
from PIL import ImageEnhance
from matplotlib import pyplot as plt

from utils import read_mesh

from .images import convert_img
from .images import draw_correspondences
from .scene_definition import load_sensor
from .scene_definition import load_scene


class Mitsuba:

    def __init__(self, shape_path1, shape_path2, num_rotations, radius=15, two_axis=True, resolution=224):
        self.radius = radius
        self.phis   = np.linspace(0, 360.0, num_rotations, endpoint=False)
        self.thetas = [0.0]
        if two_axis:
            self.thetas = np.linspace(-90, 90, int(num_rotations / 2), endpoint=False)

        self.shape_path1 = shape_path1
        self.shape_path2 = shape_path2

        v1, f1, _, _ = read_mesh(shape_path1)
        v2, f2, _, _ = read_mesh(shape_path2)

        self.v1 = torch.from_numpy(v1)
        self.v2 = torch.from_numpy(v2)
        self.f1 = torch.from_numpy(f1).long()
        self.f2 = torch.from_numpy(f2).long()

        self.sensor = load_sensor(self.radius, 0, 0, resolution=resolution)
        self.scene1 = load_scene(self.shape_path1, 0, 0)
        self.scene2 = load_scene(self.shape_path2, 0, 0)

        params1 = mi.traverse(self.scene1)
        params2 = mi.traverse(self.scene2)
        self.initial_v1 = dr.unravel(mi.Point3f, params1['shape.vertex_positions'])
        self.initial_v2 = dr.unravel(mi.Point3f, params2['shape.vertex_positions'])


    def to_pil(self, image: torch.Tensor):
        im = Image.fromarray((image*255).astype(np.uint8))
        converter = ImageEnhance.Color(im)
        im = converter.enhance(2.0)
        return im

    def update_scene(self, scene, initial_v, phi, theta):
        params = mi.traverse(scene)
        transf = mi.Transform4f.rotate([0, 0, 1], theta).rotate([0, 1, 0], phi)
        params['shape.vertex_positions'] = dr.ravel(transf @ initial_v)
        params.update()
        return params


    def render_scene(self, scene, initial_v, phi, theta):
        params = self.update_scene(scene, initial_v, phi, theta)
        image  = mi.render(scene, params, spp=200, sensor=self.sensor)
        return image


    def render_all_images(self, R1, R2, outfolder='./', for_figure=-1):

        self.R1_phi   = R1
        self.R1_theta = 0
        if len(R1) > 1:
            self.R1_phi   = R1[0]
            self.R1_theta = R1[1]
        self.R2_phi   = R2
        self.R2_theta = 0
        if len(R2) > 1:
            self.R2_phi   = R2[0]
            self.R2_theta = R2[1]

        images1 = []
        images2 = []
        angles  = []

        idx = 0

        for phi in self.phis:
            for theta in self.thetas:
                if for_figure != idx and for_figure >= 0:
                    idx +=1
                    if idx > for_figure:
                        return
                    continue

                image_1 = self.render_scene(self.scene1, self.initial_v1, phi + self.R1_phi, theta + self.R1_theta)
                image_2 = self.render_scene(self.scene2, self.initial_v2, phi + self.R2_phi, theta + self.R2_theta)

                angles.append([phi, theta])

                images1.append(convert_img(image_1, return_tensor=True))
                images2.append(convert_img(image_2, return_tensor=True))


                if for_figure == idx and for_figure >= 0:

                    im1 = self.to_pil(images1[-1])
                    im2 = self.to_pil(images2[-1])
                    fig1 = draw_correspondences([], [], im1, im2)
                    # if args.figure is not None:
                    debug_outfile = os.path.join(outfolder, f'corresp_{idx:03d}') + '_renderings.pdf'
                    fig1.savefig(debug_outfile, bbox_inches='tight', pad_inches=0, dpi=160) #if args.figure is not None else 60)
                    debug_outfile = os.path.join(outfolder, f'corresp_{idx:03d}') + '_renderings.png'
                    fig1.savefig(debug_outfile, bbox_inches='tight', pad_inches=0, dpi=160) #if args.figure is not None else 60)
                    plt.close('all')

                idx += 1

        return angles, images1, images2


    def lift_matches(self, dino_helper, angles, img_size, descriptors_1, descriptors_2, fg_masks_1, fg_masks_2, num_patches):

        src_3D   = []
        tgt_3D   = []
        src_fcs  = []
        tgt_fcs  = []
        src_bary = []
        tgt_bary = []
        similarity = []

        for idx in range(descriptors_1.shape[0]):

            phi, theta = angles[idx]

            points1, points2, sim = dino_helper.features_to_matches(descriptors_1[idx], descriptors_2[idx], fg_masks_1[idx], fg_masks_2[idx], num_patches)

            points1 = [(pts[1], pts[0]) for pts in points1]
            points2 = [(pts[1], pts[0]) for pts in points2]


            self.update_scene(self.scene1, self.initial_v1, phi + self.R1_phi, theta + self.R1_theta)
            self.update_scene(self.scene2, self.initial_v2, phi + self.R2_phi, theta + self.R2_theta)

            pts_3D_1, fcs_idx_1, bary_1, mask_1 = self.extract_3D_matches(self.sensor, self.scene1, points1, img_size, (self.v1, self.f1, phi + self.R1_phi, theta + self.R1_theta))
            pts_3D_2, fcs_idx_2, bary_2, mask_2 = self.extract_3D_matches(self.sensor, self.scene2, points2, img_size, (self.v2, self.f2, phi + self.R2_phi, theta + self.R2_theta))

            mask = mask_1 * mask_2
            src_3D.append(pts_3D_1[mask])
            tgt_3D.append(pts_3D_2[mask])
            src_fcs.append(fcs_idx_1[mask])
            tgt_fcs.append(fcs_idx_2[mask])
            src_bary.append(bary_1.squeeze()[mask])
            tgt_bary.append(bary_2.squeeze()[mask])
            similarity.append(sim[mask])

        return src_3D, tgt_3D, src_fcs, tgt_fcs, src_bary, tgt_bary, similarity


    def extract_3D_matches(self, sensor, scene, points, image_res, mesh_data):

        sis = self.extract_intersection(sensor, scene, points, image_res)

        # intersection_3D = sis.p.torch()
        valid       = sis.is_valid()
        valid       = torch.tensor(valid).bool()
        triangle_id = torch.tensor(sis.prim_index).long()

        v, f, phi, theta = mesh_data
        intersection_3D = (mi.Transform4f.rotate([0, 0, 1], theta).rotate([0, 1, 0], phi)).inverse() @ sis.p
        intersection_3D = intersection_3D.torch()

        face_valid = f[triangle_id[valid]]
        tris       = v[face_valid]
        valid_intersection = intersection_3D[valid]
        baryc_valid = Mitsuba.compute_barycentric(valid_intersection, tris[:, 0], tris[:, 1], tris[:, 2])
        baryc = torch.zeros_like(intersection_3D)
        baryc[valid] = baryc_valid.float()

        errors = baryc < 0
        if errors.sum() > 0:
            math_errors = errors & (baryc > -1.e-4)
            baryc[math_errors] = 0.0 # fix math errors

            # fix barycentric to sum to 1
            row_to_rebalance = math_errors.sum(-1).bool()
            baryc[row_to_rebalance] /= baryc[row_to_rebalance].reshape(-1, 3).sum(dim=1, keepdim=True)

        if (baryc < 0).sum() > 0:
            print_error('Some barycentric coordinates are negative!')
            exit(1)

        return intersection_3D, triangle_id, baryc, valid


    def extract_intersection_angle(self, sensor, scene, points, image_res):
        sis         = self.extract_intersection(sensor, scene, points, image_res)
        direction   = sis.wi.torch()
        triangle_id = torch.tensor(sis.prim_index).long()
        valid       = sis.is_valid()
        valid       = torch.tensor(valid).bool()

        return torch.acos(direction[..., -1]), triangle_id, valid


    def extract_intersection(self, sensor, scene, points, image_res):

        mi_points = mi.Point2f(np.array(points).reshape(-1, 2).astype('float') / image_res)
        rays, _   = sensor.sample_ray(0, 0, mi_points, 0)
        sis       = scene.ray_intersect(rays)
        return sis


    @staticmethod
    def compute_barycentric(pt, vert_a, vert_b, vert_c):
        v0 = vert_b - vert_a
        v1 = vert_c - vert_a
        v2 = pt - vert_a

        d00 = (v0 * v0).sum(-1)
        d01 = (v0 * v1).sum(-1)
        d11 = (v1 * v1).sum(-1)
        d20 = (v2 * v0).sum(-1)
        d21 = (v2 * v1).sum(-1)

        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        return torch.stack([u, v, w], dim=1)

    def aggregate_descriptors_to_shape(self, dino_helper, angles, img_size, descriptors_1, descriptors_2, fg_masks_1, fg_masks_2, num_patches):

        # list all pixels (1 for each patch)
        img1_indices_to_show = torch.arange(descriptors_1.shape[-2])
        img2_indices_to_show = torch.arange(descriptors_1.shape[-2])

        points1, points2 = dino_helper.convert_patches_to_pixels(img1_indices_to_show, img2_indices_to_show, num_patches)
        points1 = [ (pts[1], pts[0]) for pts in points1]
        points2 = [ (pts[1], pts[0]) for pts in points2]

        # empty descriptors
        F1_descs = torch.zeros(self.f1.shape[0], descriptors_1.shape[-1])
        F2_descs = torch.zeros(self.f2.shape[0], descriptors_1.shape[-1])
        F1_count = torch.zeros(self.f1.shape[0])
        F2_count = torch.zeros(self.f2.shape[0])


        for idx in range(descriptors_1.shape[0]):

            phi, theta = angles[idx]

            self.update_scene(self.scene1, self.initial_v1, phi + self.R1_phi, theta + self.R1_theta)
            self.update_scene(self.scene2, self.initial_v2, phi + self.R2_phi, theta + self.R2_theta)

            angles1, triangle_id1, valid1 = self.extract_intersection_angle(self.sensor, self.scene1, points1, img_size)
            angles2, triangle_id2, valid2 = self.extract_intersection_angle(self.sensor, self.scene2, points2, img_size)

            mask_1 = (angles1 * 180 / np.pi).abs() < 30.0
            mask_2 = (angles2 * 180 / np.pi).abs() < 30.0
            # mask_1 = valid1
            # mask_2 = valid2

            mask_1 &= valid1
            mask_2 &= valid2

            F1_descs[triangle_id1[mask_1]] += descriptors_1[idx].squeeze()[mask_1]
            F2_descs[triangle_id2[mask_2]] += descriptors_2[idx].squeeze()[mask_2]
            F1_count[triangle_id1[mask_1]] += 1
            F2_count[triangle_id2[mask_2]] += 1

        mask = F1_count > 0
        F1_descs[mask] /= F1_count[mask].unsqueeze(-1)
        mask = F2_count > 0
        F2_descs[mask] /= F2_count[mask].unsqueeze(-1)

        return F1_descs, F2_descs, F1_count, F2_count
