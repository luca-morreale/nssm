
import numpy as np
import pymeshlab
import torch
import trimesh
import igl

from .io_mesh import mesh_to_trimesh_object
from .io_mesh import write_mesh
from .logging import print_info
from .tensor_move import tensor_to_numpy


def get_mesh_edges(V, F):
    mesh = mesh_to_trimesh_object(V, F)
    boundary       = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1) # edges which appears only once
    vertices_index = mesh.edges[boundary]
    return vertices_index


def normalize(V, F):

    ms   = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(V, F)
    ms.add_mesh(mesh)

    ms.compute_matrix_from_scaling_or_normalization(unitflag=True)
    ms.compute_matrix_from_translation(traslmethod='Center on Scene BBox')
    # ms.transform_rotate(angle=-90.0)
    # ms.transform_rotate(angle=90.0, rotaxis='Z axis')

    mesh = ms.current_mesh()
    V_small  = mesh.vertex_matrix()
    F_small  = mesh.face_matrix()
    N_small  = mesh.vertex_normal_matrix()

    return V_small, F_small, N_small


def compute_normals(V, F):
    ms   = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(V, F)
    ms.add_mesh(mesh)
    ms.compute_normal_per_vertex()
    ms.compute_normal_per_face()

    mesh    = ms.current_mesh()
    norm_v = mesh.vertex_normal_matrix()
    norm_f = mesh.face_normal_matrix()

    return norm_v, norm_f

def compute_vertices_normal(V, F):
    return compute_normals(V, F)[0]


def compute_faces_normal(V, F):
    return compute_normals(V, F)[1]


def compute_curvature(V, F, type='Gaussian Curvature'):

    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(V, F)
    ms.add_mesh(mesh)

    ms.compute_scalar_by_discrete_curvature_per_vertex(curvcolormethod=type)
    mesh = ms.current_mesh()
    #curvature = mesh.vertex_quality_array()
    curvature = mesh.vertex_scalar_array()

    return curvature

def compute_curvature_directions(V, F):
    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(V, F)
    ms.add_mesh(mesh)

    ms.compute_curvature_principal_directions_per_vertex()
    mesh = ms.current_mesh()
    e1   = mesh.vertex_curvature_principal_dir1_matrix()
    e2   = mesh.vertex_curvature_principal_dir2_matrix()

    return e1, e2


def filter_out_unused_vertices(vertices, faces, uvs, colors, scalars):

    selection          = torch.zeros(vertices.size(0))
    selection[faces.reshape(-1).unique()] = 1
    selection          = selection.bool()
    indices            = torch.zeros(vertices.size(0)).long()
    indices[selection] = torch.arange(selection.sum()).long()

    new_vertices = vertices[selection]
    new_faces    = indices[faces.cpu()]

    new_uvs     = uvs[selection] if uvs is not None else None
    new_colors  = colors[selection] if colors is not None else None
    new_scalars = None

    if scalars is not None:
        new_scalars = {}
        for k, v in scalars.items():
            new_scalars[k] = v[selection]

    return new_vertices, new_faces, new_uvs, new_colors, new_scalars, indices

## remove unreferenced vertices and compute normals
def clean_mesh(V, F):

    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(V, F)
    mesh.add_vertex_custom_scalar_attribute(np.arange(V.shape[0]), 'idx')
    ms.add_mesh(mesh)
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_re_orient_faces_coherentely()
    ms.compute_normal_per_vertex()

    mesh = ms.current_mesh()

    V_small  = mesh.vertex_matrix()
    F_small  = mesh.face_matrix()
    N_small  = mesh.vertex_normal_matrix()
    NF_small = mesh.face_normal_matrix()
    V_idx = mesh.vertex_custom_scalar_attribute_array('idx').astype(np.int64)

    return V_small, F_small, N_small, V_idx, NF_small


### upsample the mesh
def upsample_mesh(V, F, uv, threshold=0.2):

    write_mesh('/tmp/file.obj', V, F, uv, None)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh('/tmp/file.obj')

    p = pymeshlab.PercentageValue(threshold)
    ms.meshing_surface_subdivision_midpoint(threshold=p)
    ms.compute_normal_per_vertex()
    ms.compute_normal_per_face()

    mesh = ms.current_mesh()

    V_large  = mesh.vertex_matrix()
    F_large  = mesh.face_matrix()
    try:
        UV_large = mesh.vertex_tex_coord_matrix()
    except:
        UV_large = None

    return V_large, F_large, UV_large


## compute genus and area size
def compute_mesh_geo_measures(V, F, target_area=np.pi):

    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(V, F)
    ms.add_mesh(mesh)

    out_dict = ms.get_geometric_measures()
    A = out_dict['surface_area']
    print_info(f'Initial surface area {A}')
    C = np.sqrt( target_area / A )

    out_dict = ms.get_topological_measures()

    print_info('Double checking C')
    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(V*C, F)
    ms.add_mesh(mesh)
    double_check = ms.get_geometric_measures()
    A2 = double_check['surface_area']
    assert np.abs(A2 - target_area) < 1.0e-4, f'Error! Normalized mesh area is not correct - {A2} vs {target_area}'
    print_info('All good, C is correct')

    return C, out_dict['number_holes']-1


def remove_mesh_dangling_faces(faces, points_selected):
    # remove points\faces that are not fully selected, onlu faces that have 3 vertices selected

    # identify which faces are completely selected (all 3 vertex)
    face_mask  = points_selected[faces]
    face_mask  = face_mask.sum(axis=-1)
    keep_faces = face_mask > 2.0

    # identify which points are selected by more than 1 face
    mask = np.zeros(points_selected.shape[0])
    for face_idx in np.nonzero(keep_faces)[0]:
        mask[faces[face_idx]] += 1.0

    mask_binary = mask >= 2.0 # remove points if they are not selected by at least 2 faces

    ### flap ears -> remove vertices that belong to only one face
    ### repeat (recursive)
    return mask_binary, (mask_binary != points_selected).sum() > 0


def faces_to_vertices(points, faces, scalar, to_torch=False):
    points = tensor_to_numpy(points)
    faces  = tensor_to_numpy(faces)
    scalar = tensor_to_numpy(scalar)

    # convert to numpy
    mesh = trimesh.Trimesh(points, faces, process=False)
    vertex = mesh.faces_sparse.dot(scalar.astype(np.float64))
    vertex_val = (vertex / mesh.vertex_degree.reshape(-1)).astype(np.float64)

    if to_torch:
        vertex_val = torch.from_numpy(vertex_val).float()

    return vertex_val


def close_holes(v, f):
    holesize = 10
    _, genus = compute_mesh_geo_measures(v, f, target_area=1.0)

    while genus != 0 and holesize < 500:
        ## close holes
        ms = pymeshlab.MeshSet()
        mesh = pymeshlab.Mesh(v, f)
        ms.add_mesh(mesh)
        ms.meshing_re_orient_faces_coherentely()

        ms.meshing_close_holes(maxholesize=holesize, newfaceselected=False, selfintersection=False)

        ## get new faces and vertices
        mesh = ms.current_mesh()
        v = mesh.vertex_matrix()
        f = mesh.face_matrix()

        ## check genus
        v = np.array(v.tolist())
        _, genus = compute_mesh_geo_measures(v, f, target_area=1.0)

        holesize += 10

    if genus > 0:
        print('FAILED')
        return None

    return v, f


def simplify_mesh(v, f, target_num_faces=None, preserve_boundary=True):
    if target_num_faces is None:
        target_num_faces = int(f.shape[0] / 2)

    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(v, f)
    ms.add_mesh(mesh)
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_num_faces,
                            preservetopology=True, preserveboundary=preserve_boundary)

    mesh = ms.current_mesh()
    v = mesh.vertex_matrix()
    f = mesh.face_matrix()
    v = np.array(v.tolist())
    f = np.array(f.tolist())

    return v, f

def extract_boundary_mask(v, f):

    bnd  = igl.boundary_loop(f)
    mask = np.zeros(v.shape[0])
    mask[bnd] = 1
    return mask.astype(bool)

def face_to_vertex_scalar(vertices, faces, per_face_scalar):
    ### Interpolate values from faces to vertices

    F = tensor_to_numpy(faces)
    V = tensor_to_numpy(vertices)
    D = tensor_to_numpy(per_face_scalar)

    mesh            = trimesh.Trimesh(vertices=V, faces=F, process=False)
    indices         = mesh.vertex_faces
    mask            = indices >= 0
    per_vert_scalar = (D[indices] * mask).sum(axis=-1) / mask.sum(axis=-1)
    return torch.from_numpy(per_vert_scalar).float()


def remove_small_components(v, f, v_idx):
    ## remove small components of the mesh (leave 1 connected component at the end)
    ms   = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(v, f)

    mesh.add_vertex_custom_scalar_attribute(v_idx, 'idx')
    ms.add_mesh(mesh)
    ms.compute_selection_by_small_disconnected_components_per_face()
    ms.meshing_remove_selected_faces()
    ms.meshing_remove_unreferenced_vertices()
    ms.compute_selection_by_non_manifold_per_vertex()
    ms.meshing_remove_selected_vertices()
    ms.meshing_re_orient_faces_coherentely()

    # extend selection to non-manifold vertices, this makes easier later stages (parametrization)
    mesh = ms.current_mesh()

    v   = mesh.vertex_matrix()
    f   = mesh.face_matrix()
    idx = mesh.vertex_custom_scalar_attribute_array('idx').astype(np.int64)
    v   = np.array(v.tolist())
    f   = np.array(f.tolist())
    idx = np.array(idx.tolist())

    return v, f, idx


def is_point_in_triangle(pt, v1, v2, v3, return_mask=False):

    pt = pt.reshape(1, -1, pt.shape[-1]).double()
    v1 = v1.reshape(-1, 1, pt.shape[-1]).double()
    v2 = v2.reshape(-1, 1, pt.shape[-1]).double()
    v3 = v3.reshape(-1, 1, pt.shape[-1]).double()

    d1 = halfplane_sign(pt, v1, v2)
    d2 = halfplane_sign(pt, v2, v3)
    d3 = halfplane_sign(pt, v3, v1)

    # has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    # has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)

    # return ~(has_neg * has_pos)
    mask = (d1 >= 0) & (d2 >= 0) & (d3 >= 0)
    mask &= ((d1 > 0) | (d2 > 0) | (d3 > 0))
    if return_mask:
        return mask

    idx  = mask.transpose(0,1).nonzero()[:, 1]
    return idx

def halfplane_sign(p1, p2, p3):
    return (p1[..., 0] - p3[..., 0]) * (p2[..., 1] - p3[..., 1]) - (p2[..., 0] - p3[..., 0]) * (p1[..., 1] - p3[..., 1])

