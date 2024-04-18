
import numpy as np
import trimesh

from .colors import sinebow
from .read_OBJ import readOBJ
from .tensor_move import tensor_to_numpy
from .write_OBJ import writeOBJ


def write_point_cloud(filename, V, scalars=None):
    ext = filename[filename.rfind('.')+1:]

    V = tensor_to_numpy(V)
    mesh = trimesh.Trimesh(V)
    if scalars is not None:
        for k, v in scalars.items():
            mesh.vertex_attributes[k] = tensor_to_numpy(v)

    mesh.export(filename, include_attributes=True)


def write_mesh(filename, V, F, UV, N, scalars=None, colors=None):
    ext = filename[filename.rfind('.')+1:]

    if ext == 'obj':
        writeOBJ(filename, V, F, UV, N)
    elif ext == 'ply':

        mesh = mesh_to_trimesh_object(V, F, colors)
        if scalars is not None:
            for k, v in scalars.items():
                mesh.vertex_attributes[k] = tensor_to_numpy(v)

        if UV is not None:
            UV = tensor_to_numpy(UV)
            mesh.vertex_attributes['texture_u'] = UV[:, 0]
            mesh.vertex_attributes['texture_v'] = UV[:, 1]
            mesh.vertex_attributes['s'] = UV[:, 0] # for blender visualization
            mesh.vertex_attributes['t'] = UV[:, 1] # for blender visualization

        # not sure how much this will affect later computations
        mesh.remove_unreferenced_vertices()

        mesh.export(filename, include_attributes=True)

    elif ext == 'off':
        mesh = mesh_to_trimesh_object(V, F)
        mesh.export(filename)


def read_mesh(filename):

    ext = filename[filename.rfind('.')+1:]

    if ext == 'obj':
        V, F, UV, _, N = readOBJ(filename)
    elif ext == 'ply':
        mesh = trimesh.load(filename, process=False)
        V = mesh.vertices
        F = mesh.faces
        UV = None
        if mesh.visual.uv is not None:
            UV = mesh.visual.uv
        N = mesh.vertex_normals
    else:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(filename)
        mesh = ms.current_mesh()

        V = mesh.vertex_matrix()
        F = mesh.face_matrix()
        N = mesh.vertex_normal_matrix()
        UV = None

    return V,F,UV,N


def save_mesh_with_correspondences(filename, points, faces, correspondences, vertices_colors=None):
    L = len(correspondences)
    correspondences_colors = [ np.concatenate((sinebow(i/L), [1]))*255 for i in range(L) ]

    mesh = mesh_to_trimesh_object(points, faces, vertices_colors)

    for i, point in enumerate(correspondences):
        sphere = generate_sphere(point, radius=0.015)
        mesh  += sphere
        mesh.visual.vertex_colors[-sphere.visual.vertex_colors.shape[0]:] = correspondences_colors[i]

    mesh.export(filename)

def save_correspondences(filename, v1, f1, v2, f2, L1, L2, axis=2, for_figure=False):
    d = 0.3

    v1 = np.copy(tensor_to_numpy(v1))
    v2 = np.copy(tensor_to_numpy(v2))
    L1 = np.copy(tensor_to_numpy(L1))
    L2 = np.copy(tensor_to_numpy(L2))

    # set min y to 0 for both
    disp_1_y = v1[..., 1].min()
    disp_2_y = v2[..., 1].min()

    v1[..., 1] -= disp_1_y
    v2[..., 1] -= disp_2_y

    # displace v2 to not overlap
    disp_2 = -v2[..., axis].min() + v1[..., axis].max() + d
    v2[..., axis] += disp_2

    mesh  = mesh_to_trimesh_object(v1, f1)
    mesh += mesh_to_trimesh_object(v2, f2)

    if for_figure:
        meshes_filename = filename[:-4] + '_meshes.ply'
        mesh.export(meshes_filename)

    L1[..., 1]    -= disp_1_y
    L2[..., 1]    -= disp_2_y
    L2[..., axis] += disp_2

    L = L1.shape[0]
    correspondences_colors = [ np.concatenate((sinebow(i/L), [1]))*255 for i in range(L) ]

    # add sphere around points
    corresp_mesh = None
    for i, point in enumerate(L1):
        sphere = generate_sphere(point, radius=0.005)
        if corresp_mesh is None:
            corresp_mesh  = sphere
        else:
            corresp_mesh += sphere
        corresp_mesh.visual.vertex_colors[-sphere.visual.vertex_colors.shape[0]:] = correspondences_colors[i]

        sphere        = generate_sphere(L2[i], radius=0.005)
        corresp_mesh += sphere
        corresp_mesh.visual.vertex_colors[-sphere.visual.vertex_colors.shape[0]:] = correspondences_colors[i]


    # add lines connecting the points
    for i in range(L1.shape[0]):
        cylinder = trimesh.creation.cylinder(0.001, segment=np.vstack([L2[i], L1[i]]))
        corresp_mesh += cylinder
        corresp_mesh.visual.vertex_colors[-cylinder.visual.vertex_colors.shape[0]:] = correspondences_colors[i]

    if for_figure:
        meshes_filename = filename[:-4] + '_correspondences.ply'
        corresp_mesh.export(meshes_filename)

    mesh += corresp_mesh
    mesh.export(filename)


def mesh_to_trimesh_object(points, faces, vertices_colors=None):
    vertices = tensor_to_numpy(points)
    faces    = tensor_to_numpy(faces)
    if vertices_colors is None:
        vertices_colors = np.tile(np.array([0.82]*3), [points.shape[0], 1])
    else:
        vertices_colors = tensor_to_numpy(vertices_colors)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertices_colors, process=False)
    return mesh


def generate_sphere(center, radius=0.0025, color=None):
    center = tensor_to_numpy(center)
    if color is None:
        color = trimesh.visual.random_color()

    sphere = trimesh.primitives.Sphere(center=center, radius=radius)
    sphere.visual.vertex_colors = color

    return sphere
