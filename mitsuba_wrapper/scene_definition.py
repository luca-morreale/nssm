
import os
import mitsuba as mi
from mitsuba import ScalarTransform4f as T


def load_sensor(r, phi, h, resolution=224, fov=4.5):
    # Apply two rotations to convert from spherical coordinates to world 3D coordinates.
    origin = (T.rotate([0, 1, 0], phi).translate([0, h, 0])) @ mi.ScalarPoint3f([0, 0, r])

    return mi.load_dict({
        'type': 'perspective',
        'fov': fov,
        'to_world': T.look_at(
            origin=origin,
            target=[0, h, 0],
            up=[0, 1, 0]
        ),
        'sampler': {
            'type': 'independent',
            'sample_count': 100
        },
        "focus_distance":1000,
        'film': {
            'type': 'hdrfilm',
            'width':  resolution,
            'height': resolution,
            'rfilter': {
                'type': 'tent',
            },
            'pixel_format': 'rgba',
        },
    })

def load_scene(shape_path, phi, theta):
    scene_dict = {
        'type': 'scene',
        # The keys below correspond to object IDs and can be chosen arbitrarily
        'integrator': {
            'type': 'direct',
            'hide_emitters':True,
        },
        'ligh_1':{
            'type': 'point',
            'to_world': T.look_at(
                origin=[-0.5, 3, 0.5],
                target=[0, 0.0, 0],
                up=[0, 1, 0]
            ),
            'intensity': {
                'type': 'rgb',
                'value': 4.0,
            },
        },
        'light_yellow': {
            'type': 'point',
            'to_world': T.look_at(
                origin=[0.5, -1, 0.5],
                target=[0, 0.0, 0],
                up=[0, 1, 0]
            ),
            'intensity': {
                'type': 'rgb',
                'value': [2.5, 2.5, 0.0],
                # 'value': 5,
            },
        },
        'light_red': {
            'type': 'point',
            'to_world': T.look_at(
                origin=[1, 1, 1],
                target=[0, 0.0, 0],
                up=[0, 1, 0]
            ),
            'intensity': {
                'type': 'rgb',
                'value': [5.0, 0.0, 0.0],
                # 'value': 5.0,
            },
        },
        'light_blue': {
            'type': 'point',
            'to_world': T.look_at(
                origin=[-1, 1, 1],
                target=[0, 0.0, 0],
                up=[0, 1, 0]
            ),
            'intensity': {
                'type': 'rgb',
                'value': [0.0, 0.0, 5.0],
                # 'value': 5.0,
            }
        },
        'light_green': {
            'type': 'point',
            'to_world': T.look_at(
                origin=[0, 1.5, 0.0],
                target=[0, 0.0, 0],
                up=[0, 1, 0]
            ),
            'intensity': {
                'type': 'rgb',
                'value': [0.0, 5.0, 0.0],
                # 'value': 5.0,
            }
        },
    }

    if os.path.exists(shape_path[:-3] + 'bmp'):
        scene_dict['shape_texture'] = {
                'type': 'diffuse',
                'reflectance' : {
                    'type': 'bitmap',
                    'filter_type':'bilinear',
                    'filename': shape_path[:-3] + 'bmp',
                    'wrap_mode': 'mirror'
                },
        }
        scene_dict['light_red']['intensity']['value']    = 4.0
        scene_dict['light_blue']['intensity']['value']   = 4.0
        scene_dict['light_green']['intensity']['value']  = 4.0
        scene_dict['light_yellow']['intensity']['value'] = 4.0
    scene_dict['light_red']['intensity']['value']    = 4.0
    scene_dict['light_blue']['intensity']['value']   = 4.0
    scene_dict['light_green']['intensity']['value']  = 4.0
    scene_dict['light_yellow']['intensity']['value'] = 4.0


    # order matter so add shape as last thing!
    scene_dict['shape'] = {
            'type': shape_path[-3:],
            'filename': shape_path,
            'to_world': T.rotate([0, 0, 1], theta).rotate([0, 1, 0], phi),
            'flip_tex_coords':True,
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.5, 0.5, 0.5]},
            },
        }
    if os.path.exists(shape_path[:-3] + 'bmp'):
        scene_dict['shape']['bsdf'] = {
                'type':'ref',
                'id':'shape_texture'
        }

    return mi.load_dict(scene_dict)

def load_alignment_scene(shape_path, phi):
    scene_dict = {
        'type': 'scene',
        # The keys below correspond to object IDs and can be chosen arbitrarily
        'integrator': {
            'type': 'direct',
            'hide_emitters':True,
        },
        'shape': {
            'type': shape_path[-3:],
            'filename': shape_path,
            'to_world': T.rotate([0, 1, 0], phi),
            'flip_tex_coords':False,
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.5, 0.5, 0.5]},
            },
        },
        'ligh_1':{
            'type': 'point',
            'to_world': T.look_at(
                origin=[-0.5, 3, 0.5],
                target=[0, 1.5, 0],
                up=[0, 1, 0]
            ),
            'intensity': {
                'type': 'rgb',
                'value': 1.0,
            },
        },
        'light_2': {
            'type': 'point',
            'to_world': T.look_at(
                origin=[0.5, -1, 0.5],
                target=[0, 1.5, 0],
                up=[0, 1, 0]
            ),
            'intensity': {
                'type': 'rgb',
                'value': 1.0,
            },
        },
        'light_red': {
            'type': 'point',
            'to_world': T.look_at(
                origin=[1, 1, 1],
                target=[0, 1.5, 0],
                up=[0, 1, 0]
            ),
            'intensity': {
                'type': 'rgb',
                'value': [5.0, 0.0, 0.0],
            },
        },
        'light_blue': {
            'type': 'point',
            'to_world': T.look_at(
                origin=[-1, 1, 1],
                target=[0, 1.5, 0],
                up=[0, 1, 0]
            ),
            'intensity': {
                'type': 'rgb',
                'value': [0.0, 0.0, 5.0],
            }
        },
        'light_green': {
            'type': 'point',
            'to_world': T.look_at(
                origin=[0, 1.5, 0.0],
                target=[0, 1.5, 0],
                up=[0, 1, 0]
            ),
            'intensity': {
                'type': 'rgb',
                'value': [0.0, 5.0, 0.0],
            }
        },
    }
    return mi.load_dict(scene_dict)
