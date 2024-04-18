
import os

def automap_paths(folder):
    paths = {}
    paths['source_mesh']   = automap_source_mesh(folder)
    paths['source_sample'] = automap_source_sample(folder)
    paths['target_mesh']   = automap_target_mesh(folder)
    paths['target_sample'] = automap_target_sample(folder)
    paths['matches']       = automap_dino_matches(folder)
    paths['visualization'] = automap_visualization(folder)
    paths['evaluation']    = automap_evaluation(folder)
    paths['gt_source']     = automap_evaluation_source_GT(folder)
    paths['gt_target']     = automap_evaluation_target_GT(folder)
    paths['cut']           = automap_cut(folder)

    paths['ours']          = automap_ours(folder)
    paths['param']         = automap_param(folder)

    paths['alignment']     = automap_alignment(folder)
    paths['overfit_source'] = automap_overfit_source(folder)
    paths['overfit_target'] = automap_overfit_target(folder)

    return paths

def automap_source_mesh(folder):
    return os.path.join(folder, "meshes/source.obj")

def automap_target_mesh(folder):
    return os.path.join(folder, "meshes/target.obj")

def automap_source_sample(folder):
    return os.path.join(folder, "samples/source.pth")

def automap_target_sample(folder):
    return os.path.join(folder, "samples/target.pth")

def automap_alignment(folder):
    return os.path.join(folder, "alignment/rotations.txt")

def automap_dino_matches(folder):
    return os.path.join(folder, "matches/matches.pth")

def automap_ours(folder):
    return os.path.join(folder, "map/map")

def automap_param(folder):
    return os.path.join(folder, "parametrization/map")

def automap_overfit_source(folder):
    return os.path.join(folder, "overfit/overfit_source")

def automap_overfit_target(folder):
    return os.path.join(folder, "overfit/overfit_target")

def automap_visualization(folder):
    return os.path.join(folder, "visualization")

def automap_evaluation(folder):
    return os.path.join(folder, "evaluation")

def automap_evaluation_source_GT(folder):
    return os.path.join(folder, "correspondances/source.txt")

def automap_evaluation_target_GT(folder):
    return os.path.join(folder, "correspondances/target.txt")

def automap_cut(folder):
    return os.path.join(folder, "cut")
