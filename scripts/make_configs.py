
import os
import torch
from argparse import ArgumentParser
from pathlib import Path

from utils import logging
from utils import print_info


def copy_file(source_path, target_path, name, full_folder, normal_type='square'):
    with open(source_path, "r") as fin:
        with open(target_path, "w") as fout:
            for line in fin:
                if 'REPLACE_NAME' in line:
                    line = line.replace('REPLACE_NAME', name)
                if 'REPLACE_PATH' in line:
                    line = line.replace('REPLACE_PATH', full_folder)
                if 'NORMAL' in line:
                    line = line.replace('NORMAL', normal_type)

                fout.write(line)


parser = ArgumentParser(description='Save evaluation data to file')
parser.add_argument('--folder',  type=str, help='path to experiment folder',  required=True)
parser.add_argument('--verbose', action='store_true', help='verbose', default=False, required=False)
args = parser.parse_args()

logging.LOGGING_INFO = args.verbose


torch.set_grad_enabled(False)

folder_configs = os.path.join(args.folder, 'configs')
source_path    = os.path.join(folder_configs, 'source.json')
target_path    = os.path.join(folder_configs, 'target.json')
map_path       = os.path.join(folder_configs, 'map.json')

manual_source_path = os.path.join(folder_configs, 'source_manual.json')
manual_target_path = os.path.join(folder_configs, 'target_manual.json')
manual_map_path    = os.path.join(folder_configs, 'map_manual.json')

overfit_config_path    = 'resources/overfit.json'
map_config_path        = 'resources/map.json'
manual_map_config_path = 'resources/map_manual.json'

os.makedirs(folder_configs, exist_ok=True)

full_path = os.path.abspath(args.folder)
pair_name = Path(args.folder).name

print_info('Copying over the templates')

# config for nssm
copy_file(overfit_config_path, source_path, 'source', full_path)
copy_file(overfit_config_path, target_path, 'target', full_path)
copy_file(map_config_path, map_path, pair_name, full_path)

# config for nsm
copy_file(overfit_config_path, manual_source_path, 'source_manual', full_path, 'circle')
copy_file(overfit_config_path, manual_target_path, 'target_manual', full_path, 'circle')
copy_file(manual_map_config_path, manual_map_path, pair_name, full_path)
