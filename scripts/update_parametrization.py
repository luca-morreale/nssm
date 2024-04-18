
import torch
from argparse import ArgumentParser
from igl import boundary_loop

from utils import read_mesh
from utils import print_info
from utils import print_error
from utils import logging


parser = ArgumentParser('Update parametrization of given sample')
parser.add_argument('--pth', type=str, required=True)
parser.add_argument('--new', type=str, required=True)
parser.add_argument('--verbose', action='store_true', required=False, default=False)
args = parser.parse_args()

logging.LOGGING_INFO = args.verbose

print_info('Reading sample')
data = torch.load(args.pth)
Vo  = data['points'].double().numpy() * data['C']
Fo  = data['faces'].long().numpy()
TCo = data['param'].double().numpy()

print_info('Reading mesh')
Vn, Fn, TCn, _ = read_mesh(args.new)

bnd_o = boundary_loop(Fo)
bnd_n = boundary_loop(Fn)

print_info('Checking order of elements has not changed')

error_vertices = ((Vo-Vn)**2).sum(-1).mean()
error_faces    = ((Fo-Fn)**2).sum(-1).mean()
error_bnd      = ((TCo[bnd_o] - TCn[bnd_n])**2).sum(-1).mean()


if error_vertices < 1.0e-6:
    print_info('Vertices OK')
else:
    print(error_vertices)
    print_error('Vertices order is different')
    exit(1)

if error_faces < 1.0e-6:
    print_info('Faces OK')
else:
    print_error('Faces order is different')

if error_bnd < 1.0e-6:
    print_info('Boundary OK')
else:
    print_error('Boundary order is different')

print_info('Updating parametrization')

data['param'] = torch.from_numpy(TCn).float()
torch.save(data, args.pth)

print_info('Done')
