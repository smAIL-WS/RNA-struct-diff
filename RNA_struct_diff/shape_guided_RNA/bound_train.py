import torch
import os
import sys
sys.path.append("/home/gback/discrete_diffusion")
import argparse
from diffusion_utils.utils import add_parent_path, set_seeds
from diffusion_utils.utils import is_main_process


# Exp
from experiment import Experiment, add_exp_args

# Data
add_parent_path(level=1)
from datasets.bucket_dataset import get_data, get_data_id, add_data_args

# Model
from model import get_model_id, add_model_args, get_model_class, get_model_bind,get_model_bind_RNAfold

# Optim
from diffusion_utils.optim.multistep import get_optim, get_optim_id, add_optim_args



###########
## Setup ##
###########



parser = argparse.ArgumentParser()
add_exp_args(parser)
add_data_args(parser)
add_model_args(parser)
add_optim_args(parser)

parser.add_argument('--padding', action='store_true', help='Enable padding (default: False)')
parser.add_argument('--ddp', action='store_true', help='Enable DistributedDataParallel training')
#parser.add_argument('--local-rank',dest='local_rank' type=int, default=0, help='Local rank for DDP')
parser.add_argument('--project_contact_map', type=eval, default=True,
                    help='Use a CNN projection for the contact map (default: True). If False, only interpolation is used.')
parser.add_argument('--arch', type=str, default="2d",
                    help='1d/2d architecture')
### edited for early stop
parser.add_argument('--es_patience', type=int, default=None,
                    help='Number of evals with no improvement before stopping; None disables.')
parser.add_argument('--es_min_delta', type=float, default=0.0,
                    help='Minimum improvement to reset patience (absolute).')
parser.add_argument('--K', type=int, default=2,
                    help='K used for fourier transformation')

parser.add_argument('--absolute', type=eval, default=False,
                    help='if distance for fourier transformation should be relative or absolute')

args = parser.parse_args()
args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
#8 because 0-3 is "unbound augc" and 4-7 bound ones
args.pad_token = 8
args.num_classes = 8
if not hasattr(args, "local_rank"):
    print("No local_rank provided; falling back to 0")
    args.local_rank = 0
else:
     print(f"[DEBUG] local_rank = {args.local_rank}")
if args.ddp:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    args.device = f'cuda:{args.local_rank}'
else:
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
if is_main_process():
    print(f"Running on {args.device}")
'''
#adapted for DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
dist.init_process_group(backend='nccl')
torch.cuda.set_device(args.local_rank)
args.device = f'cuda:{args.local_rank}'
'''

set_seeds(args.seed)
##################
## Specify data ##
##################

train_loader, eval_loader, num_classes = get_data(args)
args.num_classes = num_classes
data_id = get_data_id(args)

###################
## Specify model ##
###################

#model = get_model(args)
#model = get_model_class(args)

#model = get_model_bind(args)

model = get_model_bind_RNAfold(args)
model_id = get_model_id(args)

#######################
## Specify optimizer ##
#######################

optimizer, scheduler_iter, scheduler_epoch = get_optim(args, model)
optim_id = get_optim_id(args)

##############
## Training ##
##############

exp = Experiment(args=args,
                 data_id=data_id,
                 model_id=model_id,
                 optim_id=optim_id,
                 train_loader=train_loader,
                 eval_loader=eval_loader,
                 model=model,
                 optimizer=optimizer,
                 scheduler_iter=scheduler_iter,
                 scheduler_epoch=scheduler_epoch)

exp.run()


#how to run ddp