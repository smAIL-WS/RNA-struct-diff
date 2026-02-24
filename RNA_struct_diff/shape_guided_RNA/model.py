import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from diffusion_utils.diffusion_multinomial import MultinomialDiffusion, MultinomialDiffusion_shapeless, MultinomialDiffusion_class_guided,MultinomialDiffusion_class_padding
from layers.layers import SegmentationUnet_pairwise
from layers.diffusion_multinomial_adapted import  MultinomialDiffusion_bound_token,MultinomialDiffusion_RNAfold_bound
from layers.layers_1D import SegmentationUnet1D

def add_model_args(parser):
    # Model params
    parser.add_argument('--loss_type', type=str, default='vb_stochastic')
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--diffusion_dim', type=int, default=64)
    parser.add_argument('--dp_rate', type=float, default=0.)


def get_model_id(args):
    return 'multinomial_diffusion'


def get_model(args):

    '''
    data_shape = torch.Size(data_shape)
    current_shape = data_shape

    if (data_shape[-1] // 4) % 2 == 0:
        dim_mults = (1, 2, 4, 8)
    else:
        dim_mults = (1, 4, 8)
    '''
    #dim_mults = (1, 4, 8)
    dim_mults = getattr(args, 'dim_mults', (1, 4, 8))
    dynamics = SegmentationUnet_pairwise(
        num_classes=args.num_classes,
        dim=args.diffusion_dim,
        num_steps=args.diffusion_steps,
        dim_mults=dim_mults,
        dropout=args.dp_rate,
        padding=args.padding
    )
    base_dist = MultinomialDiffusion_shapeless(
        args.num_classes, dynamics, timesteps=args.diffusion_steps,
        loss_type=args.loss_type)

    return base_dist


def get_model_class(args):

    '''
    data_shape = torch.Size(data_shape)
    current_shape = data_shape

    if (data_shape[-1] // 4) % 2 == 0:
        dim_mults = (1, 2, 4, 8)
    else:
        dim_mults = (1, 4, 8)
    '''
    dim_mults = (1, 4, 8)
    if getattr(args, "arch", None) == "1d":
        print("1d training initiated")
        dynamics = SegmentationUnet1D(
            num_classes=args.num_classes,
            dim=args.diffusion_dim,
            num_steps=args.diffusion_steps,
            dim_mults=dim_mults,
            dropout=args.dp_rate,
            classes_guidance=None,
            #padding=args.padding
            padding = True,
            pad_token = args.pad_token,
            K=val if (val := getattr(args, "K", None)) is not None else 2,
            absolute= val if (val := getattr(args, "absolute", None)) is not None else False
        )
    else:
        dynamics = SegmentationUnet_pairwise(
            num_classes=args.num_classes,
            dim=args.diffusion_dim,
            num_steps=args.diffusion_steps,
            dim_mults=dim_mults,
            dropout=args.dp_rate,
            classes_guidance=None,
            padding=args.padding
        )


    '''
    dynamics = SegmentationUnet_pairwise(
        num_classes=args.num_classes,
        dim=args.diffusion_dim,
        num_steps=args.diffusion_steps,
        dim_mults=dim_mults,
        dropout=args.dp_rate,
        classes_guidance=None,
        padding=args.padding
    )
    '''
    if args.padding:
        print("hello!")
        base_dist = MultinomialDiffusion_class_padding(
            args.num_classes, dynamics, timesteps=args.diffusion_steps,
            loss_type=args.loss_type)

    else:
        base_dist = MultinomialDiffusion_class_guided(
            args.num_classes, dynamics, timesteps=args.diffusion_steps,
            loss_type=args.loss_type)

    return base_dist
def get_model_bind(args):
    dim_mults = (1, 4, 8)
    print(args.num_classes)
    print(args.pad_token)
    if getattr(args, "arch", None) == "1d":
        print("1d training initiated")
        dynamics = SegmentationUnet1D(
            num_classes=args.num_classes,
            dim=args.diffusion_dim,
            num_steps=args.diffusion_steps,
            dim_mults=dim_mults,
            dropout=args.dp_rate,
            classes_guidance=None,
            #padding=args.padding
            padding = True,
            pad_token = args.pad_token,
            K=val if (val := getattr(args, "K", None)) is not None else 2,
            absolute= val if (val := getattr(args, "absolute", None)) is not None else False
        )
    else:
        dynamics = SegmentationUnet_pairwise(
            num_classes=args.num_classes,
            dim=args.diffusion_dim,
            num_steps=args.diffusion_steps,
            dim_mults=dim_mults,
            dropout=args.dp_rate,
            classes_guidance=None,
            padding=True,
            pad_token=args.pad_token,
            project_contact_map=args.project_contact_map  # this enables or disables CNN projection
        )


    '''
    dynamics = SegmentationUnet_pairwise(
        num_classes=args.num_classes,
        dim=args.diffusion_dim,
        num_steps=args.diffusion_steps,
        dim_mults=dim_mults,
        dropout=args.dp_rate,
        classes_guidance=None,
        padding=True,
        pad_token=args.pad_token,
        project_contact_map=args.project_contact_map  # this enables or disables CNN projection
    )
    '''
    model = MultinomialDiffusion_bound_token(
        num_classes=args.num_classes,
        denoise_fn=dynamics,
        timesteps=args.diffusion_steps,
        loss_type=args.loss_type,
        pad_token=args.pad_token
    )

    return model

def get_model_bind_RNAfold(args):
    dim_mults = getattr(args, 'dim_mults', (1, 4, 8))
    print(args.num_classes)
    print(args.pad_token)
    if getattr(args, "arch", None) == "1d":
        print("1d training initiated")
        dynamics = SegmentationUnet1D(
            num_classes=args.num_classes,
            dim=args.diffusion_dim,
            num_steps=args.diffusion_steps,
            dim_mults=dim_mults,
            dropout=args.dp_rate,
            classes_guidance=None,
            #padding=args.padding
            padding = True,
            pad_token = args.pad_token,
            K=val if (val := getattr(args, "K", None)) is not None else 2,
            absolute= val if (val := getattr(args, "absolute", None)) is not None else False
        )
    else:
        dynamics = SegmentationUnet_pairwise(
            num_classes=args.num_classes,
            dim=args.diffusion_dim,
            num_steps=args.diffusion_steps,
            dim_mults=dim_mults,
            dropout=args.dp_rate,
            classes_guidance=None,
            padding=True,
            pad_token=args.pad_token,
            project_contact_map=args.project_contact_map  # this enables or disables CNN projection
        )
    model = MultinomialDiffusion_RNAfold_bound(
        num_classes=args.num_classes,
        denoise_fn=dynamics,
        timesteps=args.diffusion_steps,
        loss_type=args.loss_type,
        pad_token=args.pad_token,
        alphabet={0: 'A', 1: 'C', 2: 'G', 3: 'U', 4: "A", 5: "C", 6: "G", 7: "U", 8: ""}
    )

    return model