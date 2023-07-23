import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

def generate_positives_using_mixup(prototypes,initial_value, inds=None,num_to_generate=1, args=None, highest_sim_inds=None):
    B,D = initial_value.shape
    start_point = initial_value.clone().unsqueeze(1).repeat(1,num_to_generate,1).view(initial_value.shape[0]*num_to_generate,initial_value.shape[1]) # repeat starting point and reshape to [BxNg] x D
    inds_to_sample = torch.from_numpy(np.random.choice(range(inds.shape[1]),(inds.shape[0],num_to_generate),replace=True)) #randomly choose a target index for num_generate
    selected_target = torch.cat([a[i].unsqueeze(0) for a, i in zip(inds, inds_to_sample)]).view(-1) #map to actual indices
    selected_prototype = prototypes.unsqueeze(0).repeat(selected_target.shape[0],1,1) #repeat 
    selected_prototype = selected_prototype[np.arange(selected_target.shape[0]),selected_target,:]

    p0 = prototypes.unsqueeze(0).repeat(selected_target.shape[0],1,1)
    indsm1 = highest_sim_inds.unsqueeze(-1).repeat(1,num_to_generate).view(-1)
    p0 = p0[np.arange(selected_target.shape[0]),indsm1,:] 
    sim_zk_diffP = torch.einsum('bd,bd->b', [start_point,p0-selected_prototype])   
    sim_psel_p0 = torch.einsum('bd,bd->b', [selected_prototype,p0])
    sim_psel_k = torch.einsum('bd,bd->b', [selected_prototype,start_point])   

    K = (1-sim_psel_p0)/(sim_zk_diffP + 1E-8)
    omega = torch.acos(sim_psel_k)
    max_t = ((1/(omega+1E-8))*torch.atan((torch.sin(omega))/(K + torch.cos(omega) + 1E-8)))
    min_t = torch.zeros_like(max_t)
    
    upper_limit = float(args.lambda_pos)
    if upper_limit == 1000.0:
        t_to_use = max_t - 1E-2
    else:
        zero_one = torch.FloatTensor(start_point.shape[0]).uniform_(0,upper_limit).to(start_point.device)
        t_to_use = zero_one*min_t + (1-zero_one)*max_t

    alpha = (torch.sin((1-t_to_use)*omega)/(torch.sin(omega) + 1E-8)).unsqueeze(-1)
    beta = (torch.sin((t_to_use*omega))/(torch.sin(omega) + 1E-8)).unsqueeze(-1)
    
    generated_points = alpha*start_point + beta*selected_prototype
    
    return generated_points.view(B,num_to_generate,D), selected_target.view(B,num_to_generate), selected_prototype.view(B,num_to_generate,D)

def pos_generator(anchor=None, prototypes=None, args=None, precomputed_similarity=None):
    # anchor : B x 128
    # prototypes : NP x 128

    similarities = precomputed_similarity

    batch,K = similarities.shape
    _, D = prototypes.shape
    num_to_ignore = args.num_closest_to_ignore_positives

    inds = torch.argsort(similarities,descending=False) # similarities low to high
    inds_high_similarity = inds[:,-num_to_ignore-args.skip_closest_positives:-args.skip_closest_positives]
    
    generate_normally = True
    if generate_normally:
        generated_points, selected_target, selected_prototypes = generate_positives_using_mixup(prototypes,anchor, inds=inds_high_similarity,num_to_generate=args.num_positives, \
                                                                        args=args, highest_sim_inds=inds[:,-1])

    generated_points_normed = F.normalize(generated_points, dim=-1)
    return generated_points_normed, generated_points, selected_prototypes