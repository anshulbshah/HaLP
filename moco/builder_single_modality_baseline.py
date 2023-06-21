import torch
import torch.nn as nn
import torch.nn.functional as F

from .GRU import BIGRU

# initilize weight
def weights_init_gru(model):
    with torch.no_grad():
        for child in list(model.children()):
            print(child)
            for param in list(child.parameters()):
                  if param.dim() == 2:
                        nn.init.xavier_uniform_(param)
    print('GRU weights initialization finished!')

class MoCo(nn.Module):
    def __init__(self, skeleton_representation, args_bi_gru, dim=128, K=65536, m=0.999, T=0.07,
                 teacher_T=0.05, student_T=0.1, cmd_weight=1.0, topk=1024, mlp=False, pretrain=True, modality_to_use='joint'):
        super(MoCo, self).__init__()
        self.pretrain = pretrain
        self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

        self.modality_to_use = modality_to_use
        if not self.pretrain:
            self.encoder_q = BIGRU(**args_bi_gru)
            weights_init_gru(self.encoder_q)
        else:
            self.K = K
            self.m = m
            self.T = T
            self.teacher_T = teacher_T
            self.student_T = student_T
            self.cmd_weight = cmd_weight
            self.topk = topk
            mlp=mlp
            print(" MoCo parameters",K,m,T,mlp)
            print(" CMD parameters: teacher-T %.2f, student-T %.2f, cmd-weight: %.2f, topk: %d"%(teacher_T,student_T,cmd_weight,topk))
            print(skeleton_representation)


            self.encoder_q = BIGRU(**args_bi_gru)
            self.encoder_k = BIGRU(**args_bi_gru)
            weights_init_gru(self.encoder_q)
            weights_init_gru(self.encoder_k)

            #projection heads
            if mlp:  # hack: brute-force replacement
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                    nn.ReLU(),
                                                    self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                    nn.ReLU(),
                                                    self.encoder_k.fc)
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)    # initialize
                param_k.requires_grad = False       # not update by gradient


            # create the queue
            self.register_buffer("queue", torch.randn(dim, self.K))
            self.queue = F.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

            self.total_sk_seen = 0

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr



    def forward(self, im_q, im_k=None, view='joint', knn_eval=False, update_prototypes=False,generate_this_step=False):
        N, C, T, V, M = im_q.size()
        im_q_motion = torch.zeros_like(im_q)
        im_q_motion[:, :, :-1, :, :] = im_q[:, :, 1:, :, :] - im_q[:, :, :-1, :, :]
        im_q_bone = torch.zeros_like(im_q)
        for v1, v2 in self.Bone:
            im_q_bone[:, :, :, v1 - 1, :] = im_q[:, :, :, v1 - 1, :] - im_q[:, :, :, v2 - 1, :]
        # Permute and Reshape
        
        im_q = im_q.permute(0,2,3,1,4).reshape(N,T,-1)
        im_q_motion = im_q_motion.permute(0,2,3,1,4).reshape(N,T,-1)
        im_q_bone = im_q_bone.permute(0,2,3,1,4).reshape(N,T,-1)

        if not self.pretrain or knn_eval:
            if view == 'joint':
                return self.encoder_q(im_q, knn_eval)
            elif view == 'motion':
                return self.encoder_q_motion(im_q_motion, knn_eval)
            elif view == 'bone':
                return self.encoder_q_bone(im_q_bone, knn_eval)
            elif view == 'all':
                return (self.encoder_q(im_q, knn_eval) + \
                        self.encoder_q_motion(im_q_motion, knn_eval) + \
                            self.encoder_q_bone(im_q_bone, knn_eval)) / 3.
            else:
                raise ValueError        
        
        im_k_motion = torch.zeros_like(im_k)
        im_k_motion[:, :, :-1, :, :] = im_k[:, :, 1:, :, :] - im_k[:, :, :-1, :, :]

        im_k_bone = torch.zeros_like(im_k)
        for v1, v2 in self.Bone:
            im_k_bone[:, :, :, v1 - 1, :] = im_k[:, :, :, v1 - 1, :] - im_k[:, :, :, v2 - 1, :]

        # Permute and Reshape
        im_k = im_k.permute(0,2,3,1,4).reshape(N,T,-1)
        im_k_motion = im_k_motion.permute(0,2,3,1,4).reshape(N,T,-1)
        im_k_bone = im_k_bone.permute(0,2,3,1,4).reshape(N,T,-1)

        # compute query features
        if self.modality_to_use == 'joint':
            q = self.encoder_q(im_q)  # queries: NxC
            q = F.normalize(q, dim=1)
        
        elif self.modality_to_use == 'motion':
            q = self.encoder_q(im_q_motion)
            q = F.normalize(q, dim=1)
        
        elif self.modality_to_use == 'bone':
            q = self.encoder_q(im_q_bone)
            q = F.normalize(q, dim=1)

        # compute key features for  s1 and  s2  skeleton representations 
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            if self.modality_to_use == 'joint':
                k = self.encoder_k(im_k)  # keys: NxC
                k = F.normalize(k, dim=1)

            elif self.modality_to_use == 'motion':
                k = self.encoder_k(im_k_motion)
                k = F.normalize(k, dim=1)

            elif self.modality_to_use == 'bone':
                k = self.encoder_k(im_k_bone)
                k = F.normalize(k, dim=1)

        # MOCO
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels, torch.Tensor([0.0]), None, None