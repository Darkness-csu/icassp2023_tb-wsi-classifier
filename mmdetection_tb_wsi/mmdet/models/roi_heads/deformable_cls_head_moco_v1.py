import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import build_loss
from ..utils import concat_all_gather

# from ..builder import build_loss

class MLPBlock(nn.Module):
    def __init__(self, mlp_dim:int, hidden_dim:int, dropout = 0.):
        super(MLPBlock, self).__init__()
        self.mlp_dim = mlp_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.Linear1 = nn.Linear(mlp_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.Linear2 = nn.Linear(hidden_dim, mlp_dim)
    def forward(self,x):
        x = self.Linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.Linear2(x)
        x = self.dropout(x)
        return x

class Mixer_struc(nn.Module):
    def __init__(self, patches: int , token_dim: int, dim: int,channel_dim: int,dropout = 0.):
        super(Mixer_struc, self).__init__()
        self.patches = patches
        self.channel_dim = channel_dim
        self.token_dim = token_dim
        self.dropout = dropout

        self.MLP_block_token = MLPBlock(patches,token_dim,self.dropout)
        self.MLP_block_chan = MLPBlock(dim,channel_dim,self.dropout)
        self.LayerNorm = nn.LayerNorm(dim)

    def forward(self,x):
        out = self.LayerNorm(x)
        out = out.permute(0,2,1)
        out = self.MLP_block_token(out)
        out = out.permute(0,2,1)
        out += x
        out2 = self.LayerNorm(out)
        out2 = self.MLP_block_chan(out2)
        out2+=out
        return out2

# class MLP_Mixer(nn.Module):
#     def __init__(self, patch_num, token_dim, channel_dim, dim, num_blocks):
#         super(MLP_Mixer, self).__init__()
#         n_patches =patch_num
#         self.blocks = nn.ModuleList([
#             Mixer_struc(patches=n_patches, token_dim=token_dim,channel_dim=channel_dim,dim=dim) for i in range(num_blocks)
#         ])

#         self.Layernorm1 = nn.LayerNorm(dim)
#         self.classifier = nn.Linear(dim, 1)
#     def forward(self,x):
#         for block in self.blocks:
#             x = block(x)
#         out = self.Layernorm1(x)
#         out = out.mean(dim = 1) # out（n_sample,dim）
#         result = self.classifier(out)
#         return out,result

class Encoder(nn.Module):
    def __init__(self, patch_num, token_dim, channel_dim, dim, num_blocks):
        super(Encoder, self).__init__()
        n_patches =patch_num
        self.dim = dim
        self.blocks = nn.ModuleList([
            Mixer_struc(patches=n_patches, token_dim=token_dim,channel_dim=channel_dim,dim=dim) for i in range(num_blocks)
        ])
        self.Layernorm1 = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, 1)
        self.proj = nn.Sequential(
                nn.Linear(dim, 4*dim), nn.ReLU(), nn.Linear(4*dim, 128)
            )
    def forward(self,x):
        for block in self.blocks:
            x = block(x)
        out = self.Layernorm1(x)
        feat = out.mean(dim = 1) # out（n_sample,dim）
        proj = self.proj(feat)
        cls_result = self.classifier(feat)
        return feat,proj,cls_result



class CLS_Head_MoCo_V1(nn.Module): #包含小型的MLP Mixer
    def __init__(self,
                num_class=2,
                dim=256,
                K=4096,
                m=0.999,
                T=0.07,
                loss_weight=dict(
                    cls=1.,
                    supcon=1.,
                ),
                pos_list=[1, 2, 3, 4, 5]
                ):
        super().__init__()
        self.pos_list = pos_list
        self.num_class = num_class
        self.loss_weight_cls = loss_weight['cls']
        self.loss_weight_supcon = loss_weight['supcon']
        self.dim = dim
        self.K = K
        self.m = m
        self.T = T
        self.K_div = int(self.K/self.num_class)
        # self.loss_cls_binary = build_loss(loss_cls_binary)
        #self.mlp_mixer = MLP_Mixer(patch_num=300, dim=256, num_blocks=4, token_dim=128, channel_dim=1024)
        self.encoder_q = Encoder(patch_num=300, dim=self.dim, num_blocks=6, token_dim=128, channel_dim=1024)
        self.encoder_k = Encoder(patch_num=300, dim=self.dim, num_blocks=6, token_dim=128, channel_dim=1024)
        # initialize encoder_k by encoder_q
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  
            param_k.requires_grad = False  # not update by gradient
        # create the queue
        self.register_buffer("queue", torch.randn(self.num_class, self.K_div, 128))
        self.register_buffer("update_mask", torch.zeros(self.num_class, self.K_div, dtype=torch.bool))
        self.queue = nn.functional.normalize(self.queue, dim=(0,1))
        self.register_buffer("queue_ptr", torch.zeros(self.num_class, dtype=torch.int))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        # gather keys before updating queue
        keys = concat_all_gather(keys) #num_gpu*n*D
        labels = concat_all_gather(labels)
        assert keys.shape[0] == labels.shape[0]  # for safety

        for i in range(self.num_class):
            i_keys = keys[labels.squeeze(dim=1) == i]
            num = int(i_keys.shape[0])
            if num == 0:
                continue
            num_remain = self.K_div - self.queue_ptr[i]
            if num_remain < num:
                self.queue[i,self.queue_ptr[i]:,:] = i_keys[:num_remain]
                self.queue[i,:(num - num_remain),:] = i_keys[num_remain:]
            else:
                self.queue[i,self.queue_ptr[i]:self.queue_ptr[i]+num,:] = i_keys
            if not self.update_mask[i,:].all(): 
                if num_remain < num:
                    self.update_mask[i,self.queue_ptr[i]:] = True
                else: 
                    self.update_mask[i,self.queue_ptr[i]:self.queue_ptr[i]+num] = True
            self.queue_ptr[i] = (self.queue_ptr[i] + num) % (self.K_div)
            #print(self.queue_ptr[i])

        

    def process_labels(self, gt_labels, device):
        #print(gt_labels)
        labels = []
        for gt in gt_labels:
            label = 0.  # negative
            for index in self.pos_list:
                if index in gt:
                    label = 1.  # positive
                    break
            labels.append(label)
        return torch.tensor(labels).reshape(-1, 1).to(device)


    #only compute contrastive loss
    def moco_supcon_loss(self,projs,labels):
        '''
        projs shape(N,128)
        labels shape(N)
        update_mask shape(num_class, self.K/self.num_class)
        queue shape(num_class,self.K/self.num_class,128)
        '''
        #初始不计算,因为字典没引入新的元素
        if not self.update_mask.any():
            return torch.tensor(0.0).to(projs.device)*torch.sum(projs)
        
        mini_batch = projs.shape[0]

        update_mask = self.update_mask.clone().detach().unsqueeze(dim=0).repeat(mini_batch, 1, 1) #shape (N,num_class, self.K/self.num_class)
        
        queue_label_mask = torch.zeros(self.num_class, self.K_div, dtype=torch.int).cuda()
        for i in range(self.num_class):
            queue_label_mask[i] = i
        
        label_mask = torch.eq(labels.unsqueeze(dim=1).unsqueeze(dim=1), queue_label_mask.unsqueeze(dim=0)).cuda() #shape (N,num_class,self.K/self.num_class), dtype = bool

        dot_contrast = torch.div(torch.einsum('ik,jrk->ijr', projs, self.queue.clone().detach()), self.T)
        logits_max, _ = torch.max(dot_contrast, dim=1, keepdim=True)
        logits_max, _ = torch.max(logits_max, dim=2, keepdim=True)
        dot_contrast = dot_contrast - logits_max.detach()
        exp_logits = torch.exp(dot_contrast) * update_mask
        log_prob = dot_contrast - torch.log(exp_logits.sum(1, keepdim=True).sum(2, keepdim=True) + 1e-16)
        mean_log_prob_pos = (label_mask * log_prob).sum(2).sum(1) / (label_mask.sum(2).sum(1) + 1e-16)
        
        loss = -1*mean_log_prob_pos.mean()

        return loss


    def forward_train(self, x_in, gt_labels=None):
        _,proj_q,x_q = self.encoder_q(x_in)
        proj_q = nn.functional.normalize(proj_q, dim=1)
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            _,proj_k,_ = self.encoder_k(x_in)
            proj_k = nn.functional.normalize(proj_k, dim=1)
        losses = dict()
        if gt_labels is None:
            losses['loss_binary_cls'] = torch.tensor(0.0).to(x_q.device)
            losses['loss_supcon_moco'] = torch.tensor(0.0).to(x_q.device)
        else:
            labels = self.process_labels(gt_labels, x_q.device)
            #print(labels)
            loss_cls = F.binary_cross_entropy_with_logits(x_q, labels.float())
            loss_supcon_moco = self.moco_supcon_loss(proj_q,labels)
            self._dequeue_and_enqueue(proj_k, labels)
            losses['loss_binary_cls'] = self.loss_weight_cls * loss_cls
            losses['loss_supcon_moco'] = self.loss_weight_supcon * loss_supcon_moco 
        return losses

    def forward_test(self, x_in):
        feat_q,_,x_q = self.encoder_q(x_in)
        out_score = torch.sigmoid(x_q)
        out_score = out_score.cpu().numpy().tolist()
        out_score = torch.Tensor(out_score)
        out = torch.cat((feat_q.cpu(), out_score),dim=1)
        #print(out.size())
        return out

    def simple_test(self, x):
        x = self.forward_test(x)
        #return x.detach().numpy().tolist()        #make sure decoder_querys and cls_result in the same device
        return x

    def onnx_export(self, x):
        return self.forward_test(x)