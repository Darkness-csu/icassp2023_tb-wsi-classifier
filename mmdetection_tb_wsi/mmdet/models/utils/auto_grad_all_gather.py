import torch
import torch.distributed as dist

class AllGather(torch.autograd.Function):
    """
    all_gather with gradient back-propagation
    """
    @staticmethod
    def forward(ctx, tensor_list, tensor):
        dist.all_gather(tensor_list, tensor)
        return tuple(tensor_list)

    @staticmethod
    def backward(ctx, *grad_list):
        grad_list = list(grad_list)
        rank = dist.get_rank()

        dist_ops = [
            dist.reduce(grad_list[i], i, async_op=True) for i in range(dist.get_world_size())
        ]

        for op in dist_ops:
            op.wait()

        return None, grad_list[rank]


all_gather = AllGather.apply

def concat_all_gather(tensor, gather_grad=False):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    if gather_grad:
        output = torch.cat(all_gather(tensors_gather, tensor), dim=0)
    else:
        dist.all_gather(tensors_gather, tensor)
        output = torch.cat(tensors_gather, dim=0)

    return output