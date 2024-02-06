from federatedscope.register import register_regularizer
try:
    from torch.nn import Module
    import torch
except ImportError:
    Module = object
    torch = None

REGULARIZER_NAME = "proximal_regularizer"


class ProximalRegularizer(Module):
    """Returns the norm of the specific weight update.

        Arguments:
            p (int): The order of norm.
            tensor_before: The original matrix or vector
            tensor_after: The updated matrix or vector

        Returns:
            Tensor: the norm of the given udpate.
    """
    def __init__(self):
        super(ProximalRegularizer, self).__init__()

    def forward(self, ctx, p=2):
        norm = 0.
        # for w_init, w in zip(ctx.weight_init, ctx.model.parameters()):
        #     norm += torch.pow(torch.norm(w - w_init, p), p)
        for name, param in ctx.model.named_parameters():
            # print(name)
            if param.requires_grad:
                if 'lora_B' in name:
                    B = param
                    B_BT = torch.matmul(B, B.transpose(0, 1))
                    I_B = torch.eye(B.size(0), device=B.device)  # Identity matrix matching B's row size
                    norm_x = torch.norm(B_BT - I_B, p='fro') ** 2
                elif 'lora_A' in name:
                    A = param
                    AT_A = torch.matmul(A.transpose(0, 1), A)
                    I_A = torch.eye(A.size(1), device=A.device)  # Identity matrix matching A's column size
                    norm_x = torch.norm(AT_A - I_A, p='fro') ** 2
        # for w in ctx.model.parameters():
                norm += norm_x
        # print(norm)
        return norm * 1. / float(p)


def call_proximal_regularizer(type):
    if type == REGULARIZER_NAME:
        regularizer = ProximalRegularizer
        return regularizer


register_regularizer(REGULARIZER_NAME, call_proximal_regularizer)
