import torch
import torch.nn as nn

from collections import OrderedDict

  
def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()

class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # For compatibility with Torchmeta
        self.named_meta_parameters = self.named_parameters
        self.meta_parameters = self.parameters

    def update_params(self, loss, params=None, step_size=0.5, first_order=False):
        """Apply one step of gradient descent on the loss function `loss`, with 
        step-size `step_size`, and returns the updated parameters of the neural 
        network.
        """
        if params is None:
            params = OrderedDict(self.named_meta_parameters())

        grads = torch.autograd.grad(loss, params.values(),
                                    create_graph=not first_order)
        updated_params = OrderedDict()
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - step_size * grad
        return updated_params

    # def update_params(
    #     self, loss, params=None, step_size=0.5, first_order=False
    # ):
    #     """
    #     Uses .backward() plus a functional parameter update to retain grad_fn.

    #     Args:
    #         loss (torch.Tensor): Scalar loss to backprop from.
    #         params (OrderedDict): Parameter name -> nn.Parameter dictionary.
    #         step_size (float): Learning rate or step size.
    #         create_graph (bool): Whether to create a graph for higher-order gradients.

    #     Returns:
    #         OrderedDict: A new OrderedDict of updated parameters,
    #                     each with a grad_fn if create_graph=True.
    #     """
    #     # 1. Prepare parameters
    #     if params is None:
    #         params = OrderedDict(self.named_meta_parameters())  # or named_parameters()

    #     # 2. Zero out existing gradients
    #     for p in params.values():
    #         if p.grad is not None:
    #             p.grad.zero_()

    #     # 3. Standard backward pass
    #     #    create_graph=True is needed if you want second-order gradients
    #     loss.backward(create_graph=not first_order)

    #     # 4. Build new parameter dictionary without in-place data changes
    #     updated_params = OrderedDict()
    #     for name, param in params.items():
    #         # The new param is a tensor with grad_fn
    #         # param.grad is now computed by .backward()
    #         param_new = param - step_size * param.grad
    #         updated_params[name] = param_new

    #     return updated_params
