import torch
from torch.optim import Optimizer
# from torch.optim.optimizer import Optimizer, required
from abc import *
import torch.distributed as dist


class PerAvgOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(PerAvgOptimizer, self).__init__(params, defaults)

    def step(self, beta=0):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if(beta != 0):
                    p.data.add_(other=d_p, alpha=-beta)
                else:
                    p.data.add_(other=d_p, alpha=-group['lr'])


class FEDLOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, server_grads=None, pre_grads=None, eta=0.1):
        self.server_grads = server_grads
        self.pre_grads = pre_grads
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, eta=eta)
        super(FEDLOptimizer, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            i = 0
            for p in group['params']:
                p.data.add_(- group['lr'] * (p.grad.data + group['eta'] * \
                    self.server_grads[i] - self.pre_grads[i]))
                i += 1


class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)

    def step(self, local_model, device):
        group = None
        weight_update = local_model.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                localweight = localweight.to(device)
                # approximate local model
                p.data = p.data - group['lr'] * (p.grad.data + group['lamda'] * (p.data - localweight.data) + group['mu'] * p.data)

        return group['params']


# class pFedMeOptimizer(Optimizer):
#     def __init__(self, params, lr=0.01, lamda=0.1 , mu = 0.001):
#         #self.local_weight_updated = local_weight # w_i,K
#         if lr < 0.0:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         defaults = dict(lr=lr, lamda=lamda, mu = mu)
#         super(pFedMeOptimizer, self).__init__(params, defaults)
    
#     def step(self, local_weight_updated, closure=None):
#         loss = None
#         if closure is not None:
#             loss = closure
#         weight_update = local_weight_updated.copy()
#         for group in self.param_groups:
#             for p, localweight in zip( group['params'], weight_update):
#                 p.data = p.data - group['lr'] * (p.grad.data + group['lamda'] * (p.data - localweight.data) + group['mu']*p.data)
#         return  group['params'], loss
    
#     def update_param(self, local_weight_updated, closure=None):
#         loss = None
#         if closure is not None:
#             loss = closure
#         weight_update = local_weight_updated.copy()
#         for group in self.param_groups:
#             for p, localweight in zip( group['params'], weight_update):
#                 p.data = localweight.data
#         #return  p.data
#         return  group['params']


class APFLOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(APFLOptimizer, self).__init__(params, defaults)

    def step(self, beta=1, n_k=1):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = beta * n_k * p.grad.data
                p.data.add_(-group['lr'], d_p)


class PerturbedGradientDescent(Optimizer):
    def __init__(self, params, lr=0.01, mu=0.0):
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')

        default = dict(lr=lr, mu=mu)

        super().__init__(params, default)

    @torch.no_grad()
    def step(self, global_params, device):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                g = g.to(device)
                if p.grad is not None:
                    d_p = p.grad.data + group['mu'] * (p.data - g.data)
                    p.data.add_(d_p, alpha=-group['lr'])

class BaseOptimizer(metaclass=ABCMeta):
    """Federated optimization algorithm.
    """
    @abstractmethod
    def step(self, closure=None):
        raise NotImplementedError

    @abstractmethod
    def accumulate(self, **kwargs):
        raise NotImplementedError

class FedadamOptimizer(BaseOptimizer, torch.optim.Optimizer):
    def __init__(self, params, **kwargs):
        lr = kwargs.get('lr')
        v0 = kwargs.get('v0')
        tau = kwargs.get('tau')
        momentum = kwargs.get('betas')
        defaults = dict(lr=lr, momentum=momentum, v0=v0, tau=tau)
        BaseOptimizer.__init__(self);
        torch.optim.Optimizer.__init__(self, params=params, defaults=defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for idx, group in enumerate(self.param_groups):
            (beta1, beta2) = group['momentum']
            tau = group['tau']
            lr = group['lr']
            v0 = group['v0']
            for param in group['params']:
                if param.grad is None:
                    continue
                # get (\Delta_t)
                delta = -param.grad.data

                if idx == 0:  # idx == 0: parameters; optimize according to algorithm
                    # calculate m_t
                    if 'momentum_buffer1' not in self.state[param]:
                        self.state[param]['momentum_buffer1'] = torch.zeros_like(param).detach()
                    self.state[param]['momentum_buffer1'].mul_(beta1).add_(
                        delta.mul(1. - beta1))  # \beta1 * m_t + (1 - \beta1) * \Delta_t
                    m_new = self.state[param]['momentum_buffer1']

                    # calculate v_t
                    if 'momentum_buffer2' not in self.state[param]:
                        self.state[param]['momentum_buffer2'] = v0 * beta2 + delta.pow(2).mul(1. - beta2)
                    self.state[param]['momentum_buffer2'].mul_(beta2).add_(
                        delta.pow(2).mul(1. - beta2))  # \beta2 * v_t + (1 - \beta2) * \Delta_t^2
                    v_new = self.state[param]['momentum_buffer2']

                    # update parameters
                    param.data.add_(m_new.div(v_new.pow(0.5).add(tau)).mul(lr))
                elif idx == 1:  # idx == 1: buffers; just averaging
                    param.data.add_(delta)
        return loss

    def accumulate(self, mixing_coefficient, local_layers_iterator,
                   check_if=lambda name: 'num_batches_tracked' in name):
        for group in self.param_groups:
            for server_param, (name, local_signals) in zip(group['params'], local_layers_iterator):
                if check_if(name):
                    server_param.data.zero_()
                    server_param.data.grad = torch.zeros_like(server_param)
                    continue
                local_delta = (server_param - local_signals).mul(mixing_coefficient).data.type(server_param.dtype)
                if server_param.grad is None:  # NOTE: grad buffer is used to accumulate local updates!
                    server_param.grad = local_delta
                else:
                    server_param.grad.data.add_(local_delta)

class FedadagradOptimizer(BaseOptimizer, torch.optim.Optimizer):
    def __init__(self, params, **kwargs):
        lr = kwargs.get('lr')
        v0 = kwargs.get('v0')
        tau = kwargs.get('tau')
        momentum = kwargs.get('beta')
        defaults = dict(lr=lr, momentum=momentum, v0=v0, tau=tau)
        BaseOptimizer.__init__(self);
        torch.optim.Optimizer.__init__(self, params=params, defaults=defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for idx, group in enumerate(self.param_groups):
            beta = group['momentum']
            tau = group['tau']
            lr = group['lr']
            v0 = group['v0']
            for param in group['params']:
                if param.grad is None:
                    continue
                # get (\Delta_t)
                delta = -param.grad.data

                if idx == 0:  # idx == 0: parameters; optimize according to algorithm
                    # calculate m_t
                    if 'momentum_buffer1' not in self.state[param]:
                        self.state[param]['momentum_buffer1'] = torch.zeros_like(param).detach()
                    self.state[param]['momentum_buffer1'].mul_(beta).add_(
                        delta.mul(1. - beta))  # \beta * m_t + (1 - \beta) * \Delta_t
                    m_new = self.state[param]['momentum_buffer1']

                    # calculate v_t
                    if 'momentum_buffer2' not in self.state[param]:
                        self.state[param]['momentum_buffer2'] = v0 + delta.pow(2)
                    self.state[param]['momentum_buffer2'].add_(delta.pow(2))  # v_t + \Delta_t^2

                    # update parameters
                    param.data.add_((m_new.div(self.state[param]['momentum_buffer2'].pow(0.5).add(tau))).mul(lr))
                elif idx == 1:  # idx == 1: buffers; just averaging
                    param.data.add_(delta)
        return loss

    def accumulate(self, mixing_coefficient, local_layers_iterator,
                   check_if=lambda name: 'num_batches_tracked' in name):
        for group in self.param_groups:
            for server_param, (name, local_signals) in zip(group['params'], local_layers_iterator):
                if check_if(name):
                    server_param.data.zero_()
                    server_param.data.grad = torch.zeros_like(server_param)
                    continue
                local_delta = (server_param - local_signals).mul(mixing_coefficient).data.type(server_param.dtype)
                if server_param.grad is None:  # NOTE: grad buffer is used to accumulate local updates!
                    server_param.grad = local_delta
                else:
                    server_param.grad.data.add_(local_delta)


class FedyogiOptimizer(BaseOptimizer, torch.optim.Optimizer):
    def __init__(self, params, **kwargs):
        lr = kwargs.get('lr')
        v0 = kwargs.get('v0')
        tau = kwargs.get('tau')
        momentum = kwargs.get('betas')
        defaults = dict(lr=lr, momentum=momentum, v0=v0, tau=tau)
        BaseOptimizer.__init__(self);
        torch.optim.Optimizer.__init__(self, params=params, defaults=defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for idx, group in enumerate(self.param_groups):
            (beta1, beta2) = group['momentum']
            tau = group['tau']
            lr = group['lr']
            v0 = group['v0']
            for param in group['params']:
                if param.grad is None:
                    continue
                # get (\Delta_t)
                delta = -param.grad.data

                if idx == 0:  # idx == 0: parameters; optimize according to algorithm
                    # calculate m_t
                    if 'momentum_buffer1' not in self.state[param]:
                        self.state[param]['momentum_buffer1'] = torch.zeros_like(param).detach()
                    self.state[param]['momentum_buffer1'].mul_(beta1).add_(
                        delta.mul(1. - beta1))  # \beta1 * m_t + (1 - \beta1) * \Delta_t
                    m_new = self.state[param]['momentum_buffer1']

                    # calculate v_t
                    if 'momentum_buffer2' not in self.state[param]:
                        self.state[param]['momentum_buffer2'] = v0 - delta.pow(2).mul(1. - beta2).mul(
                            (v0 - delta).sign())
                    v_curr = self.state[param]['momentum_buffer2']
                    self.state[param]['momentum_buffer2'].sub_(delta.pow(2).mul(1. - beta2).mul(
                        v_curr.sub(delta.pow(2)).sign()))  # v_t - (1 - \beta2) * \Delta_t^2 * sgn(v_t - \Delta_t)
                    v_new = self.state[param]['momentum_buffer2']

                    # update parameters
                    param.data.add_((m_new.div(v_new.pow(0.5).add(tau))).mul(lr))
                elif idx == 1:  # idx == 1: buffers; just averaging
                    param.data.add_(delta)
        return loss

    def accumulate(self, mixing_coefficient, local_layers_iterator,
                   check_if=lambda name: 'num_batches_tracked' in name):
        for group in self.param_groups:
            for server_param, (name, local_signals) in zip(group['params'], local_layers_iterator):
                if check_if(name):
                    server_param.data.zero_()
                    server_param.data.grad = torch.zeros_like(server_param)
                    continue
                local_delta = (server_param - local_signals).mul(mixing_coefficient).data.type(server_param.dtype)
                if server_param.grad is None:  # NOTE: grad buffer is used to accumulate local updates!
                    server_param.grad = local_delta
                else:
                    server_param.grad.data.add_(local_delta)


class FedNova(Optimizer):
    r"""Implements federated normalized averaging (FedNova).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        ratio (float): relative sample size of client
        gmf (float): global/server/slow momentum factor
        mu (float): parameter for proximal local SGD
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, ratio, gmf, mu=0, lr=0.01, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, variance=0):

        self.gmf = gmf
        self.ratio = ratio
        self.momentum = momentum
        self.mu = mu
        self.local_normalizing_vec = 0
        self.local_counter = 0
        self.local_steps = 0

        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, variance=variance)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(FedNova, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FedNova, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        loss = None
        if closure is not None:
            loss = closure()

        # scale = 1**self.itr

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                param_state = self.state[p]
                if 'old_init' not in param_state:
                    param_state['old_init'] = torch.clone(p.data).detach()

                local_lr = group['lr']

                # apply momentum updates
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # apply proximal updates
                if self.mu != 0:
                    d_p.add_(self.mu, p.data - param_state['old_init'])

                # update accumalated local updates
                if 'cum_grad' not in param_state:
                    param_state['cum_grad'] = torch.clone(d_p).detach()
                    param_state['cum_grad'].mul_(local_lr)

                else:
                    param_state['cum_grad'].add_(local_lr, d_p)

                p.data.add_(-local_lr, d_p)

        # compute local normalizing vector a_i
        if self.momentum != 0:
            self.local_counter = self.local_counter * self.momentum + 1
            self.local_normalizing_vec += self.local_counter

        self.etamu = local_lr * self.mu
        if self.etamu != 0:
            self.local_normalizing_vec *= (1 - self.etamu)
            self.local_normalizing_vec += 1

        if self.momentum == 0 and self.etamu == 0:
            self.local_normalizing_vec += 1

        self.local_steps += 1

        return loss

    def average(self, weight=0, tau_eff=0):
        if weight == 0:
            weight = self.ratio
        if tau_eff == 0:
            if self.mu != 0:
                tau_eff_cuda = torch.tensor(self.local_steps * self.ratio).cuda()
            else:
                tau_eff_cuda = torch.tensor(self.local_normalizing_vec * self.ratio).cuda()
            dist.all_reduce(tau_eff_cuda, op=dist.ReduceOp.SUM)
            tau_eff = tau_eff_cuda.item()

        param_list = []
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                scale = tau_eff / self.local_normalizing_vec
                param_state['cum_grad'].mul_(weight * scale)
                param_list.append(param_state['cum_grad'])

        self.communicate(param_list, dist.all_reduce)

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                param_state = self.state[p]

                if self.gmf != 0:
                    if 'global_momentum_buffer' not in param_state:
                        buf = param_state['global_momentum_buffer'] = torch.clone(param_state['cum_grad']).detach()
                        buf.div_(lr)
                    else:
                        buf = param_state['global_momentum_buffer']
                        buf.mul_(self.gmf).add_(1 / lr, param_state['cum_grad'])
                    param_state['old_init'].sub_(lr, buf)
                else:
                    param_state['old_init'].sub_(param_state['cum_grad'])

                p.data.copy_(param_state['old_init'])
                param_state['cum_grad'].zero_()

                # Reinitialize momentum buffer
                if 'momentum_buffer' in param_state:
                    param_state['momentum_buffer'].zero_()

        self.local_counter = 0
        self.local_normalizing_vec = 0
        self.local_steps = 0

    def communicate(self, tensors, communication_op):
        """
        Reference: https://github.com/facebookresearch/stochastic_gradient_push

        Communicate a list of tensors.
        Arguments:
            tensors (Iterable[Tensor]): list of tensors.
            communication_op: a method or partial object which takes a tensor as
                input and communicates it. It can be a partial object around
                something like torch.distributed.all_reduce.
        """
        flat_tensor = self.flatten_tensors(tensors)
        communication_op(tensor=flat_tensor)
        for f, t in zip(self.unflatten_tensors(flat_tensor, tensors), tensors):
            t.set_(f)

    def flatten_tensors(self, tensors):
        """
        Reference: https://github.com/facebookresearch/stochastic_gradient_push

        Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
        same dense type.
        Since inputs are dense, the resulting tensor will be a concatenated 1D
        buffer. Element-wise operation on this buffer will be equivalent to
        operating individually.
        Arguments:
            tensors (Iterable[Tensor]): dense tensors to flatten.
        Returns:
            A 1D buffer containing input tensors.
        """
        if len(tensors) == 1:
            return tensors[0].view(-1).clone()
        flat = torch.cat([t.view(-1) for t in tensors], dim=0)
        return flat

    def unflatten_tensors(self, flat, tensors):
        """
        Reference: https://github.com/facebookresearch/stochastic_gradient_push

        View a flat buffer using the sizes of tensors. Assume that tensors are of
        same dense type, and that flat is given by flatten_dense_tensors.
        Arguments:
            flat (Tensor): flattened dense tensors to unflatten.
            tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
                unflatten flat.
        Returns:
            Unflattened dense tensors with sizes same as tensors and values from
            flat.
        """
        outputs = []
        offset = 0
        for tensor in tensors:
            numel = tensor.numel()
            outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
            offset += numel
        return tuple(outputs)
