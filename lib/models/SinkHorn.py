import torch


class SinkhornDistance(torch.nn.Module):
    r"""
        Given two empirical measures each with :math:`P_1` locations
        :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
        outputs an approximation of the regularized OT cost for point clouds.
        Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
        'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
        'mean': the sum of the output will be divided by the number of
        elements in the output, 'sum': the output will be summed. Default: 'none'
        Shape:
            - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
            - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps=1e-3, max_iter=100, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, mu, nu, C):
        u = torch.ones_like(mu)
        v = torch.ones_like(nu)

        # Sinkhorn iterations
        for i in range(self.max_iter):
            v = self.eps * \
                (torch.log(
                    nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            u = self.eps * \
                (torch.log(
                    mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        # pi = torch.exp(self.M(C, U, V)).detach()
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))
        # return pi
        return cost, pi

    def M(self, C, u, v):
        '''
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / epsilon$"
        '''
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps


if __name__ == '__main__':
    sinkhorn = SinkhornDistance(eps=0.1, max_iter=50)

    att = torch.randn((3, 10))
    att.requires_grad = True
    softmax_dim0 = torch.nn.Softmax(dim=0)
    att_norm = softmax_dim0(att)
    mu = torch.tensor([0.75, 0.125, 0.125]) * att.size(1)
    mu.requires_grad = True
    nu = torch.ones(att.size(1)) / att.size(1)  # normal distribution
    nu.requires_grad = True

    cost, pi = sinkhorn(mu, nu, att_norm)
    print(mu.size(), nu.size(), cost.item(), pi.size())

    cost.backward()
    print(mu.grad[0])
    print(nu.grad[0])
    print(att.grad[0, 0])

    # temporal_len = 128
    # mu = torch.ones(2) * temporal_len
    # nu = torch.ones(128)
    # C = torch.randn((2, 128))
    # sinkhorn = SinkhornDistance()
    #
    # cost, pi = sinkhorn(mu, nu, C)
    # print(mu.size(), nu.size(), pi.size())
