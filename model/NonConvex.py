import torchfrom torch import nnfrom torch.autograd import Variableclass Function1(nn.Module):    def __init__(self):        super(Function1, self).__init__()        self.x = nn.Parameter(torch.tensor(5.), requires_grad=True)        self.y = nn.Parameter(torch.tensor(5.), requires_grad=True)    def forward(self):        y = torch.pow(self.x, 2) + torch.pow(self.y, 2) + 5 * torch.sin(self.x + torch.pow(self.y, 2))        return y