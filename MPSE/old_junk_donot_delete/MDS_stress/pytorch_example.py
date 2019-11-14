import torch as tr

x = tr.autograd.Variable(tr.Tensor([2]),requires_grad=True)
y = tr.autograd.Variable(tr.Tensor([2]),requires_grad=True)

z = 5*x**4 + 3*y**3 + 7*x**2 + 9*x - 5

z.backward()

x.grad
y.grad


x = tr.autograd.Variable(tr.Tensor([1]),requires_grad=True)
y=3+x**2





import torch
from torch.autograd import Variable
tns = torch.FloatTensor([3])
x = Variable(tns, requires_grad=True)
opt = torch.optim.Adam([x], lr=.01, betas=(0.5, 0.999))
for i in range(30):
    opt.zero_grad()
    z = 4+x*x
    z.backward() # Calculate gradients
    opt.step()
    print(x.grad)
