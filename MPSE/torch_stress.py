import torch

def stress(x, d):
    s = torch.zeros(1)
    for i in range(len(x)):
        for j in range(i):
            de = torch.sqrt(
                    torch.sum( 
                        (x[i,0:2]-x[j,0:2])**2
                ))
            s += (de - d[i][j])**2
    return s

def test_grad():
  x = torch.rand(3, 3, requires_grad = True)
  d = torch.ones(3, 3)

  for i in range(len(d)):
    for j in range(len(d)):
      if i==j:
        d[i][j] = 0
      else:
        d[i][j] = 1
  y = stress(x, d)
  y.backward()
  print(x.grad)


