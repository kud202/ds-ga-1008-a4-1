require 'nngraph'
x = nn.Identity()()
y = nn.Identity()()
z = nn.Identity()()
a = nn.CAddTable()({
  nn.CMulTable()({
      nn.Square()({
        nn.Tanh()({
          nn.Linear(4,2)({
            x
          })
        })
      }),
      nn.Square()({
        nn.Tanh()({
          nn.Linear(5,2)({
            y
          })
        })
      })
  }),
  z
})
g = nn.gModule({x,y,z},{a})
x_in = torch.rand(1,4)
y_in = torch.rand(1,5)
z_in = torch.rand(1,2)
grad = torch.Tensor({1,1})
f = g:forward({x_in,y_in,z_in})
b = g:backward({x_in,y_in,z_in},grad)
