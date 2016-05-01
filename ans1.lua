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
