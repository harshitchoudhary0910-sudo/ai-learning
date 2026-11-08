import torch
import torch.nn as nn
import torch.functional as f

class MLP(nn.Module):

   def __init__(self,emb_dim,hidden_size):       
        super().__init__()
        self.dim=emb_dim
        self.C = nn.Embedding(29,emb_dim)
        self.linear1= nn.Linear(emb_dim*3,hidden_size)
        self.linear2=nn.Linear(hidden_size,29)

   def forward(self,X):
       
       emb=self.C(X)
       embcat = emb.view(len(X), 3*self.dim)
       h=torch.tanh(self.linear1(embcat))
       return self.linear2(h)






                                 



        

