import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as dsets
import torch.nn.functional as F
from   torch.nn.init import xavier_normal
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import numpy as np



def normal(tensor, mean=0, std=1):
    """Fills the input Tensor or Variable with values drawn from a normal distribution with the given mean and std
    Args:
        tensor: a n-dimension torch.Tensor
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.normal(w)
    """
    if isinstance(tensor, Variable):
        normal(tensor.data, mean=mean, std=std)
        return tensor
    else:
        return tensor.normal_(mean, std)


def uniform(tensor, a=0, b=1):
    """Fills the input Tensor or Variable with values drawn from the uniform
    distribution :math:`U(a, b)`.
    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        a: the lower bound of the uniform distribution
        b: the upper bound of the uniform distribution
    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.uniform(w)
    """
    if isinstance(tensor, Variable):
        uniform(tensor.data, a=a, b=b)
        return tensor

    return tensor.uniform_(a, b)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # initialize
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())

        # forward path
        out, _ = self.lstm(x, (h0, c0))
        shp=(out.size()[0], out.size()[1])
        out = out.contiguous().view(shp[0] *shp[1] , self.hidden_size)
        #out = self.fc(out[:, -1, :]) 
        out = self.fc(out)
        out = out.view(shp[0], shp[1], self.num_classes)
        return out
    
    def print_log(self):
        model_name = '_LSTM_'
        model_log = ' LSTM.......'
        return (model_name, model_log)


class RNN_LSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        

    def forward(self, x):
        outputs = []
        h_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        c_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            input_t = input_t.contiguous().view(input_t.size()[0], input_t.size()[-1])
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            outputs += [h_t]
        outputs = torch.stack(outputs, 1).squeeze(2)
        shp=(outputs.size()[0], outputs.size()[1])
        out = outputs.contiguous().view(shp[0] *shp[1] , self.hidden_size)
        out = self.fc(out)
        out = out.view(shp[0], shp[1], self.num_classes)

        return out

    def print_log(self):
        model_name = '_regular-LSTM_'
        model_log = ' Regular LSTM.......'
        return (model_name, model_log)  


class RNN_LSTM_attention(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_LSTM_attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.tanh = torch.nn.Tanh()
        self.w_t_1 = nn.Parameter(normal(torch.Tensor(self.hidden_size * 2, self.hidden_size), mean = 0.0, std = 0.01))
        self.w_t_2 = nn.Parameter(normal(torch.Tensor(self.hidden_size, 1), mean = 0.0, std = 0.01))
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        batch_size  = x.size(0)
        time_size   = x.size(1)
        input_size  = x.size(2)
        hidden_size = self.hidden_size

        outputs     = []
        h_t         = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        c_t         = Variable(torch.zeros(x.size(0), self.hidden_size).cuda()) 
        h_old       = h_t.view(batch_size, 1, hidden_size)

        attn_all    = []
        attn_w_viz  = []


        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            remember_size = h_old.size(1)
            input_t    = input_t.contiguous().view(input_t.size()[0], input_t.size()[-1])
             
            h_t, c_t   = self.lstm1(input_t, (h_t, c_t))
            h_repeated =  h_t.unsqueeze(1).repeat(1, remember_size, 1)
            mlp_h_attn = torch.cat((h_repeated, h_old), 2)
        
            mlp_h_attn = mlp_h_attn.view(batch_size*remember_size,hidden_size * 2)
            
            attn_w     = torch.mm(mlp_h_attn, self.w_t_1)
            
            attn_w     = self.tanh(attn_w)
            attn_w     = torch.mm(attn_w, self.w_t_2)
            
            h_old      = h_old.view(batch_size * remember_size, hidden_size)
            attn_w     = (attn_w.expand_as(h_old)).contiguous()
            attn_w     = attn_w.view(batch_size *  remember_size, hidden_size)
            h_old_w    = attn_w * h_old
            h_old_w    = h_old_w.view(batch_size, remember_size, hidden_size)
            attn_c     = torch.sum(h_old_w, 1).squeeze(1)
            h_old      = h_old.view(batch_size, remember_size, hidden_size)
            h_old      = torch.cat((h_old, h_t.view(batch_size, 1, hidden_size)), dim=1)
            outputs    += [h_t]
            attn_all += [attn_c]

        outputs = torch.stack(outputs, 1)
        attn_all  = torch.stack(attn_all, 1)
        #outputs   = torch.cat  ((outputs, attn_all), 2)
        outputs   += outputs + attn_all
        
        shp = outputs.size()
        
        out = outputs.contiguous().view(shp[0] *shp[1] ,shp[2] )
        out = self.fc(out)
        out = out.view(shp[0], shp[1], self.num_classes)

        return out

    def print_log(self):
        model_name = 'LSTM_self_attention_'
        model_log = ' LSTM with self attention.......'
        return (model_name, model_log)


class RNN_LSTM_attention_one_step(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_LSTM_attention_one_step, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.tanh = torch.nn.Tanh()
        self.w_t_1 = nn.Parameter(normal(torch.Tensor(self.hidden_size * 2, self.hidden_size), mean = 0.0, std = 0.01))
        self.w_t_2 = nn.Parameter(normal(torch.Tensor(self.hidden_size, 1), mean = 0.0, std = 0.01))
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        batch_size  = x.size(0)
        time_size   = x.size(1)
        input_size  = x.size(2)
        hidden_size = self.hidden_size

        h_t         = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        c_t         = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        h_old       = h_t.view(batch_size, 1, hidden_size)

        attn_w_viz  = []


        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            remember_size = h_old.size(1)
            input_t    = input_t.contiguous().view(input_t.size()[0], input_t.size()[-1])

            h_t, c_t   = self.lstm1(input_t, (h_t, c_t))
            h_repeated =  h_t.unsqueeze(1).repeat(1, remember_size, 1)
            mlp_h_attn = torch.cat((h_repeated, h_old), 2)

            mlp_h_attn = mlp_h_attn.view(batch_size*remember_size,hidden_size * 2)

            attn_w     = torch.mm(mlp_h_attn, self.w_t_1)

            attn_w     = self.tanh(attn_w)
            attn_w     = torch.mm(attn_w, self.w_t_2)

            #attn_w     = attn_w.unsqueeze(-1)

            h_old      = h_old.view(batch_size * remember_size, hidden_size)
            attn_w     = (attn_w.expand_as(h_old)).contiguous()
            #attn_w     = attn_w.repeat(1, 1, hidden_size)
            attn_w     = attn_w.view(batch_size *  remember_size, hidden_size)
            h_old_w    = attn_w * h_old
            h_old_w    = h_old_w.view(batch_size, remember_size, hidden_size)
            attn_c     = torch.sum(h_old_w, 1).squeeze(1)
            h_old      = h_old.view(batch_size, remember_size, hidden_size)
            h_old      = torch.cat((h_old, h_t.view(batch_size, 1, hidden_size)), dim=1)
        
        #outputs   = torch.cat  ((outputs, attn_all), 2)
        out   = h_t + attn_c
        out = self.fc(out)

        return out

    def print_log(self):
        model_name = 'LSTM_self_attention_1step_'
        model_log = ' LSTM with self attention with 1 step predication.......'
        return (model_name, model_log)




class RNN_LSTM_one_step(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_LSTM_one_step, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        h_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        c_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            input_t = input_t.contiguous().view(input_t.size()[0], input_t.size()[-1])
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
        out = self.fc(h_t)
        return out

    def print_log(self):
        model_name = '_regular-LSTM_one_step'
        model_log = ' Regular LSTM one step.......'
        return (model_name, model_log)




class RNN_LSTM_EMBED(nn.Module):

    def __init__(self, input_size,embed_size, hidden_size, num_layers, num_classes, hidden_reuse=False):
        super(RNN_LSTM_EMBED, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.embed_size = embed_size
        self.embed = nn.Embedding(num_classes, embed_size)
        self.lstm1 = nn.LSTMCell(self.embed_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes) 
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.hidden_reuse = hidden_reuse

    def forward(self, x):
        outputs = []
        x_embed = self.embed(x.view(x.size(0) * x.size(1), 1).long())
        x_embed = x_embed.view(x.size(0), x.size(1), self.embed_size)
        
        
        ht, ct = self.init_hidden_for(x)
        for i, input_t in enumerate(x_embed.chunk(x_embed.size(1), dim=1)):
            input_t  = input_t.contiguous().view(input_t.size(0), self.embed_size)
            ht, ct   = self.lstm1(input_t, (ht, ct))
            outputs += [ht]
        self.lastH = Variable(ht.data)
        self.lastC = Variable(ct.data)
        
        outputs = torch.stack(outputs, 1).squeeze(2)
        shp     = (outputs.size(0), outputs.size(1))
        out     = outputs.contiguous().view(shp[0] *shp[1] , self.hidden_size)
        out     = self.fc(out)
        out     = out.view(shp[0], shp[1], self.num_classes)

        return out

    def print_log(self):
        model_name = '_regular-LSTM_'
        model_log = ' Regular LSTM.......'
        return (model_name, model_log)
    
    def init_hidden_for(self, x):
        if(not self.hidden_reuse           or
           not hasattr(self, "lastH")      or
           not hasattr(self, "lastC")      or
           self.lastH.size(0) != x.size(0)):
            self.lastH = torch.zeros(x.size(0), self.hidden_size)
            self.lastC = torch.zeros(x.size(0), self.hidden_size)
            if x.data.is_cuda:
                self.lastH = self.lastH.cuda(x.data.get_device())
                self.lastC = self.lastC.cuda(x.data.get_device())
            self.lastH = Variable(self.lastH)
            self.lastC = Variable(self.lastC)
        
        return self.lastH, self.lastC


class RNN_LSTM_truncated_embed (nn.Module):

    def __init__(self, input_size, embed_size, hidden_size, num_layers, num_classes, truncate_length=1):
        super(RNN_LSTM_truncated_embed, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.embed_size = embed_size
        self.embed = nn.Embedding(num_classes, embed_size)
        self.truncate_length = truncate_length
        self.lstm1 = nn.LSTMCell(self.embed_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.embed.weight.data.uniform_(-0.1, 0.1)


    def forward(self, x):
        outputs = []
        h_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        c_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        x_embed = self.embed(x.view(x.size()[0] * x.size()[1], 1).long())
        x_embed = x_embed.view(x.size()[0], x.size()[1], self.embed_size)

        for i, input_t in enumerate(x_embed.chunk(x_embed.size(1), dim=1)):
            input_t = input_t.contiguous().view(input_t.size()[0], input_t.size()[-1])
            if (i  + 1) % self.truncate_length == 0 :
                h_t , c_t = self.lstm1(input_t, (h_t.detach(), c_t.detach()))
                #c_t = c_t.detach()
            else:
                h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            outputs += [h_t]
        outputs = torch.stack(outputs, 1).squeeze(2)
        shp=(outputs.size()[0], outputs.size()[1])
        out = outputs.contiguous().view(shp[0] *shp[1] , self.hidden_size)
        out = self.fc(out)
        out = out.view(shp[0], shp[1], self.num_classes)

        return out

    def print_log(self):
        model_name = '_trun-LSTM_trun_len_'+str(self.truncate_length)
        model_log = ' trun LSTM.....trun_len:'+str(self.truncate_length)
        return (model_name, model_log)








class RNN_LSTM2_truncated (nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes, truncate_length=100):
        super(RNN_LSTM_truncated, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.truncate_length = truncate_length
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        outputs = []
        h_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        c_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        h_t2 = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        c_t2 = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())

        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            input_t = input_t.contiguous().view(input_t.size()[0], 1)
            if i % self.truncate_length == 0 :  
                h_t = h_t.detach()
                c_t = c_t.detach()
                h_t2 = h_t2.detach()
                c_t2 = c_t2.detach()
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(c_t, (h_t2, c_t2))
            outputs += [h_t2]
        outputs = torch.stack(outputs, 1).squeeze(2)
        shp=(outputs.size()[0], outputs.size()[1])
        out = outputs.contiguous().view(shp[0] *shp[1] , self.hidden_size)
        out = self.fc(out)
        out = out.view(shp[0], shp[1], self.num_classes)

        return out



class RNN_LSTM_truncated (nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes, truncate_length=1):
        super(RNN_LSTM_truncated, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.truncate_length = truncate_length
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        outputs = []
        h_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        c_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())

        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            input_t = input_t.contiguous().view(input_t.size()[0], input_t.size()[-1])
            if (i  + 1) % self.truncate_length == 0 :
                h_t , c_t = self.lstm1(input_t, (h_t.detach(), c_t.detach()))
                #c_t = c_t.detach()
            else:
                h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            outputs += [h_t]
        outputs = torch.stack(outputs, 1).squeeze(2)
        shp=(outputs.size()[0], outputs.size()[1])
        out = outputs.contiguous().view(shp[0] *shp[1] , self.hidden_size)
        out = self.fc(out)
        out = out.view(shp[0], shp[1], self.num_classes)

        return out
       
    def print_log(self):
        model_name = '_trun-LSTM_trun_len_'+str(self.truncate_length)
        model_log = ' trun LSTM.....trun_len:'+str(self.truncate_length)
        return (model_name, model_log)

class RNN_LSTM_truncated_one_step (nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes, truncate_length=1):
        super(RNN_LSTM_truncated_one_step, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.truncate_length = truncate_length
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        c_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())

        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            input_t = input_t.contiguous().view(input_t.size()[0], input_t.size()[-1])
            if (i  + 1) % self.truncate_length == 0 :
                h_t , c_t = self.lstm1(input_t, (h_t.detach(), c_t.detach()))
                #c_t = c_t.detach()
            else:
                h_t, c_t = self.lstm1(input_t, (h_t, c_t))
        out = self.fc(h_t)
        return out

    def print_log(self):
        model_name = '_trun-LSTM_1step_trun_len_'+str(self.truncate_length)
        model_log = ' trun LSTM.. one step...trun_len:'+str(self.truncate_length)
        return (model_name, model_log)




class self_LSTM2(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, sliding_window=None, block_attn_grad_past=True):
        super(self_LSTM2, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.sliding_window = sliding_window
        self.block_attn_grad_past = block_attn_grad_past
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.w_t = nn.Parameter(torch.zeros(self.hidden_size * 2, 1))

    def forward(self, x):
        outputs = []
        h_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        c_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        h_t2 = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        c_t2 = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        attn_all = []
        old_h = h_t2

        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            input_t = input_t.contiguous().view(input_t.size()[0], 1)
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(c_t, (h_t2, c_t2))
            if self.sliding_window and i >= self.sliding_window:
                window_size = self.sliding_window
            else:
                window_size = i
            
            h_repeated =  h_t2.repeat(window_size + 1, 1)
            MLP_attn_input = torch.cat((h_repeated, old_h), 1)
 
            if self.block_attn_grad_past:
                MLP_attn_input = Variable(MLP_attn_input.data)

            attn_w = torch.mm(MLP_attn_input, self.w_t)
            attn_w = F.softmax(attn_w.t()).t()
            attn_c = (attn_w.repeat(1, self.hidden_size) * old_h)
            attn_c = attn_c.view(window_size+1, input_t.size()[0], self.hidden_size).sum(0).view(input_t.size()[0],self.hidden_size)

            if self.sliding_window and i >= self.sliding_window :
                old_h = torch.cat((old_h[input_t.size()[0]:, :], h_t2))
            else:
                old_h = torch.cat((old_h, h_t2))

            outputs += [h_t2]
            attn_all += [attn_c]

        outputs = torch.stack(outputs, 1).squeeze(2)
        attn_all = torch.stack(attn_all, 1).squeeze(2)
        outputs += attn_all
        shp=(outputs.size()[0], outputs.size()[1])
        out = outputs.contiguous().view(shp[0] *shp[1] , self.hidden_size)
        out = self.fc(out)
        out = out.view(shp[0], shp[1], self.num_classes)

        return out


class self_LSTM_truncated(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, truncate_length=100, block_attn_grad_past=False, attn_every_k=1):
        super(self_LSTM_truncated, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.block_attn_grad_past = block_attn_grad_past
        self.truncate_length = truncate_length
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.attn_every_k = attn_every_k
        self.w_t = nn.Parameter(torch.zeros(self.hidden_size * 2, 1))
        self.tanh = nn.Tanh()

    def forward(self, x):
        outputs = []
        h_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        c_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        attn_all = []
        old_h = h_t
        sig = torch.nn.Sigmoid()

        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            input_t = input_t.contiguous().view(input_t.size()[0], 1)
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            if (i +1) % self.truncate_length == 0:
                h_t = h_t.detach()
                c_t = c_t.detach()
            
            #was i+1
            h_repeated =  h_t.repeat(old_h.size(0)/x.size(0), 1)

            MLP_attn_input = torch.cat((h_repeated, old_h), 1)
            MLP_attn_input = self.tanh(MLP_attn_input)
            
            if self.block_attn_grad_past:
                MLP_attn_input = Variable(MLP_attn_input.data)

            attn_w = torch.mm(MLP_attn_input, self.w_t)
            attn_w = sig(attn_w)
            attn_c = (attn_w.repeat(1, self.hidden_size) * old_h)
            attn_c = attn_c.view(old_h.size(0)/x.size(0), input_t.size()[0], self.hidden_size).sum(0).view(input_t.size()[0],self.hidden_size)
            
            if i % self.attn_every_k == 0:
                old_h = torch.cat((old_h, h_t))

            outputs += [h_t]
            attn_all += [attn_c]

        outputs = torch.stack(outputs, 1).squeeze(2)
        attn_all = torch.stack(attn_all, 1).squeeze(2)
        outputs += attn_all
        shp=(outputs.size()[0], outputs.size()[1])
        out = outputs.contiguous().view(shp[0] *shp[1] , self.hidden_size)
        out = self.fc(out)
        out = out.view(shp[0], shp[1], self.num_classes)

        return out

    def print_log(self):
        model_name = '_LSTM-truncated-full-attention_truncate_length_'+str(self.truncate_length) + '_block_MLP_grad_'+str(self.block_attn_grad_past)
        model_log = ' LSTM truncated with full attention.....truncate_length:'+str(self.truncate_length)+'...block MLP grad: ' + str(self.block_attn_grad_past)
        return (model_name, model_log)




class Sback(nn.Module):
    def __init__(self, top_k = 5):
        super(Sback, self).__init__()
        self.top_k = top_k
        self.register_backward_hook(Sback.backward)

    def forward(self, attn_w,old_h, hidden_size):
        attn_w = attn_w.view(attn_w.size()[0], attn_w.size()[1], 1)
        old_h_w = (attn_w.repeat(1, 1, hidden_size) * old_h)
        out = torch.sum(old_h_w, 1).squeeze(1)
        # get rid of dim 0, reshaping this
        return out
    
    def backward(self, grad_input, grad_output):
        grad_input, = grad_input
        length = grad_input.size()[1]
        topk = min(self.top_k, length)
        grad_input_norm = (grad_input ** 2 ).sum(2)
        topk_idx = torch.topk(grad_input_norm, topk, dim = 1)[1]
        mask = torch.zeros(grad_input.size())
        mask.index_fill_(1, topk_idx.data.cpu().squeeze(1).squeeze(1), 1)
        # mask = torch.index.fill_index_copy(mask, topk_idx, 0)
        grad_input = grad_input * Variable(mask).cuda()
        return grad_input,

class Sparse_attention(nn.Module):
    def __init__(self, top_k = 5):
        super(Sparse_attention,self).__init__()
        self.top_k = top_k

    def forward(self, attn_s):

        # normalize the attention weights using piece-wise Linear function
        # only top k should
        attn_plot = []
        # torch.max() returns both value and location
        #attn_s_max = torch.max(attn_s, dim = 1)[0]
        #attn_w = torch.clamp(attn_s_max, min = 0, max = attn_s_max)
        eps = 10e-8
        time_step = attn_s.size()[1]
        if time_step <= self.top_k:
            # just make everything greater than 0, and return it
            #delta = torch.min(attn_s, dim = 1)[0]
            return attn_s
        else:
            # get top k and return it
            # bottom_k = attn_s.size()[1] - self.top_k
            # value of the top k elements 
            #delta = torch.kthvalue(attn_s, bottm_k, dim= 1 )[0]
            delta = torch.topk(attn_s, self.top_k+1, dim= 1)[0][:,-1] - eps
            #delta = attn_s_max - torch.topk(attn_s, self.top_k, dim= 1)[0][:,-1] - eps
            # normalize
        attn_w = attn_s - delta.view(-1, 1).repeat(1, time_step)
        attn_w = torch.clamp(attn_w, min = 0)
        attn_w_sum = torch.sum(attn_w, dim = 1)
        attn_w_sum = attn_w_sum + eps 
        attn_w_normalize = attn_w / attn_w_sum.view(-1, 1).repeat(1, time_step)
        return attn_w_normalize


class State_selector(nn.Module):
    def __init__ (self, hidden_size, num_states = 20, hidden_init=None ):
        super(State_selector, self).__init__()
        self.num_states     = num_states
        # previous predication error measured in L2 norm
        self.hidden_size    = hidden_size
        self.old_norms      = None
        # predicating next state
        self.tanh          = torch.nn.Tanh()
        if hidden_init is None:
            self.w1         = nn.Parameter(normal(torch.Tensor(self.hidden_size, self.hidden_size), mean = 0.0, std = 1))
        else:
            self.w1         = nn.Parameter(hidden_init.data.clone())
        self.w2            = nn.Parameter(torch.eye(self.hidden_size , self.hidden_size))

    def forward(self, old_states, new_state, prev_state):
        # decide whether or not to store the new state s
        # use prev_state to predicate new_state
        if prev_state is None:
            # store the norm of dim=1 (each mini-batch individually
            l2_norm = new_state.norm(dim=1).view(-1, 1)
            self.old_norms = l2_norm
            old_states =  new_state.view(-1, 1, self.hidden_size)
            return (old_states, l2_norm)

        pred_state = torch.mm(prev_state.detach(), self.w1)
        pred_state = self.tanh(pred_state)
        #pred_state = torch.mm(pred_state, self.w2)
        l2_norm = (pred_state - new_state.detach()).norm(dim=1).view(-1,1)
        if self.old_norms.size()[1] < self.num_states:
            self.old_norms = torch.cat((self.old_norms, l2_norm), dim=1)
            states, idx =self.old_norms.sort(dim=1) 
            self.old_norms = states
            old_states = torch.cat((old_states, new_state.view(-1, 1, self.hidden_size)), dim =1)
            shp = old_states.size()
            idx = (Variable(shp[1] * torch.arange(64).view(-1, 1).long().cuda()) + idx)
            old_states = old_states.view(-1, self.hidden_size)
            idx = idx.view(-1)
            old_states = old_states[idx]
            old_states = old_states.view(shp[0], shp[1], shp[2])
            return (old_states, l2_norm)
        else:
            # need to compare l2_norm with the smallest of the old_norms
            replace_mask = (l2_norm.view(-1) > self.old_norms[:, 0]).float()
            new_l2_norm = replace_mask * l2_norm.view(-1) + (1 - replace_mask) * self.old_norms[:, 0]
            self.old_norms = torch.cat((new_l2_norm.unsqueeze(1), self.old_norms[:,1:]), dim = 1)
            states, idx = self.old_norms.sort(dim=1)
            self.old_norms = states
            new_state = replace_mask.view(-1, 1).expand_as(new_state) * new_state + (1 - replace_mask.view(-1, 1)).expand_as(new_state) * old_states[:, 0, :]
            old_states = torch.cat((new_state.unsqueeze(1), old_states[:, 1:, :]), dim=1)
            shp = old_states.size()
            idx = (Variable(shp[1] * torch.arange(64).view(-1, 1).long().cuda()) + idx)
            old_states = old_states.view(-1, self.hidden_size)
            idx = idx.view(-1)
            old_states = old_states[idx]
            old_states = old_states.view(shp[0], shp[1], shp[2])
            return (old_states, l2_norm)





class Sparse_attention_epilson(nn.Module):
    def __init__(self, top_k = 5, epilson = 1e-4):
        super(Sparse_attention_epilson,self).__init__()
        self.top_k = top_k
        self.epilson = epilson
        #self.delta = nn.Parameter(normal(torch.Tensor(self.hidden_size * 2, 1), mean = 0.0, std = 0.01)) 

    def forward(self, attn_s):
        # normalize the attention weights using piece-wise Linear function
        # only top k should

        # torch.max() returns both value and location
        attn_s_max = torch.max(attn_s, dim = 1)[0]
        #attn_w = torch.clamp(attn_s_max, min = 0, max = attn_s_max)

        time_step = attn_s.size()[1]

        if time_step <= self.top_k:
            # just make everything greater than 0, and return it
            delta = torch.min(attn_s, dim = 1)[0]

        else:
            # get top k and return it
            # bottom_k = attn_s.size()[1] - self.top_k
            # value of the top k elements 
            #delta = torch.kthvalue(attn_s, bottm_k, dim= 1 )[0]
            delta = torch.topk(attn_s, self.top_k, dim= 1)[0][:,-1]
            # normalize
        attn_w = attn_s - attn_s_max.repeat(1, time_step) + delta.repeat(1, time_step)
        attn_w = torch.clamp(attn_w, min = 0)
        attn_w_sum = torch.sum(attn_w, dim = 1)
        eps = 10e-8
        attn_w_sum = attn_w_sum + eps
        attn_w_normalize = attn_w / attn_w_sum.repeat(1, time_step)

        return attn_w_normalize




def attention_visualize(attention_timestep, filename):
    # visualize attention
    plt.matshow(attention_timestep)
    filename += '_attention.png'
    plt.savefig(filename)
    
class self_LSTM_sparse_attn_predict(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, truncate_length=100, predict_m = 10, block_attn_grad_past=False, attn_every_k=1, top_k = 5):
        # latest sparse attentive backprop implementation
        super(self_LSTM_sparse_attn_predict, self).__init__()
        self.hidden_size          = hidden_size
        self.num_layers           = num_layers
        self.num_classes          = num_classes
        self.block_attn_grad_past = block_attn_grad_past
        self.truncate_length      = truncate_length
        self.lstm1                = nn.LSTMCell(input_size, hidden_size)
        self.fc                   = nn.Linear(hidden_size * 2, num_classes)
        self.attn_every_k         = attn_every_k
        self.top_k                = top_k
        self.tanh                 = torch.nn.Tanh()
        self.w_t                  = nn.Parameter(normal(torch.Tensor(self.hidden_size * 2, 1), mean = 0.0, std = 0.01)) #nn.Parameter(torch.zeros(self.hidden_size * 2, 1))
        self.sparse_attn          = Sparse_attention(top_k = self.top_k)
        self.predict_m            = nn.Linear(hidden_size, hidden_size)


    def print_log(self):
        model_name = '_LSTM-sparse_attn_topk_attn_in_h' + str(self.top_k) + '_truncate_length_'+str(self.truncate_length) +'attn_everyk_' + str(self.attn_every_k)# + '_block_MLP_gradient_' + str(self.block_attn_grad_past)
        model_log = ' LSTM Sparse attention in h........topk:' + str(self.top_k)  +'....attn_everyk_' + str(self.attn_every_k) + '.....truncate_length:'+str(self.truncate_length)
        return (model_name, model_log)

    def forward(self, x):
        batch_size  = x.size(0)
        time_size   = x.size(1)
        input_size  = x.size(2)
        hidden_size = self.hidden_size
        h_t         = Variable(torch.zeros(batch_size, hidden_size).cuda())
        c_t         = Variable(torch.zeros(batch_size, hidden_size).cuda())
        predict_h   = Variable(torch.zeros(batch_size, hidden_size).cuda())

        # Will eventually grow to (batch_size, time_size, hidden_size)
        # with more and more concatenations.
        h_old       = h_t.view(batch_size, 1, hidden_size)

        outputs     = []
        attn_all    = []
        attn_w_viz  = []
        predicted_all = []

        for i, input_t in enumerate(x.chunk(time_size, dim=1)):
            remember_size = h_old.size(1)
            if (i + 1) % self.truncate_length == 0:
                h_t, c_t = h_t.detach(), c_t.detach()

            # Feed LSTM Cell
            input_t    = input_t.contiguous().view(batch_size, input_size)
            h_t, c_t   = self.lstm1(input_t, (h_t, c_t))
            predict_h  = self.predict_m(h_t.detach())
            predicted_all.append(predict_h)


            # Broadcast and concatenate current hidden state against old states
            h_repeated =  h_t.unsqueeze(1).repeat(1, remember_size, 1)
            mlp_h_attn = torch.cat((h_repeated, h_old), 2)
            if self.block_attn_grad_past:
                mlp_h_attn = mlp_h_attn.detach()

            #
            # Feed the concatenation to the MLP.
            # The tensor shapes being multiplied are
            #     mlp_h_attn.size() = (batch_size, remember_size, 2*hidden_size)
            # by
            #     self.w_t.size()   = (2*hidden_size, 1)
            # Desired result is
            #     attn_w.size()     = (batch_size, remember_size, 1)
            #
            mlp_h_attn = self.tanh(mlp_h_attn)
            if False: # PyTorch 0.2.0
                attn_w     = torch.matmul(mlp_h_attn, self.w_t)
            else:     # PyTorch 0.1.12
                mlp_h_attn = mlp_h_attn.view(batch_size*remember_size, 2*hidden_size)
                attn_w     = torch.mm(mlp_h_attn, self.w_t)
                attn_w     = attn_w.view(batch_size, remember_size, 1)

            #
            # For each batch example, "select" top-k elements by sparsifying
            # attn_w.size() = (batch_size, remember_size, 1). The top k elements
            # are left non-zero and the other ones are zeroed.
            #
            attn_w  = attn_w.view(batch_size, remember_size)
            attn_w  = self.sparse_attn(attn_w)
            attn_w  = attn_w.view(batch_size, remember_size, 1)

            if i >=100:
                attn_w_viz.append(attn_w.mean(dim=0).view(remember_size))

            #
            # Broadcast the weights against the past remembered hidden states,
            # then compute the attention information attn_c.
            #
            attn_w  = attn_w.repeat(1, 1, hidden_size)
            h_old_w = attn_w * h_old
            attn_c  = torch.sum(h_old_w, 1).squeeze(1)

            # Feed attn_c to hidden state h_t
            h_t = h_t + attn_c

            #
            # At regular intervals, remember a hidden state.
            #
            if (i + 1) % self.attn_every_k == 0:
                h_old = torch.cat((h_old, h_t.view(batch_size, 1, hidden_size)), dim=1)

            # Record outputs
            outputs += [h_t]

            # For visualization purposes:
            attn_all += [attn_c]

        #
        # Compute return values. These should be:
        #     out        = (batch_size, time_size, num_classes)
        #     attn_w_viz = len([(remember_size)]) == time_size-100
        #
        predicted_all = torch.stack(predicted_all, 1)
        outputs   = torch.stack(outputs,  1)
        attn_all  = torch.stack(attn_all, 1)
        h_outs    = outputs.detach()
        outputs   = torch.cat  ((outputs, attn_all), 2)
        shp = outputs.size()
        out = outputs.contiguous().view(shp[0] *shp[1] , shp[2])
        out = self.fc(out)
        out = out.view(shp[0], shp[1], self.num_classes)

        return (out, attn_w_viz, predicted_all, h_outs)


class self_LSTM_sparse_one_step(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, truncate_length=100, block_attn_grad_past=False, print_attention_step = 1, attn_every_k=1, top_k = 5):
        # latest sparse attentive backprop implementation
        super(self_LSTM_sparse_one_step, self).__init__()
        self.hidden_size          = hidden_size
        self.num_layers           = num_layers
        self.num_classes          = num_classes
        self.block_attn_grad_past = block_attn_grad_past
        self.truncate_length      = truncate_length
        self.lstm1                = nn.LSTMCell(input_size, hidden_size)
        self.fc                   = nn.Linear(hidden_size * 2, num_classes)
        self.attn_every_k         = attn_every_k
        self.top_k                = top_k
        self.tanh                 = torch.nn.Tanh()
        #self.w_t                  = nn.Parameter(normal(torch.Tensor(self.hidden_size * 2, 1), mean = 0.0, std = 0.01)) #nn.Parameter(torch.zeros(self.hidden_size * 2, 1))
        self.w_t_1                = nn.Parameter(normal(torch.Tensor(self.hidden_size * 2, self.hidden_size * 1), mean = 0.0, std = 1)) #nn.Parameter(torch.zeros(self.hidden_size * 2, 1))
        self.w_t_2                = nn.Parameter(normal(torch.Tensor(self.hidden_size , 1), mean = 0.0, std = 0.01))
        self.sparse_attn          = Sparse_attention(top_k = self.top_k)
        self.atten_print          = print_attention_step

    def print_log(self):

        model_name = '_LSTM-sparse_1_step_attn_topk_attn_in_h' + str(self.top_k) + '_truncate_length_'+str(self.truncate_length) +'attn_everyk_' + str(self.attn_every_k) + '_block_MLP_gradient_' + str(self.block_attn_grad_past)
        model_log = ' LSTM Sparse attention 1 step in h........topk:' + str(self.top_k)  +'....attn_everyk_' + str(self.attn_every_k) + '.....truncate_length:'+str(self.truncate_length) + '_block_MLP_gradient_' + str(self.block_attn_grad_past)
        return (model_name, model_log)

    def forward(self, x):
        batch_size  = x.size(0)
        time_size   = x.size(1)
        input_size  = x.size(2)
        hidden_size = self.hidden_size
        h_t         = Variable(torch.zeros(batch_size, hidden_size).cuda())
        c_t         = Variable(torch.zeros(batch_size, hidden_size).cuda())

        # Will eventually grow to (batch_size, time_size, hidden_size)
        # with more and more concatenations.
        h_old       = h_t.view(batch_size, 1, hidden_size)

        attn_w_viz  = []

        for i, input_t in enumerate(x.chunk(time_size, dim=1)):
            remember_size = h_old.size(1)
            
            input_t    = input_t.contiguous().view(batch_size, input_size)
            if (i + 1) % self.truncate_length == 0:
                h_t, c_t = self.lstm1(input_t, (h_t.detach(), c_t.detach()))

            # Feed LSTM Cell
            else:
                h_t, c_t   = self.lstm1(input_t, (h_t, c_t))

            # Broadcast and concatenate current hidden state against old states
            h_repeated =  h_t.unsqueeze(1).repeat(1, remember_size, 1)
            mlp_h_attn = torch.cat((h_repeated, h_old), 2)
            
            if self.block_attn_grad_past:
                mlp_h_attn = mlp_h_attn.detach()

            #
            # Feed the concatenation to the MLP.
            # The tensor shapes being multiplied are
            #     mlp_h_attn.size() = (batch_size, remember_size, 2*hidden_size)
            # by
            #     self.w_t.size()   = (2*hidden_size, 1)
            # Desired result is
            #     attn_w.size()     = (batch_size, remember_size, 1)
            #
            mlp_h_attn = self.tanh(mlp_h_attn)
            if False: # PyTorch 0.2.0
                attn_w     = torch.matmul(mlp_h_attn, self.w_t)
            else:     # PyTorch 0.1.12
                mlp_h_attn = mlp_h_attn.view(batch_size*remember_size, 2*hidden_size)
                #attn_w     = torch.mm(mlp_h_attn, self.w_t)
                #attn_w     = attn_w.view(batch_size, remember_size, 1)
                attn_w     = torch.mm(mlp_h_attn, self.w_t_1)
                attn_w     = self.tanh(attn_w)
                attn_w     = torch.mm(attn_w, self.w_t_2)


            #
            # For each batch example, "select" top-k elements by sparsifying
            # attn_w.size() = (batch_size, remember_size, 1). The top k elements
            # are left non-zero and the other ones are zeroed.
            #
            attn_w  = attn_w.view(batch_size, remember_size)
            if self.top_k > 0:
                # if topk = -1, then keep all attention (dense attention)
                attn_w  = self.sparse_attn(attn_w)

            attn_w  = attn_w.view(batch_size, remember_size, 1)

            if self.atten_print > (time_size - i - 1) :
                attn_w_viz.append(attn_w.view(attn_w.size()[:-1]))
                #attn_w_viz.append(attn_w.mean(dim=0).view(remember_size))

            #
            # Broadcast the weights against the past remembered hidden states,
            # then compute the attention information attn_c.
            #
            attn_w  = attn_w.repeat(1, 1, hidden_size)
            h_old_w = attn_w * h_old
            attn_c  = torch.sum(h_old_w, 1).squeeze(1)

            # Feed attn_c to hidden state h_t
            h_t    += attn_c

            #
            # At regular intervals, remember a hidden state.
            #
            if (i + 1) % self.attn_every_k == 0:
                h_old = torch.cat((h_old, h_t.view(batch_size, 1, hidden_size)), dim=1)

            # For visualization purposes:

        #
        # Compute return values. These should be:
        #     out        = (batch_size, time_size, num_classes)
        #     attn_w_viz = len([(remember_size)]) == time_size-100
        #
        out   = torch.cat  ((h_t, attn_c), 1)
        #shp = outputs.size()
        #out = outputs.contiguous().view(shp[0] *shp[1] , shp[2])
        out = self.fc(out)
        #out = out.view(shp[0], shp[1], self.num_classes)

        return (out, attn_w_viz)
 
class self_LSTM_sparse_embed(nn.Module):
    def __init__(self, input_size,embed_size, hidden_size, num_layers, num_classes, truncate_length=100, block_attn_grad_past=False, print_attention_step = 1, attn_every_k=1, top_k = 5):
        # latest sparse attentive backprop implementation
        super(self_LSTM_sparse_embed,  self).__init__()
        self.hidden_size          = hidden_size
        self.embed_size           = embed_size
        self.num_layers           = num_layers
        self.num_classes          = num_classes
        self.block_attn_grad_past = block_attn_grad_past
        self.truncate_length      = truncate_length
        self.lstm1                = nn.LSTMCell(embed_size, hidden_size)
        self.fc                   = nn.Linear(hidden_size * 2, num_classes)
        self.attn_every_k         = attn_every_k
        self.top_k                = top_k
        self.tanh                 = torch.nn.Tanh()
        self.w_t                  = nn.Parameter(normal(torch.Tensor(self.hidden_size * 2, 1), mean = 0.0, std = 0.01)) #nn.Parameter(torch.zeros(self.hidden_size * 2, 1))
        self.sparse_attn          = Sparse_attention(top_k = self.top_k)
        self.atten_print          = print_attention_step
        
        self.embed                = nn.Embedding(num_classes, embed_size)
        
        self.embed.weight.data.uniform_(-0.1, 0.1)


    def print_log(self):
        model_name = '_LSTM-sparse_attn_topk_attn_in_h' + str(self.top_k) + '_truncate_length_'+str(self.truncate_length) +'attn_everyk_' + str(self.attn_every_k)# + '_block_MLP_gradient_' + str(self.block_attn_grad_past)
        model_log = ' LSTM Sparse attention in h........topk:' + str(self.top_k)  +'....attn_everyk_' + str(self.attn_every_k) + '.....truncate_length:'+str(self.truncate_length)
        return (model_name, model_log)

    def forward(self, x):
        batch_size  = x.size(0)
        time_size   = x.size(1)
        input_size  = x.size(2)
        hidden_size = self.hidden_size
        h_t         = Variable(torch.zeros(batch_size, hidden_size).cuda())
        c_t         = Variable(torch.zeros(batch_size, hidden_size).cuda())
        
        x_embed = self.embed(x.view(x.size()[0] * x.size()[1], 1).long())
        x_embed = x_embed.view(x.size()[0], x.size()[1], self.embed_size)
        # Will eventually grow to (batch_size, time_size, hidden_size)
        # with more and more concatenations.
        h_old       = h_t.view(batch_size, 1, hidden_size)

        outputs     = []
        attn_all    = []
        #attn_w_viz  = []
        
        for i, input_t in enumerate(x_embed.chunk(time_size, dim=1)):
            remember_size = h_old.size(1)

            input_t    = input_t.contiguous().view(batch_size, self.embed_size)
            if (i + 1) % self.truncate_length == 0:
                h_t, c_t = self.lstm1(input_t, (h_t.detach(), c_t.detach()))

            else:
                # Feed LSTM Cell
                h_t, c_t   = self.lstm1(input_t, (h_t, c_t))

            # Broadcast and concatenate current hidden state against old states
            h_repeated =  h_t.unsqueeze(1).repeat(1, remember_size, 1)
            mlp_h_attn = torch.cat((h_repeated, h_old), 2)
            if self.block_attn_grad_past:
                mlp_h_attn = mlp_h_attn.detach()

            #
            # Feed the concatenation to the MLP.
            # The tensor shapes being multiplied are
            #     mlp_h_attn.size() = (batch_size, remember_size, 2*hidden_size)
            # by
            #     self.w_t.size()   = (2*hidden_size, 1)
            # Desired result is
            #     attn_w.size()     = (batch_size, remember_size, 1)
            #
            mlp_h_attn = self.tanh(mlp_h_attn)
            if False: # PyTorch 0.2.0
                attn_w     = torch.matmul(mlp_h_attn, self.w_t)
            else:     # PyTorch 0.1.12
                mlp_h_attn = mlp_h_attn.view(batch_size*remember_size, 2*hidden_size)
                attn_w     = torch.mm(mlp_h_attn, self.w_t)
                attn_w     = attn_w.view(batch_size, remember_size, 1)

            #
            # For each batch example, "select" top-k elements by sparsifying
            # attn_w.size() = (batch_size, remember_size, 1). The top k elements
            # are left non-zero and the other ones are zeroed.
            #
            attn_w  = attn_w.view(batch_size, remember_size)
            attn_w  = self.sparse_attn(attn_w)
            attn_w  = attn_w.view(batch_size, remember_size, 1)

            #if self.atten_print >= (time_size - i - 1) :
            #    attn_w_viz.append(attn_w.mean(dim=0).view(remember_size))

            #
            # Broadcast the weights against the past remembered hidden states,
            # then compute the attention information attn_c.
            #
            attn_w  = attn_w.repeat(1, 1, hidden_size)
            h_old_w = attn_w * h_old
            attn_c  = torch.sum(h_old_w, 1).squeeze(1)

            # Feed attn_c to hidden state h_t
            h_t    += attn_c

            #
            # At regular intervals, remember a hidden state.
            #
            if (i + 1) % self.attn_every_k == 0:
                h_old = torch.cat((h_old, h_t.view(batch_size, 1, hidden_size)), dim=1)

            # Record outputs
            outputs += [h_t]
            # For visualization purposes:
            attn_all += [attn_c]

        #
        # Compute return values. These should be:
        #     out        = (batch_size, time_size, num_classes)
        #     attn_w_viz = len([(remember_size)]) == time_size-100
        #
        outputs   = torch.stack(outputs,  1)
        attn_all  = torch.stack(attn_all, 1)
        outputs   = torch.cat  ((outputs, attn_all), 2)
        shp = outputs.size()
        out = outputs.contiguous().view(shp[0] *shp[1] , shp[2])
        out = self.fc(out)
        out = out.view(shp[0], shp[1], self.num_classes)

        return out


class FAST_SAB(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_marco_states = 20, truncate_length=10, block_attn_grad_past=False, print_attention_step = 10, attn_every_k=2, top_k = 5):
        # SAB while keeping only a few states in marco state
        super(FAST_SAB, self).__init__()
        self.hidden_size          = hidden_size
        self.num_marco_states     = num_marco_states
        self.num_layers           = num_layers
        self.num_classes          = num_classes
        self.block_attn_grad_past = block_attn_grad_past
        self.truncate_length      = truncate_length
        self.lstm1                = nn.LSTMCell(input_size, hidden_size)
        self.fc                   = nn.Linear(hidden_size * 2, num_classes)
        self.attn_every_k         = attn_every_k
        self.top_k                = top_k
        self.tanh                 = torch.nn.Tanh()
        self.w_t                  = nn.Parameter(normal(torch.Tensor(self.hidden_size * 2 , self.hidden_size * 1), mean = 0.0, std = 1))
        #self.w_t_11               = nn.Parameter(normal(torch.Tensor(self.hidden_size , self.hidden_size * 1), mean = 0.0, std = 1)) #nn.Parameter(torch.zeros(self.hidden_size * 2, 1))
        #self.w_t_12               = nn.Parameter(normal(torch.Tensor(self.hidden_size, self.hidden_size * 1 ), mean = 0.0, std = 1))
        self.w_t_2                = nn.Parameter(normal(torch.Tensor(self.hidden_size , 1), mean = 0.0, std = 0.01))
        self.sparse_attn          = Sparse_attention(top_k = self.top_k)
        self.atten_print          = print_attention_step
        self.relu                 = torch.nn.ReLU()
        self.state_selector     = State_selector(hidden_size=hidden_size, num_states=num_marco_states, hidden_init=self.lstm1.weight_hh[-hidden_size:, :])

    def print_log(self):
        model_name = '_FAST_SAB_topk_' + str(self.top_k) + '_truncate_length_'+str(self.truncate_length) +'attn_everyk_' + str(self.attn_every_k) + '_block_MLP_gradient_' + str(self.block_attn_grad_past)
        model_log = ' FAST SAB....topk:' + str(self.top_k)  +'....attn_everyk_' + str(self.attn_every_k) + '.....truncate_length:'+str(self.truncate_length) + '...block_MLP_gradient...' + str(self.block_attn_grad_past)
        return (model_name, model_log)

    def forward(self, x):
        batch_size  = x.size(0)
        time_size   = x.size(1)
        input_size  = x.size(2)
        hidden_size = self.hidden_size
        h_t         = Variable(torch.zeros(batch_size, hidden_size).cuda())
        c_t         = Variable(torch.zeros(batch_size, hidden_size).cuda())
        # stores previous hidden state h_{t-1}
        h_t_1       = None

        # Will eventually grow to (batch_size, time_size, hidden_size)
        # with more and more concatenations.
        h_old       = h_t.view(batch_size, 1, hidden_size)

        outputs     = []
        attn_all    = []
        attn_w_viz  = []
        l2_norm_sum = None
        for i, input_t in enumerate(x.chunk(time_size, dim=1)):
            remember_size = h_old.size(1)

            input_t    = input_t.contiguous().view(batch_size, input_size)
            if (i + 1) % self.truncate_length == 0:
                h_t, c_t = self.lstm1(input_t, (h_t.detach(), c_t.detach()))

            else:
                # Feed LSTM Cell
                h_t, c_t   = self.lstm1(input_t, (h_t, c_t))

            # Broadcast and concatenate current hidden state against old states
            h_repeated =  h_t.unsqueeze(1).repeat(1, remember_size, 1)
            mlp_h_attn = torch.cat((h_repeated, h_old), 2)

            if self.block_attn_grad_past:
                mlp_h_attn = mlp_h_attn.detach()

            if False: # PyTorch 0.2.0
                attn_w     = torch.matmul(mlp_h_attn, self.w_t)
            else:     # PyTorch 0.1.12
                
                # attention part changed to concat -- > linear --> Relu -- > linear -- > Tanh
                mlp_h_attn = mlp_h_attn.view(batch_size*remember_size, 2*hidden_size)
                attn_w       = torch.mm(mlp_h_attn, self.w_t)
                attn_w     = self.tanh(attn_w)
                attn_w     = torch.mm(attn_w, self.w_t_2)
                attn_w     = self.tanh(attn_w)
                attn_w     = attn_w.view(batch_size, remember_size, 1)
            
            # For each batch example, "select" top-k elements by sparsifying
            # attn_w.size() = (batch_size, remember_size, 1). The top k elements
            # are left non-zero and the other ones are zeroed.
            #
            attn_w  = attn_w.view(batch_size, remember_size)
            attn_w  = self.sparse_attn(attn_w)
            attn_w  = attn_w.view(batch_size, remember_size, 1)

            if self.atten_print > (time_size - i - 1) :
                attn_w_viz.append(attn_w.mean(dim=0).view(remember_size))

            #
            # Broadcast the weights against the past remembered hidden states,
            # then compute the attention information attn_c.
            #
            attn_w  = attn_w.repeat(1, 1, hidden_size)
            h_old_w = attn_w * h_old
            attn_c  = torch.sum(h_old_w, 1).squeeze(1)

            # Feed attn_c to hidden state h_t
            h_t    += attn_c
            
            #
            # decide whether or not to store h_old
            #
            #h_old = torch.cat((h_old, h_t.view(batch_size, 1, hidden_size)), dim=1)
            h_old, l2_norm = self.state_selector(h_old, h_t, h_t_1)
            if l2_norm_sum is None:
                l2_norm_sum = l2_norm.mean()
            else:
                l2_norm_sum += l2_norm.mean()

            h_t_1 = h_t
            # Record outputs
            outputs += [h_t]

            # For visualization purposes:
            attn_all += [attn_c]

        #
        # Compute return values. These should be:
        #     out        = (batch_size, time_size, num_classes)
        #     attn_w_viz = len([(remember_size)]) == time_size-100
        #
        outputs   = torch.stack(outputs,  1)
        attn_all  = torch.stack(attn_all, 1)
        outputs   = torch.cat  ((outputs, attn_all), 2)
        shp = outputs.size()
        out = outputs.contiguous().view(shp[0] *shp[1] , shp[2])
        out = self.fc(out)
        out = out.view(shp[0], shp[1], self.num_classes)

        return (out, attn_w_viz, l2_norm_sum/ i)





class self_LSTM_sparse_attn(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, truncate_length=10, block_attn_grad_past=False, print_attention_step = 10, attn_every_k=2, top_k = 5):
        # latest sparse attentive backprop implementation
        super(self_LSTM_sparse_attn, self).__init__()
        self.hidden_size          = hidden_size
        self.num_layers           = num_layers
        self.num_classes          = num_classes
        self.block_attn_grad_past = block_attn_grad_past
        self.truncate_length      = truncate_length
        self.lstm1                = nn.LSTMCell(input_size, hidden_size)
        self.fc                   = nn.Linear(hidden_size * 2, num_classes)
        self.attn_every_k         = attn_every_k
        self.top_k                = top_k
        self.tanh                 = torch.nn.Tanh()
        self.w_t               = nn.Parameter(normal(torch.Tensor(self.hidden_size * 2 , self.hidden_size * 1), mean = 0.0, std = 1))
        #self.w_t_11               = nn.Parameter(normal(torch.Tensor(self.hidden_size , self.hidden_size * 1), mean = 0.0, std = 1)) #nn.Parameter(torch.zeros(self.hidden_size * 2, 1))
        #self.w_t_12               = nn.Parameter(normal(torch.Tensor(self.hidden_size, self.hidden_size * 1 ), mean = 0.0, std = 1))
        self.w_t_2                = nn.Parameter(normal(torch.Tensor(self.hidden_size , 1), mean = 0.0, std = 0.01))
        self.sparse_attn          = Sparse_attention(top_k = self.top_k)
        self.atten_print          = print_attention_step
        self.relu                 = torch.nn.ReLU()

    def print_log(self):
        model_name = '_LSTM-sparse_attn_topk_attn_in_h' + str(self.top_k) + '_truncate_length_'+str(self.truncate_length) +'attn_everyk_' + str(self.attn_every_k) + '_block_MLP_gradient_' + str(self.block_attn_grad_past)
        model_log = ' LSTM Sparse attention in h........topk:' + str(self.top_k)  +'....attn_everyk_' + str(self.attn_every_k) + '.....truncate_length:'+str(self.truncate_length) + '...block_MLP_gradient...' + str(self.block_attn_grad_past)
        return (model_name, model_log)

    def forward(self, x):
        batch_size  = x.size(0)
        time_size   = x.size(1)
        input_size  = x.size(2)
        hidden_size = self.hidden_size
        h_t         = Variable(torch.zeros(batch_size, hidden_size).cuda())
        c_t         = Variable(torch.zeros(batch_size, hidden_size).cuda())

        # Will eventually grow to (batch_size, time_size, hidden_size)
        # with more and more concatenations.
        h_old       = h_t.view(batch_size, 1, -1)
        
        outputs     = []
        attn_all    = []
        attn_w_viz  = []

        for i, input_t in enumerate(x.chunk(time_size, dim=1)):
            remember_size = h_old.size(1)
            
            input_t    = input_t.contiguous().view(batch_size, input_size)
            if (i + 1) % self.truncate_length == 0:
                h_t, c_t = self.lstm1(input_t, (h_t.detach(), c_t.detach()))
            
            else:
                # Feed LSTM Cell
                h_t, c_t   = self.lstm1(input_t, (h_t, c_t))
            
            # Broadcast and concatenate current hidden state against old states
            h_repeated =  h_t.unsqueeze(1).repeat(1, remember_size, 1)
            mlp_h_attn = torch.cat((h_repeated, h_old), 2)
            
            if self.block_attn_grad_past:
                mlp_h_attn = mlp_h_attn.detach()
            
            # Feed the concatenation to the MLP.
            # The tensor shapes being multiplied are
            #     mlp_h_attn.size() = (batch_size, remember_size, 2*hidden_size)
            # by
            #     self.w_t.size()   = (2*hidden_size, 1)
            # Desired result is
            #     attn_w.size()     = (batch_size, remember_size, 1)
            #
            
            
            if False: # PyTorch 0.2.0
                attn_w     = torch.matmul(mlp_h_attn, self.w_t)
            else:     # PyTorch 0.1.12
                # attention part changed to concat -- > linear --> Relu -- > linear -- > Tanh
                mlp_h_attn = mlp_h_attn.view(batch_size*remember_size, 2*hidden_size)
                attn_w       = torch.mm(mlp_h_attn, self.w_t)    
                
                # TODO: if we want to split the weight matrix
                #attn_w1     = torch.mm(h_old.view(-1, hidden_size), self.w_t_11)
                #attn_w2     = torch.mm(h_t, self.w_t_12)
                #attn_w      = attn_w1.view(remember_size,batch_size, hidden_size) + attn_w2
                #attn_w     = attn_w.view(-1, hidden_size) 
                attn_w     = self.tanh(attn_w)
                attn_w     = torch.mm(attn_w, self.w_t_2) 
                #attn_w     = self.tanh(attn_w)
                attn_w     = attn_w.view(batch_size, remember_size, 1)
            #
            # For each batch example, "select" top-k elements by sparsifying
            # attn_w.size() = (batch_size, remember_size, 1). The top k elements
            # are left non-zero and the other ones are zeroed.
            #
            attn_w  = attn_w.view(batch_size, remember_size)
            if self.top_k > 0:
                # if topk = -1, then keep all attention (dense attention)
                attn_w  = self.sparse_attn(attn_w)

            attn_w  = attn_w.view(batch_size, remember_size, 1)
            
            if self.atten_print > (time_size - i - 1) :
                attn_w_viz.append(attn_w.mean(dim=0).view(remember_size))

            #
            # Broadcast the weights against the past remembered hidden states,
            # then compute the attention information attn_c.
            #
            attn_w  = attn_w.repeat(1, 1, hidden_size)
            h_old_w = attn_w * h_old
            attn_c  = torch.sum(h_old_w, 1).squeeze(1)
            
            # Feed attn_c to hidden state h_t
            h_t    += attn_c

            #
            # At regular intervals, remember a hidden state.
            #
            if (i + 1) % self.attn_every_k == 0:
                h_old = torch.cat((h_old, h_t.view(batch_size, 1, hidden_size)), dim=1)
            # Record outputs
            outputs += [h_t]

            # For visualization purposes:
            attn_all += [attn_c]

        #
        # Compute return values. These should be:
        #     out        = (batch_size, time_size, num_classes)
        #     attn_w_viz = len([(remember_size)]) == time_size-100
        #
        outputs   = torch.stack(outputs,  1)
        attn_all  = torch.stack(attn_all, 1)
        outputs   = torch.cat  ((outputs, attn_all), 2)
        shp = outputs.size()
        out = outputs.contiguous().view(shp[0] *shp[1] , shp[2])
        out = self.fc(out)
        out = out.view(shp[0], shp[1], self.num_classes)

        return (out, attn_w_viz)



class self_LSTM_sback(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, truncate_length=100, block_attn_grad_past=False, attn_every_k=1, top_k = 5):
        super(self_LSTM_sback, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.block_attn_grad_past = block_attn_grad_past
        self.truncate_length = truncate_length
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.attn_every_k = attn_every_k
        self.top_k = top_k
        self.sback = Sback()
        self.tanh = torch.nn.Tanh()
        self.w_t = nn.Parameter(torch.zeros(self.hidden_size * 2, 1)) 


    def print_log(self):
        model_name = '_LSTM-SAB_topk_attn_in_h' + str(self.top_k) + '_truncate_length_'+str(self.truncate_length) +'attn_everyk_' + str(self.attn_every_k)# + '_block_MLP_gradient_' + str(self.block_attn_grad_past)
        model_log = ' LSTM SAB attention in h........topk:' + str(self.top_k)  +'....attn_everyk_' + str(self.attn_every_k) + '.....truncate_length:'+str(self.truncate_length)
        return (model_name, model_log)


    def forward(self, x):
        outputs = []
        h_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        c_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        attn_all = []
        old_h = h_t
        batch_size = x.size()[0]

        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            input_t = input_t.contiguous().view(input_t.size()[0], 1)
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            if (i +1) % self.truncate_length == 0:
                h_t = h_t.detach()
                c_t = c_t.detach()

            # was i+1
            # should not do this for timestep 0
            h_repeated =  h_t.repeat(old_h.size(0)/x.size(0), 1)

            MLP_attn_input = torch.cat((h_repeated, old_h), 1)

            if self.block_attn_grad_past:
                MLP_attn_input = MLP_attn_input.detach()


            MLP_attn_input = self.tanh(MLP_attn_input)
            attn_w = torch.mm(MLP_attn_input, self.w_t)
            
            attn_w = attn_w.view(batch_size, old_h.size(0)/x.size(0))
            attn_w_normal = F.softmax(attn_w)
            
            attn_c = self.sback(attn_w_normal, old_h.view(batch_size, old_h.size(0)/x.size(0) , self.hidden_size), self.hidden_size) 
            
            # also feed attn_c to hidden state h_t
            h_t = h_t + attn_c

            # passing to sback tensor in batch_size * time_step, hidden_size
            
            #attn_c = (attn_w_normal.repeat(1, self.hidden_size) * old_h)
            #attn_c = attn_c.view(old_h.size(0)/x.size(0), input_t.size()[0], self.hidden_size).sum(0).view(input_t.size()[0],self.hidden_size)

            if (i + 1) % self.attn_every_k == 0:
                old_h = torch.cat((old_h, h_t))
            outputs += [h_t]

            attn_all += [attn_c]
        
        outputs = torch.stack(outputs, 1).squeeze(2)
        attn_all = torch.stack(attn_all, 1).squeeze(2)
        outputs += attn_all
        shp=(outputs.size()[0], outputs.size()[1])
        out = outputs.contiguous().view(shp[0] *shp[1] , self.hidden_size)
        out = self.fc(out)
        out = out.view(shp[0], shp[1], self.num_classes)

        return out


'''
class condBN_LSTM_sback(nn.Module):
    # SAB LSTM with conditional batchnorm
    def __init__(self, input_size, hidden_size, num_layers, num_classes, truncate_length=100, block_attn_grad_past=False, attn_every_k=1, top_k = 5):
        super(condBN_LSTM_sback, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.block_attn_grad_past = block_attn_grad_past
        self.truncate_length = truncate_length
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.attn_every_k = attn_every_k
        self.top_k = top_k
        self.sback = Sback()
        self.w_t = nn.Parameter(torch.zeros(self.hidden_size * 2, 1))
        # affine trainsform for learning beta and gamma (weight and bias for conditional BN)
        self.w_condBN_beta = nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size))
        self.w_condBN_gamma = nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size))
        
        # running mean and variance for conditional BN
        self.register_buffer('running_mean', torch.zeros(self.hidden_size))
        self.register_buffer('running_var', torch.ones(self.hidden_size))
        
        self.epilson = 1e-6
       
    def forward(self, x):
        outputs = []
        h_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        c_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        attn_all = []
        old_h = h_t
        batch_size = x.size()[0]

        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            input_t = input_t.contiguous().view(input_t.size()[0], 1)
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            if (i +1) % self.truncate_length == 0:
                h_t = h_t.detach()
                c_t = c_t.detach()

            # was i+1
            # should not do this for timestep 0
-            h_repeated =  h_t.repeat(old_h.size(0)/x.size(0), 1)
-
-            MLP_attn_input = torch.cat((h_repeated, old_h), 1)
-
-            if self.block_attn_grad_past:
-                MLP_attn_input = MLP_attn_input.detach()
-
-            attn_w = torch.mm(MLP_attn_input, self.w_t)
-            # apply conditional BN
-            #attn_w = F.batch_norm(affine)
-            
-
-            # compute the beta and gamma for conditional batchnorm
-            condBN_beta = torch.mm(h_t, self.w_condBN_beta)
-            condBN_gamma = torch.mm(h_t, self.w_condBN_gamma)
-            
-            mean = torch.mean(attn_w)
-            var = torch.var(attn_w - mean)
-
-            self.running_mean = 0.9 * self.running_mean  + 0.1 * mean
-            self.running_var = 0.9 * self.running_var + 0.1 * var
-
-            attn_w  = condBN_gamma * (attn_w - mean) / (torch.nn.sqrt(var) + self.epilson) + condBN_beta
-            
-            attn_w = attn_w.view(batch_size, old_h.size(0)/x.size(0))
-            attn_w_normal = F.softmax(attn_w)
-



            attn_c = self.sback(attn_w_normal, old_h.view(batch_size, old_h.size(0)/x.size(0) , self.hidden_size), self.hidden_size)
            # passing to sback tensor in batch_size * time_step, hidden_size

            #attn_c = (attn_w_normal.repeat(1, self.hidden_size) * old_h)
            #attn_c = attn_c.view(old_h.size(0)/x.size(0), input_t.size()[0], self.hidden_size).sum(0).view(input_t.size()[0],self.hidden_size)

            if i % self.attn_every_k == 0:
                old_h = torch.cat((old_h, h_t))
            outputs += [h_t]

            attn_all += [attn_c]

        outputs = torch.stack(outputs, 1).squeeze(2)
        attn_all = torch.stack(attn_all, 1).squeeze(2)
        outputs += attn_all
        shp=(outputs.size()[0], outputs.size()[1])
        out = outputs.contiguous().view(shp[0] *shp[1] , self.hidden_size)
        out = self.fc(out)
        out = out.view(shp[0], shp[1], self.num_classes)

        return out
'''


class self_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(self_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.w_t = nn.Parameter(torch.zeros(self.hidden_size * 2, 1))

    def forward(self, x):
        outputs = []
        h_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        c_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        attn_all = []
        old_h = h_t

        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            input_t = input_t.contiguous().view(input_t.size()[0], 1)
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_repeated =  h_t.repeat(i + 1, 1)
            MLP_attn_input = torch.cat((h_repeated, old_h), 1)
            attn_w = torch.mm(MLP_attn_input, self.w_t)
            attn_c = (attn_w.repeat(1, self.hidden_size) * old_h)
            attn_c = attn_c.view(i+1, input_t.size()[0], self.hidden_size).sum(0).view(input_t.size()[0],self.hidden_size)
            old_h = torch.cat((old_h, h_t))
            outputs += [h_t]
            attn_all += [attn_c]

        outputs = torch.stack(outputs, 1).squeeze(2)
        attn_all = torch.stack(attn_all, 1).squeeze(2)
        outputs += attn_all
        shp=(outputs.size()[0], outputs.size()[1])
        out = outputs.contiguous().view(shp[0] *shp[1] , self.hidden_size)
        out = self.fc(out)
        out = out.view(shp[0], shp[1], self.num_classes)

        return out



























