from torch.autograd import Variable
import torch
import tqdm
import numpy as np
device = torch.device('cuda')
def get_vHv(model, direc, dataset, train_size, batch_size, args, criterion, T, N=1, M=15, seed=777):
    
    model.train()
    v = direc # Variable(torch.from_numpy( direc.astype("float32")).cuda())
    hid_size = args.lstm_size
    flat_grad_loss = None
    flat_Hv = None
    t_ind = 0

    train_x, train_y = dataset
    for z in range(int(0.1*train_size // batch_size)):
        t_ind += 1
        ind = np.random.choice(train_size, batch_size)
        inp_x, inp_y = train_x[ind], train_y[ind]
        inp_x.transpose_(0, 1)
        inp_y.transpose_(0, 1)
        h = torch.zeros(batch_size, hid_size).to(device)
        c = torch.zeros(batch_size, hid_size).to(device)

        sq_len = T + 20
        loss = 0

        p_full = args.p_full
        val = np.random.random(size=1)[0]
        # 0.8 0.6 0.4 0.2
        for i in range(sq_len):
            output, (h, c) = model(inp_x[i], (h, c))
            loss += criterion(output, inp_y[i].squeeze(1))

        loss /= (1.0 * sq_len)

        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        flat_grad_loss = torch.cat([grad.view(-1) for grad in grads])


        grad_dot_v = (flat_grad_loss * (v)).sum()

        Hv = torch.autograd.grad(grad_dot_v, model.parameters())
        if flat_Hv is None:
            flat_Hv = torch.cat([grad.contiguous().view(-1) for grad in Hv])
        else:
            flat_Hv.data.add_(torch.cat([grad.contiguous().view(-1) for grad in Hv]).data)

    flat_Hv.data.mul_(1./t_ind)

    return torch.sum(v*flat_Hv).item()
