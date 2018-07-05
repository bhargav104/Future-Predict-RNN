#!/usr/bin/env python
#
# Imports.
#
# We import matplotlib because it is critical to invoke .use('Agg') before any
# other use of matplotlib happens. Otherwise we proceed from the public to the
# local, from general to specific, and alphabetically.
#
import matplotlib
matplotlib.use('Agg')
import argparse                     as Ap
import ipdb
import matplotlib.pyplot            as plt
import matplotlib.backends.backend_pdf
import numpy
import os, random, sys, time
import torch
import torch.nn                     as nn
import torchvision.transforms       as transforms
import torchvision.datasets         as dsets

#from   pysnips.ml.experiment    import *
#from   pysnips.ml.eventlogger   import *
from   torch.autograd           import Variable

from   layers_torch             import *

#
# For convenience, abbreviate breakpoint to bp().
#
bp=ipdb.set_trace


#
# Define and formally parse the arguments early on.
#
argp = Ap.ArgumentParser()
argp.add_argument("-s", "--seed",           default=0x6a09e667f3bcc908, type=int,
    help="Seed for PRNGs. Default is 64-bit fractional expansion of sqrt(2).")
argp.add_argument("--model",                default="SAB",       type=str,
    choices=["SAB", "trunc", "baseline", "attention_baseline"],
    help="Model Selection.")
argp.add_argument("-n", "--num-epochs",     default=40,                 type=int,
    help="Number of epochs")
argp.add_argument("--bs",                   default=64,                 type=int,
    help="Training Batch Size")
argp.add_argument("--vbs",                  default=64,                 type=int,
    help="Validation Batch Size")
argp.add_argument("--rnn-dim",              default=128,                type=int,
    help="RNN hidden state size")
argp.add_argument("--rnn-layers",           default=1,                  type=int,
    help="Number of RNN layers")
argp.add_argument("--attk",                 default=2,                  type=int,
    help="Attend every K timesteps")
argp.add_argument("--topk",                 default=10,                 type=int,
    help="Attend only to the top K most important timesteps.")
argp.add_argument("--trunc",                default=10,                 type=int,
    help="Truncation length")
argp.add_argument("-T",                     default=200,                type=int,
    help="Copy Distance")
argp.add_argument("--clipnorm", "--cn",     default=1.0,              type=float,
    help="The norm of the gradient will be clipped at this magnitude.")
argp.add_argument("--lr",                   default=1e-3,               type=float,
    help="Learning rate")
argp.add_argument("--cuda",                 default=None,               type=int,
    nargs="?", const=0,
    help="CUDA device to use.")
argp.add_argument("--reload",               action="store_true",
    help="Whether to reload the network or not.")

argp.add_argument('--block-grad',           default=False,              type=bool,
    help="whether to block gradients for mental updates")

d = argp.parse_args(sys.argv[1:])



#
# Define what an experiment is
#
class SABExperiment:
    """
    Sparse Attentive Backtracking Experiment
    
    Manages snapshot & rollback of experiments.
    """
    
    def __init__      (self, d):
        #
        # Because of unfortunate code design, we must complete most of the
        # initialization without knowing the working directory.
        # As a result, we cannot invoke the superconstructor of this experiment
        # until quite late, and until we do we do *NOT* have access to its
        # APIs, so we must be careful to avoid them.
        #
        
        #
        # Consume the arguments
        #
        self.d                = d
        self.input_size       = 2
        self.rnn_dim          = self.d.rnn_dim
        self.num_layers       = self.d.rnn_layers
        self.num_classes      = 1
        self.batch_size       = self.d.bs
        self.valid_batch_size = self.d.vbs
        self.num_epochs       = self.d.num_epochs
        self.lr               = self.d.lr
        self.n_words          = 2
        self.maxlen           = 785
        self.dictionary       = 'dict_bin_mnist.npz'
        self.truncate_length  = self.d.trunc
        self.T                = self.d.T
        self.n_train          = 5000 * 128 // self.batch_size
        self.n_test           = 2056 // self.batch_size
        self.n_sequence       = 10
        self.attn_every_k     = self.d.attk
        self.re_load          = self.d.reload
        self.top_k            = self.d.topk
        self.clip_norm        = self.d.clipnorm
        self.hist_valid_loss  = 1.0
        self.block_grad       = self.d.block_grad
        
        #
        # PRNG seeding.
        #
        numpy.random.normal(self.d.seed & 0xFFFFFFFF)
        torch.manual_seed  (self.d.seed & 0xFFFFFFFF)
        if self.d.cuda is not None:
            torch.cuda.manual_seed_all(self.d.seed & 0xFFFFFFFF)
        
        #
        # Generate training data.
        #
        self.train_x, self.train_y = adding_data(self.T, self.batch_size, self.n_train)
        self.test_x,  self.test_y  = adding_data(self.T, self.batch_size, self.n_test)
        
        #
        # Create the RNN model, then place it on CPU or GPU
        #
        # NB: Actually, currently it can only be placed on GPU.
        #
        self.epoch      = 0
        self.globalStep = 0
        if   self.d.model == "SAB":
            self.rnn = self_LSTM_sparse_one_step    (self.input_size, self.rnn_dim, self.num_layers, self.num_classes, truncate_length=self.truncate_length, top_k=self.top_k ,attn_every_k=self.attn_every_k , block_attn_grad_past=self.block_grad)
        elif self.d.model == "trunc":
            self.rnn = RNN_LSTM_truncated_one_step  (self.input_size, self.rnn_dim, self.num_layers, self.num_classes, truncate_length=self.truncate_length)
        elif self.d.model == "baseline":
            self.rnn = RNN_LSTM_one_step            (self.input_size, self.rnn_dim, self.num_layers, self.num_classes)
        elif self.d.model == "attention_baseline":
            self.rnn = RNN_LSTM_attention_one_step            (self.input_size, self.rnn_dim, self.num_layers, self.num_classes)
        if self.d.cuda is None:
            self.rnn.cuda()
            """self.rnn.cpu() # FIXME: Some day this should be uncommented!"""
        else:
            self.rnn.cuda(self.d.cuda)
        
        #
        # Create the loss metrics and optimizers.
        #
        self.criterion = nn.MSELoss()
        self.opt       = torch.optim.Adam(self.rnn.parameters(), lr=self.lr)
        
        #
        # Determine our working directory by asking the model what it should be.
        #
        # There is a major difference here from Rosemary's code in that we
        # DO NOT generate this name semi-randomly, or else the snapshot & reload
        # functionality would be useless. Instead we base it on the PRNG *seed*.
        #
        self.model_id, self.model_log, = self.rnn.print_log()
        self.model_log += " clip norm " + str(self.clip_norm)
        self.model_id  = "T_{:d}{:s}_rnn_dim_{:d}_seed_{:d}".format(
            self.T,
            self.model_id,
            self.rnn_dim,
            self.d.seed,
        )
        self.folder_id = os.path.join('adding_logs', self.model_id)
        
        #
        # SUPER()-CALL. INITIALIZATION OF EXPERIMENT.
        #
        #super(SABExperiment, self).__init__(self.folder_id)
        self.mkdirp(self.logDir)
        self.file_name       = os.path.join(self.logDir, self.model_id+'.txt')
        self.model_file_name = os.path.join(self.logDir, self.model_id+'.pkl')
        self.attn_file       = os.path.join(self.logDir, self.model_id+'.npz')
        
        #
        # Begin log with message.
        #
        log_  = ''
        log_ += 'Invocation:          '+' '.join(sys.argv)+'\n'
        log_ += 'Timestamp:           '+time.asctime()+'\n'
        log_ += 'SLURM Job Id:        '+str(os.environ.get('SLURM_JOBID', '-'))+'\n'
        log_ += 'Working Directory:   '+self.workDir+'\n'
        log_ += 'Start training ...T: ' +  str(self.T) + '...' + self.model_log +'....learning rate: ' + str(self.lr)+'\n'
        sys.stdout.write(log_)
        sys.stdout.flush()
        with open(self.file_name, 'a') as f:
            f.write(log_)
            f.flush()
    
    @property
    def logDir        (self):
        return os.path.join(self.workDir, "logs")
    
    def load          (self, path):
        """
        Load parameters, ... from disk but do not do any heavy computation.
        """
        
        snapFile = os.path.join(path, "snapshot.pkl")
        (
            self.rnn,
            self.opt,
            self.epoch,
            self.globalStep,
            self.hist_valid_loss,
        ) = torch.load(snapFile)
        return self
    
    def dump          (self, path):
        """
        Store parameters, ... to disk but do not do any heavy computation.
        """
        
        snapFile = os.path.join(path, "snapshot.pkl")
        torch.save((
            self.rnn,
            self.opt,
            self.epoch,
            self.globalStep,
            self.hist_valid_loss,
        ), snapFile)
        return self
    
    def fromScratch   (self):
        """
        Since everything was done in the constructor, there is nothing to
        really do here.
        """
        return self
    
    def fromSnapshot  (self, path):
        """
        Reload from checkpoint. Will likely involve self.load(path) but may
        also include some processing.
        """
        return self.load(path)
    
    def run           (self):
        """
        Run the entire experiment under TensorBoard logging, snapshotting
        after each epoch.
        """
        with EventLogger(self.logDir, self.globalStep, flushSecs=5.0) as e:
            while self.epoch < self.num_epochs:
                self.runEpoch().snapshot().purge()
                e.flush()
    
    def runEpoch      (self):
        """
        Run one epoch of training.
        """
        
        """
        TRAINING TIME
        """
        self.rnn.train()
        with tagscope("train"):
            for i, (x, y) in enumerate(zip(self.train_x, self.train_y)):
                """
                Pre-step the event logger every training iteration.
                A lot of thinking has gone into why this must be here.
                Leave this code where it is.
                """
                getEventLogger().step()
                
                """Start the clock."""
                t = -time.time()
                
                """Load a training data batch."""
                x = numpy.asarray(x, dtype=numpy.float32)
                x = torch.from_numpy(x)
                y = numpy.asarray(y, dtype=numpy.float32)
                y = torch.from_numpy(y)
                images = Variable(x).cuda()
                labels = Variable(y).cuda()
                
                """Run forward pass, backward pass and optimizer update."""
                self.opt.zero_grad()
                if   self.d.model == "SAB":
                    outputs, attn_w_viz                      = self.rnn(images)
                else:
                    outputs                                  = self.rnn(images)
                shp           = outputs.size()
                outputs_last  = outputs
                labels_reshp  = labels.view(labels.size()[0], 1)
                loss          = self.criterion(outputs_last,  labels_reshp)
                if   self.d.model == "sparseattn_predict":
                    predict_loss  = ((predicted_h[:, : -predict_m,:] - real_h[:,predict_m :,:].clone() ) ** 2).mean()
                    loss         += self.beta * predict_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.rnn.parameters(), self.clip_norm)
                self.opt.step()
                
                """Stop the clock."""
                t += time.time()
                
                """Log metrics to TensorBoard."""
                cpuLoss = float(loss.data.cpu()[0])
                logScalar("loss",      cpuLoss, displayName="L2 Loss",        description="L2 distance of output from target sum.")
                logScalar("batchTime", t,       displayName="Batch Time (s)", description="Time taken for a forward & backward pass and an update, in seconds.")
                
                """Every 10 steps, record attention weights?"""
                if ((i+1) % 10 == 0) and self.d.model == "SAB":
                    max_len = attn_w_viz[-1].cpu().data.numpy().shape[0]
                    attn_all = []
                    for attn in attn_w_viz:
                        attn = attn.cpu().data.numpy()
                        attn = numpy.append(attn, numpy.zeros(max_len - len(attn)))
                        attn_all.append(attn)
                    with open(self.file_name, 'a') as f:
                        for item in attn_all:
                            #print ("%s\n" % item)
                            f.write("%s\n" % item)
                
                """Every 50 steps, write step #, loss, batch time to log and stdout."""
                if (i+1) % 50 == 0:
                    log_line = self.model_log + ' Epoch [%d/%d],  T :  %d, Step %d  Loss: %.4E, batch_time: %f\n' %(self.epoch, self.num_epochs, self.d.T, i+1, loss.data[0], t)
                    sys.stdout.write(log_line)
                    sys.stdout.flush()
                    with open(self.file_name, 'a') as f:
                        f.write(log_line)
                        f.flush()
                
                """Every 10 timesteps, evaluate validation performance."""
                if (i+1) % 500 == 0:
                    #if self.d.model == "sparseattn" :
                    self.evaluate_valid(self.test_x, self.test_y)
        
        """
        VALIDATION TIME
        """
        self.rnn.eval()
        with tagscope("valid"):
            sys.stdout.write('--- Epoch finished ----\n')
            sys.stdout.flush()
            self.evaluate_valid(self.test_x, self.test_y)
        
        """Move to next epoch"""
        self.epoch += 1
        
        return self
    
    def evaluate_valid(self, valid_x, valid_y):
        """
        Perform validation testing over provided dataset.
        """
        
        valid_loss      = []
        for i, (x, y) in enumerate(zip(valid_x, valid_y)):
            """Load a validation data batch."""
            x = numpy.asarray(x, dtype=numpy.float32)
            x = torch.from_numpy(x)
            y = numpy.asarray(y, dtype=numpy.float32)
            y = torch.from_numpy(y)
            images = Variable(x, volatile=True).cuda()
            labels = Variable(y, volatile=True).cuda()
            
            
            """Run forward pass only."""
            if   self.d.model == "SAB":
                outputs, attn_w_viz                      = self.rnn(images)
                attn_viz_file = 'iter_' + str(i)
                #self.attention_viz(attn_w_viz, attn_viz_file, x, y)
            elif self.d.model == "sparseattn_predict":
                outputs, attn_w_viz, predicted_h, real_h = self.rnn(images)
            else:
                outputs                                  = self.rnn(images)
            
            """Collect numbers required to calculate accuracy & loss statistics"""
            shp           = outputs.size()
            outputs_last  = outputs
            labels_reshp  = labels.view(labels.size()[0], 1).float()
            loss          = self.criterion(outputs_last, labels_reshp)
            valid_loss.append(float(loss.data[0]))
        
        
        """Actually compute the accuracy and losses."""
        avg_valid_loss = numpy.asarray(valid_loss     ).mean()
        
        """Print them out to several places"""
        log_line = self.model_log+' adding task  rnn dim '+str(self.rnn_dim)+' Epoch [%d/%d]  average Loss: %.4E,  validation ' % (
            self.epoch,
            self.num_epochs,
            avg_valid_loss,
        ) + '\n'
        
        sys.stdout.write(log_line)
        sys.stdout.flush()
        with open(self.file_name, 'a') as f:
            f.write(log_line)
            f.flush()
        
        logScalar("loss", avg_valid_loss, displayName="L2 Loss", description="L2 distance of output from target sum.")
        
        
        """If they are the best so far, save them."""
        if avg_valid_loss < self.hist_valid_loss:
            self.hist_valid_loss = avg_valid_loss
            save_param(self.rnn, self.model_file_name)
        
        return self
    
    def attention_viz (self, attention_timestep, filename, test_x, test_y):
        # visualize attention
        max_len = attention_timestep[-1].cpu().data.numpy().shape[0]
        attn_all = []
        attn_batch = attention_timestep[-1].cpu().data.numpy()
        test_x = test_x.numpy()
        test_y = test_y.numpy()
        all_ = []
        for i in range(64):
            temp = {}
            attn = attn_batch[i].reshape(1, 40)
            x = test_x[i]
            y = test_y[i]
            temp['attn'] = attn.tolist()
            temp['x'] = x.tolist()
            temp['y'] = y.tolist()
            all_.append(temp)
        
        import json
        with open(os.path.join(self.logDir, 'attn_text.json'), 'w') as fout:
            json.dump(all_, fout)


def _adding_data(T):
    x_values = np.random.uniform(low=0, high=1, size=T)
    x_indicator = np.zeros(T, dtype=np.bool)
    x_indicator[np.random.randint(T/2)] = 1
    x_indicator[np.random.randint(T/2, T)] = 1
    #x = np.array(list(zip(x_values, x_indicator)))[np.newaxis]
    x = np.vstack((x_values, x_indicator)).T
    #y = np.sum(x_values[x_indicator], keepdims=True)/2.
    y = np.sum(x_values[x_indicator], keepdims=True)/2.
    return x, y

def adding_data(T, batch_size, epoch_len):
    x = np.zeros((epoch_len, batch_size, T, 2))
    y = np.zeros((epoch_len, batch_size))
    for i in range(epoch_len):
        for b in range(batch_size):
            data = _adding_data(T)
            x[i][b] = data[0]
            y[i][b] = data[1][0]
    return (x, y)

def save_param(model, model_file_name):
    torch.save(model.state_dict(), model_file_name)

def load_param(model, model_file_name):
    model.load_state_dict(torch.load(model_file_name))
    return model



#
# Run an experiment.
#
# 1) Create experiment object from parsed arguments.
# 2) Attempt to roll it back to the latest snapshot if possible.
# 3) Resume the experiment at that point.
#

SABExperiment(d).rollback().run()




"""
Stale code with potential for reuse.
"""
"""
def print_norm():
    param_norm = []
    for param in rnn.parameters():
        norm = param.grad.norm(2).data[0]/ numpy.sqrt(numpy.prod(param.size()))
        #print param.size()
        param_norm.append(norm)

    return param_norm


def printgradnorm(self, grad_input, grad_output):
    '''print('Inside ' + self.__class__.__name__ + ' backward')
    print('Inside class:' + self.__class__.__name__)
    print('')
    print('grad_input: ', type(grad_input))
    print('grad_input[0]: ', type(grad_input[0]))
    print('grad_output: ', type(grad_output))
    print('grad_output[0]: ', type(grad_output[0]))
    print('')
    print('grad_input size:', grad_input[0].size())
    print('grad_output size:', grad_output[0].size()) '''
    print('grad_input norm:', grad_input[0].data.norm())
    #print('grad_output norm:', grad_output[0].data.norm())


# rnn.fc.register_backward_hook(printgradnorm)



re_load = True
import ipdb; ipdb.set_trace()
if re_load and os.path.exists(model_id):
    rnn = load_param(rnn, "T_200_LSTM-sparse_1_step_attn_topk_attn_in_h10_truncate_length_5attn_everyk_5_rnn_dim_128_7212.pkl")
    print '--- Evaluating reloaded model ----'
    epoch = 0
    import ipdb; ipdb.set_trace()
    evaluate_valid(test_x, test_y, 1.0)
    #attn_viz_file = model_id + '_epoch_'+str(epoch) + '_iter_0'
    #attention_viz(attn_w_viz, attn_viz_file)


"""

