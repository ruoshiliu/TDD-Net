import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np
import torchvision


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: int
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size // 2
        self.bias        = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())


    
class MtConvLSTM(nn.Module):
    
    def __init__(self, input_size, input_dim, output_dim, hidden_dim, kernel_size, num_layers, num_scale,
                 batch_first=False, bias=True, return_all_layers=False, interpolation = 0):
        super(MtConvLSTM, self).__init__()
        
        """
        
        Parameters
        ----------
        input_size    : size of images (has to be power of 2)
        input_dim     : dimension of image (default is 1)
        hidden_dim    : tuple containing tuple of layers (e.g. [[16,32,64],[32,64,128]])
        kernal_size   : tuple containing tuple of kernal sizes (e.g. [[3,3,3],[3,3,5]])
        num_layers    : tuple of number of layers (e.g. [3,3,5])
        predict_steps : how many steps to predict
        self.num_scale     : number of scale (k)
            
        Returns
        -------
        """

#         self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        for i_scale in range(num_scale):
            if not len(kernel_size[i_scale]) == len(hidden_dim[i_scale]) == num_layers[i_scale]:
                raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size
        (self.input_size, self.input_size) = input_size
        self.input_size = int(self.input_size)
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim # tuple containing list of layers
        self.kernel_size = kernel_size # tuple containing list of kernel sizes
        self.num_layers = num_layers # list of number of layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.num_scale = num_scale # list of scale
        
        interpolation_list = ['nearest', 'linear', 'bilinear']
        self.interpolation = interpolation_list[interpolation]

        cell_list = []
        for i_scale in range(self.num_scale):
            cell_list_scale = []
            input_size_scale = int(self.input_size / (np.power(2 , self.num_scale-1-i_scale)))
            for i in range(0, self.num_layers[i_scale]):
                
                cur_input_dim = self.input_dim + self.output_dim if i == 0 else self.hidden_dim[i_scale][i-1]

                cell_list_scale.append(ConvLSTMCell(input_size=(input_size_scale, input_size_scale),
                                              input_dim=cur_input_dim, # last layer output will be added to input tensor
                                              hidden_dim=self.hidden_dim[i_scale][i],
                                              kernel_size=self.kernel_size[i_scale][i],
                                              bias=self.bias))
#             if i != self.num_scale-1:
#                 activConv = nn.Conv2d(in_channels = np.sum(hidden_dim[i_scale]),
#                                   out_channels = input_dim,
#                                   kernel_size = 1,
#                                    padding = 0)
#             else:
            activConv = nn.Conv2d(in_channels = np.sum(hidden_dim[i_scale]),
                              out_channels = output_dim,
                              kernel_size = 1,
                               padding = 0)
            cell_list_scale.append(activConv)
            cell_list_scale = nn.ModuleList(cell_list_scale)    
            cell_list.append(cell_list_scale)
        # tuple of tuple of ConvLSTM cells dimension 0 is self.num_scale, dimension 1 is num_layer
        self.cell_list = nn.ModuleList(cell_list) 
#         self.cur_layer_input = []
#         self.cur_scale_input = []
#         self.pred_last_scale = []
        
    def forward(self, input_tensor, hidden_state=None):
        """
        # downsample the input_tensor to fit scale
        # hidden states of each layer, updated layer by layer
        # loop layer by layer
        # first use input tensor and initial state to calculate h, c for first layer
        # store first layer hidden states
        # use previous hidden states as cur_layer_input to calculate next layer
        
        Parameters
        ----------
        input_tensor: todo 
            5-D Tensor of shape (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
            
        Returns
        -------
        last_state_list      : last hidden states of each ConvLSTM
        layer_output_list    : hidden and cell states of each layer
        pred_output          : predicted image by each layer and each scale
        """
        
        layer_output_list = []
        last_state_list   = []
        pred_output       = []
        pred_image_list   = []
        batch_size = input_tensor.size(0)
        #----------------------------------------#
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=batch_size)
        for i_scale in range(self.num_scale):
            input_size_scale = int(self.input_size / (np.power(2 , self.num_scale-1-i_scale)))
            
            layer_output_list_scale = []
            last_state_list_scale   = []
            pred_output_scale       = []                               

            seq_len = input_tensor.size(1)
            # downsample input tensor                              
            cur_scale_input = self._interpolate(input_tensor, size=input_size_scale, mode=self.interpolation).cuda()
            # for the first scale, initialize pred_last_scale, dimension of pred_last_scale should match output dimension
            if i_scale == 0: 
                cur_scale_input_size = list(cur_scale_input.size())
                cur_scale_input_size[2] = self.output_dim
                pred_last_scale = torch.zeros(cur_scale_input_size).cuda()
            else:
                # upsample previous prediction
                pred_last_scale = self._interpolate(pred_last_scale, size=input_size_scale, mode=self.interpolation).cuda()
            cur_scale_input = torch.cat((cur_scale_input, pred_last_scale), 2) # concatenate cur and last as input
            cur_layer_input = cur_scale_input
            #----------------------------------------#
            for layer_idx in range(self.num_layers[i_scale]):
                h, c = hidden_state[i_scale][layer_idx]
                output_inner = []
                for t in range(seq_len):

                    h, c = self.cell_list[i_scale][layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                     cur_state=[h, c])
                    output_inner.append(h)

                layer_output = torch.stack(output_inner, dim=1)
                cur_layer_input = layer_output

                layer_output_list_scale.append(layer_output) # store all hidden states for each layer
                last_state_list_scale.append([h, c]) # store all hidden and cell states for last step
            #----------------------------------------#
            if not self.return_all_layers:
                layer_output_list_scale = layer_output_list_scale[-1:]
                last_state_list_scale   = last_state_list_scale[-1:]
            
            pred_output_scale = []
            activ_idx = self.num_layers[i_scale]
            for t in  range(seq_len):
                # append prediction at each scale
                pred_output_scale.append(self.cell_list[i_scale][activ_idx](torch.cat(layer_output_list_scale, dim=2)[:,t,:,:,:]))
                                             
            pred_output_scale = torch.stack(pred_output_scale, dim=1) # (b,t,c,h_scale,w_scale)
                
            pred_last_scale = pred_output_scale
                                             
            layer_output_list.append(layer_output_list_scale)
            last_state_list.append(last_state_list_scale)
            pred_output.append(pred_output_scale)                               
        #----------------------------------------#
        
        # for each step in the encoded sequence, combine prediction from each scale to get output confidence map
        
        for i_step in range(seq_len):
            output = torch.zeros(batch_size,self.output_dim,self.input_size, self.input_size).cuda() # output at each step
            for i_scale in range(self.num_scale):
                pred_scale = pred_output[i_scale][:,i_step,:,:,:]
                pred_scale = torch.nn.functional.interpolate(pred_scale, size=self.input_size, mode=self.interpolation)
                output += pred_scale
            pred_image_list.append(output)
        
        return layer_output_list, last_state_list, pred_output, pred_image_list

    
    
    
    def _init_hidden(self, batch_size):
        init_states = []
        for i_scale in range(0, self.num_scale):
            init_states_scale = []
            for i in range(self.num_layers[i_scale]):
                init_states_scale.append(self.cell_list[i_scale][i].init_hidden(batch_size))
            init_states.append(init_states_scale)
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    @staticmethod
    def _interpolate(tensor, size, mode):
        tensor_out = []
        for t in range(tensor.size()[1]):
            tensor_t = torch.nn.functional.interpolate(tensor[:,t,:,:,:], size=size, mode=mode)
            tensor_out.append(tensor_t)
        tensor_out = torch.stack(tensor_out, 1)
        return tensor_out