#
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.common_config import get_con_pre_model
from utils.utils import _init_weight




class TeacherStudentModel(nn.Module):
    '''

    First Only consider the contrastive within novel batch
    
    '''
    def __init__(self, p):
        """
        p: configuration dict
        """
        super(TeacherStudentModel, self).__init__()


        # create the model 
        self.model_q = get_con_pre_model(p)
        self.model_k = get_con_pre_model(p)
        self.m = p['teacher_student_kwargs']['m']
        self.K = p['teacher_student_kwargs']['K'] 

        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.dim = self.model_q.head.in_channels
        self.register_buffer("queue", torch.randn(self.dim, self.K))


    def _freeze_backbone_bn(self):
        self.model_q.backbone.eval()
        self.model_k.backbone.eval()
        
    def _parameter_groups(self, ft_layer_list):
        backbone_params = []
        decoder_params = []
        ## add head in the list
        if 'head' not in ft_layer_list:
            ft_layer_list.append('head')

        for name, param in self.model_q.named_parameters():
            for ft_layer in ft_layer_list:
                if name.startswith(ft_layer):
                    param.requires_grad = True
                    if name.startswith('backbone'):
                        print('Add {} in backbone params'.format(name))
                        backbone_params.append(param)
                    else:
                        print('Add {} in decoder params'.format(name))
                        decoder_params.append(param)

                    break

                else:
                    param.requires_grad = False
        return backbone_params, decoder_params

    def _initialize_params(self):
        _init_weight(self.model_q.decoder, mode='fan_in', nonlinearity='relu')
        _init_weight(self.model_q.head, mode='fan_in', nonlinearity='relu')

        _init_weight(self.model_k.decoder, mode='fan_in', nonlinearity='relu')
        _init_weight(self.model_k.head, mode='fan_in', nonlinearity='relu')

    def _copy_params(self):
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)



    def forward(self, im_base, im_easy, im_hard, im_hard_strong):
        ### base easy hard forward q model
        prediction_base, _ = self.model_q(im_base)
        prediction_easy, _ = self.model_q(im_easy)
        prediction_hard, _ = self.model_q(im_hard_strong) ## strong image for model optimization

        with torch.no_grad():
            self._momentum_update_key_encoder()
            weak_k, _ = self.model_k(im_hard) ## weak image for pseudo label

        output_dir = {
            'base':prediction_base,
            'easy': prediction_easy,
            'strong': prediction_hard,
            'weak': weak_k
        }
        return output_dir

