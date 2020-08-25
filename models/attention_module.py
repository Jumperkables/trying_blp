import torch
import torch.nn as nn
import torch.nn.functional as F
from memory_rand import *
from memory_module import *
################################################################################################
import os
import sys
sys.path.insert(0, os.path.expanduser("~/kable_management/pooling_pkgs/block_py2") )
import fusions
################################################################################################

class SpatialAttentionModule(nn.Module):

    def __init__(self, input_size=3072, feat_dim=7, hidden_size=512, dropout=0.2):
        """Set the hyper-parameters and build the layers."""
        super(SpatialAttentionModule, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.feat_dim = feat_dim
        
        # alignment model
        # see appendices A.1.2 of neural machine translation
        self.Wa = nn.Parameter(torch.FloatTensor(input_size, hidden_size),requires_grad=True)
        self.Ua = nn.Parameter(torch.FloatTensor(hidden_size*2, hidden_size),requires_grad=True)
        self.Va = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.ba = nn.Parameter(torch.FloatTensor(1,1,hidden_size),requires_grad=True)
        
        self.drop_keep_prob_image_embed_t = nn.Dropout(dropout)
         
        self.init_weights()
        
    def init_weights(self):
        self.Wa.data.normal_(0.0, 0.01)
        self.Ua.data.normal_(0.0, 0.01)
        self.Va.data.normal_(0.0, 0.01)
        self.ba.data.fill_(0)
        
        

    def forward(self, hidden_frames, hidden_text):
        
        # hidden_text:  1 x 1024 (tgif-qa paper Section 4.2, use a two layer one-directional LSTM, combining each layer's hidden)
        # hidden_frame: 1 x 7 x 7 x 3072 (from C3D and resnet, 1024+2048 = 3072)
        #print(hidden_frames.size(),'spatial 1---') # (1, 1, 1, 2048) or (1, 7, 7, 3072)
        assert self.feat_dim==hidden_frames.size(2)
        hidden_frames = hidden_frames.view(hidden_frames.size(0), hidden_frames.size(1) * hidden_frames.size(2), hidden_frames.size(3))
        #hidden_frames = hidden_frames.permute([0,2,1])
        #print('Spatial hidden_frames transferred size', hidden_frames.size())  # should be 1 x 49 x 3072 or (1, 1, 2048)
        
        # precompute Uahj, see Appendix A.1.2 Page12, last sentence, 
        # NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE
        Uh = torch.matmul(hidden_text, self.Ua)  # (1,512)
        Uh = Uh.view(Uh.size(0),1,Uh.size(1)) # (1,1,512)
        #Uh = Uh.repeat(1,hidden_frames.size(1),1) 
        #print('Spatial Uh size', Uh.size())  # (1, 1, 512)

        
        # see appendices A.1.2 of neural machine translation
        # Page 12 last line
        # W is 512x512, s_i-1 is 
        Ws = torch.matmul(hidden_frames, self.Wa) # (1,49,512)
        #print('Spatial Ws size',Ws.size())   # (1, 1 or 49, 512)
        att_vec = torch.matmul( torch.tanh(Ws + Uh + self.ba), self.Va )
        att_vec = F.softmax(att_vec, dim=1) # normalize by Softmax, see Eq(15)
        #print('Spatial att_vec size',att_vec.size()) # should be 1x49, as weights for each encoder output ht
        att_vec = att_vec.view(att_vec.size(0),att_vec.size(1),1) # expand att_vec from 1x49 to 1x49x1
        #print('Spatial expanded att_vec size',att_vec.size())   # (1, 1 or 49, 1)
                
        
        # Hori ICCV 2017
        # Eq(10) c_i
        ht_weighted = att_vec * hidden_frames
        #print('Spatial ht_weighted size', ht_weighted.size()) # should be (1,49,input_size)  (1, 1(or 49), 2048)
        ht_sum = torch.sum(ht_weighted, dim=1)
        ht_sum = self.drop_keep_prob_image_embed_t(ht_sum)
        #print('Spatial ht_sum size', ht_sum.size())  # should be (1,input_size)  (1, 2048)
        
        return ht_sum
         
class TemporalAttentionModule(nn.Module):

    def __init__(self, input_size, hidden_size=512):
        """Set the hyper-parameters and build the layers."""
        super(TemporalAttentionModule, self).__init__()
        self.input_size = input_size   # in most cases, 2*hidden_size
        self.hidden_size = hidden_size
        
        # alignment model
        # see appendices A.1.2 of neural machine translation
        #self.Wa = nn.Parameter(torch.FloatTensor(hidden_size*2, hidden_size),requires_grad=True)
        #self.Ua = nn.Parameter(torch.FloatTensor(hidden_size*2, hidden_size),requires_grad=True)
        self.Wa = nn.Parameter(torch.FloatTensor(input_size, hidden_size),requires_grad=True)
        self.Ua = nn.Parameter(torch.FloatTensor(input_size, hidden_size),requires_grad=True)
        self.Va = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.ba = nn.Parameter(torch.FloatTensor(1,1,hidden_size),requires_grad=True)
        
        self.init_weights()
        
        
    def init_weights(self):
        self.Wa.data.normal_(0.0, 0.01)
        self.Ua.data.normal_(0.0, 0.01)
        self.Va.data.normal_(0.0, 0.01)
        self.ba.data.fill_(0)


    def forward(self, hidden_frames, hidden_text, inv_attention=False):
        
        # hidden_text:  1 x 1024 (looks like a two layer one-directional LSTM, combining each layer's hidden)
        # hidden_frame: 1 x T x 1024 (from video encoder output, 1024 is similar from above)

        #print('Temporal hidden_text transferred size', hidden_text.size())  # should be 1 x 1024
        #print('Temporal hidden_frames transferred size', hidden_frames.size())  # should be 1 x T x 3072
        
        # precompute Uahj, see Appendix A.1.2 Page12, last sentence, 
        # NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE
        Uh = torch.matmul(hidden_text, self.Ua)  # (1,512)
        Uh = Uh.view(Uh.size(0),1,Uh.size(1)) # (1,1,512)
        #print('Temporal Uh size', Uh.size())  # (1, 1, 512)

        
        # see appendices A.1.2 of neural machine translation
        # Page 12 last line
        Ws = torch.matmul(hidden_frames, self.Wa) # (1,T,512)
        #print('Temporal Ws size',Ws.size())       # (1, T, 512)
        att_vec = torch.matmul( torch.tanh(Ws + Uh + self.ba), self.Va )
        
        if inv_attention==True:
            att_vec = - att_vec
        
        att_vec = F.softmax(att_vec, dim=1) # normalize by Softmax, see Eq(15)
        #print('Temporal att_vec size',att_vec.size()) # should be 1xT, as weights for each encoder output ht  (1,T,1)
        
#         if inv_attention==True:
#             att_vec = 1.0 - att_vec
        
        att_vec = att_vec.view(att_vec.size(0),att_vec.size(1),1) # expand att_vec from 1xT to 1xTx1 
        #print('Temporal expanded att_vec size',att_vec.size())
                
        
        # Hori ICCV 2017
        # Eq(10) c_i
        ht_weighted = att_vec * hidden_frames
        #print('Temporal ht_weighted size', ht_weighted.size()) # should be (1,T,input_size)  # (1, T, 1024)
        ht_sum = torch.sum(ht_weighted, dim=1)
        #print('Temporal ht_sum size', ht_sum.size())  # should be (1,input_size)  (1, 1024)
        
        return ht_sum

# attention-based multimodal
class MultiModalNaiveModule(nn.Module):

    def __init__(self, hidden_size=512, simple=False):
        """Set the hyper-parameters and build the layers."""
        super(MultiModalNaiveModule, self).__init__()

        self.hidden_size = hidden_size
        self.simple=simple
        
        # alignment model
        # see appendices A.1.2 of neural machine translation
        
        self.Wav = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Wat = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Uav = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Uat = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Vav = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.Vat = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.bav = nn.Parameter(torch.FloatTensor(1,1,hidden_size),requires_grad=True)
        self.bat = nn.Parameter(torch.FloatTensor(1,1,hidden_size),requires_grad=True)

        self.Whh = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Wvh = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Wth = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.bh = nn.Parameter(torch.FloatTensor(1,1,hidden_size),requires_grad=True)

        self.init_weights()
        
    def init_weights(self):
        self.Wav.data.normal_(0.0, 0.01)
        self.Wat.data.normal_(0.0, 0.01)
        self.Uav.data.normal_(0.0, 0.01)
        self.Uat.data.normal_(0.0, 0.01)
        self.Vav.data.normal_(0.0, 0.01)
        self.Vat.data.normal_(0.0, 0.01)
        self.bav.data.fill_(0)
        self.bat.data.fill_(0)

        self.Whh.data.normal_(0.0, 0.01)
        self.Wvh.data.normal_(0.0, 0.01)
        self.Wth.data.normal_(0.0, 0.01)
        self.bh.data.fill_(0)

    def forward(self, h, hidden_frames, hidden_text, inv_attention=False):
        
        # hidden_text:  1 x T1 x 1024 (looks like a two layer one-directional LSTM, combining each layer's hidden)
        # hidden_frame: 1 x T2 x 1024 (from video encoder output, 1024 is similar from above)

        #print hidden_frames.size(),hidden_text.size()
        Uhv = torch.matmul(h, self.Uav)  # (1,512)
        Uhv = Uhv.view(Uhv.size(0),1,Uhv.size(1)) # (1,1,512)

        Uht = torch.matmul(h, self.Uat)  # (1,512)
        Uht = Uht.view(Uht.size(0),1,Uht.size(1)) # (1,1,512)
        
        #print Uhv.size(),Uht.size()
        
        Wsv = torch.matmul(hidden_frames, self.Wav) # (1,T,512)
        #print Wsv.size()
        att_vec_v = torch.matmul( torch.tanh(Wsv + Uhv + self.bav), self.Vav )
        
        Wst = torch.matmul(hidden_text, self.Wat) # (1,T,512)
        att_vec_t = torch.matmul( torch.tanh(Wst + Uht + self.bat), self.Vat )
        

        
        att_vec_v = torch.softmax(att_vec_v, dim=1)
        att_vec_t = torch.softmax(att_vec_t, dim=1)
        #print att_vec_v.size(),att_vec_t.size()
        
        
        att_vec_v = att_vec_v.view(att_vec_v.size(0),att_vec_v.size(1),1) # expand att_vec from 1xT to 1xTx1 
        att_vec_t = att_vec_t.view(att_vec_t.size(0),att_vec_t.size(1),1) # expand att_vec from 1xT to 1xTx1 
                
        hv_weighted = att_vec_v * hidden_frames
        hv_sum = torch.sum(hv_weighted, dim=1)
        
        ht_weighted = att_vec_t * hidden_text
        ht_sum = torch.sum(ht_weighted, dim=1)
        
        
        output = torch.tanh( torch.matmul(h,self.Whh) + torch.matmul(hv_sum,self.Wvh) + 
                             torch.matmul(ht_sum, self.Wth) + self.bh )
        output = output.view(output.size(1),output.size(2))
        
        return output
        
class MultiModalAttentionModule(nn.Module):

    def __init__(self, opt, blp, hidden_size=512, simple=False):
        """Set the hyper-parameters and build the layers."""
        super(MultiModalAttentionModule, self).__init__()

        self.hidden_size = hidden_size
        self.simple=simple
        
        # alignment model
        # see appendices A.1.2 of neural machine translation
        
        self.Wav = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Wat = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Uav = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Uat = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Vav = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.Vat = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.bav = nn.Parameter(torch.FloatTensor(1,1,hidden_size),requires_grad=True)
        self.bat = nn.Parameter(torch.FloatTensor(1,1,hidden_size),requires_grad=True)

        self.Whh = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Wvh = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Wth = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.bh = nn.Parameter(torch.FloatTensor(1,1,hidden_size),requires_grad=True)

        self.video_sum_encoder = nn.Linear(hidden_size, hidden_size) 
        self.question_sum_encoder = nn.Linear(hidden_size, hidden_size) 
        

        self.Wb = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Vbv = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Vbt = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.bbv = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.bbt = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.wb = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)

        self.blp = blp

        
        self.init_weights()
        
    def init_weights(self):
        self.Wav.data.normal_(0.0, 0.01)
        self.Wat.data.normal_(0.0, 0.01)
        self.Uav.data.normal_(0.0, 0.01)
        self.Uat.data.normal_(0.0, 0.01)
        self.Vav.data.normal_(0.0, 0.01)
        self.Vat.data.normal_(0.0, 0.01)
        self.bav.data.fill_(0)
        self.bat.data.fill_(0)

        self.Whh.data.normal_(0.0, 0.01)
        self.Wvh.data.normal_(0.0, 0.01)
        self.Wth.data.normal_(0.0, 0.01)
        self.bh.data.fill_(0)

        self.video_sum_encoder.weight.data.normal_(0.0, 0.02)
        self.video_sum_encoder.bias.data.fill_(0)
        self.question_sum_encoder.weight.data.normal_(0.0, 0.02)
        self.question_sum_encoder.bias.data.fill_(0)


        self.Wb.data.normal_(0.0, 0.01)
        self.Vbv.data.normal_(0.0, 0.01)
        self.Vbt.data.normal_(0.0, 0.01)
        self.wb.data.normal_(0.0, 0.01)
        
        self.bbv.data.fill_(0)
        self.bbt.data.fill_(0)

        
    def forward(self, h, hidden_frames, hidden_text):
        
        # hidden_text:  1 x T1 x 1024 (looks like a two layer one-directional LSTM, combining each layer's hidden)
        # hidden_frame: 1 x T2 x 1024 (from video encoder output, 1024 is similar from above)

        #print hidden_frames.size(),hidden_text.size()
        Uhv = torch.matmul(h, self.Uav)  # (1,512)
        Uhv = Uhv.view(Uhv.size(0),1,Uhv.size(1)) # (1,1,512)

        Uht = torch.matmul(h, self.Uat)  # (1,512)
        Uht = Uht.view(Uht.size(0),1,Uht.size(1)) # (1,1,512)
        
        #print Uhv.size(),Uht.size()
        
        Wsv = torch.matmul(hidden_frames, self.Wav) # (1,T,512)
        #print Wsv.size()
        att_vec_v = torch.matmul( torch.tanh(Wsv + Uhv + self.bav), self.Vav )
        
        Wst = torch.matmul(hidden_text, self.Wat) # (1,T,512)
        att_vec_t = torch.matmul( torch.tanh(Wst + Uht + self.bat), self.Vat )

        
        att_vec_v = torch.softmax(att_vec_v, dim=1)
        att_vec_t = torch.softmax(att_vec_t, dim=1)
        #print att_vec_v.size(),att_vec_t.size()
        
        att_vec_v = att_vec_v.view(att_vec_v.size(0),att_vec_v.size(1),1) # expand att_vec from 1xT to 1xTx1 
        att_vec_t = att_vec_t.view(att_vec_t.size(0),att_vec_t.size(1),1) # expand att_vec from 1xT to 1xTx1 
                
        hv_weighted = att_vec_v * hidden_frames
        hv_sum = torch.sum(hv_weighted, dim=1)
        hv_sum2 = self.video_sum_encoder(hv_sum)

        ht_weighted = att_vec_t * hidden_text
        ht_sum = torch.sum(ht_weighted, dim=1)
        ht_sum2 = self.question_sum_encoder(ht_sum)        
        
        
        Wbs = torch.matmul(h, self.Wb)
        mt1 = torch.matmul(ht_sum, self.Vbt) + self.bbt + Wbs
        mv1 = torch.matmul(hv_sum, self.Vbv) + self.bbv + Wbs
        
        # Bilinear pooling goes here
        if self.blp is not None:
            mtv = torch.tanh(self.blp([mv1[0],mt1[0]]).view(2,-1))
        else:
            mtv =  torch.tanh(torch.cat([mv1,mt1],dim=0))


        mtv2 = torch.matmul(mtv, self.wb)
        beta = torch.softmax(mtv2,dim=0)
        #print beta.size(),beta        
        
        output = torch.tanh( torch.matmul(h,self.Whh) + beta[0] * hv_sum2 + 
                             beta[1] * ht_sum2 + self.bh )
        output = output.view(output.size(1),output.size(2))
        
        return output


class LSTMEncDec(nn.Module):

    def __init__(self, feat_channel, feat_dim, text_embed_size, hidden_size, vocab_size, num_layers, word_matrix,
                 answer_vocab_size=None, max_len=20, dropout=0.2):
        """Set the hyper-parameters and build the layers."""
        super(LSTMEncDec, self).__init__()

        # text input size
        self.text_embed_size = text_embed_size  # should be 300

        # video input size
        self.feat_channel = feat_channel
        self.feat_dim = feat_dim  # should be 7

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.drop_keep_prob_final_att_vec = nn.Dropout(dropout)

        self.embed = nn.Embedding(vocab_size, text_embed_size)

        self.lstm_text_1 = nn.LSTMCell(text_embed_size, hidden_size)
        self.lstm_text_2 = nn.LSTMCell(hidden_size, hidden_size)

        self.lstm_video_1 = nn.LSTMCell(feat_channel, hidden_size)
        self.lstm_video_2 = nn.LSTMCell(hidden_size, hidden_size)

        self.video_encoder = nn.Linear(feat_channel, hidden_size * 2)

        self.linear_decoder_count_1 = nn.Linear(hidden_size * 2, hidden_size * 2)

        if answer_vocab_size is not None:
            self.linear_decoder_count_2 = nn.Linear(hidden_size * 2, answer_vocab_size)
        else:
            self.linear_decoder_count_2 = nn.Linear(hidden_size * 2, 1)  # Count is regression problem

        self.max_len = max_len

        self.init_weights(word_matrix)

    def init_weights(self, word_matrix):
        """Initialize weights."""
        if word_matrix is None:
            self.embed.weight.data.uniform_(-0.1, 0.1)
        else:
            # init embed from glove
            self.embed.weight.data.copy_(torch.from_numpy(word_matrix))

    def init_hiddens(self):
        s_t = torch.zeros(1, self.hidden_size).cuda()
        s_t2 = torch.zeros(1, self.hidden_size).cuda()
        c_t = torch.zeros(1, self.hidden_size).cuda()
        c_t2 = torch.zeros(1, self.hidden_size).cuda()
        return s_t, s_t2, c_t, c_t2

    def forward(self, data_dict):
        ret = self.forward_action(data_dict)
        return ret

    def forward_action(self, data_dict):

        video_features = data_dict['video_features']
        questions, question_lengths = data_dict['question_words'], data_dict['question_lengths']
        num_mult_choices = data_dict['num_mult_choices']

        outputs = []
        predictions = []
        batch_size = len(questions)  # batch size has to be 1

        for j in range(batch_size):
            features_question_j = self.embed(questions[j])
            vgg, c3d = video_features[0][j], video_features[1][j]
            feature = torch.cat([vgg, c3d], dim=1)
            #print feature.size()

            nImg = vgg.shape[0]
            outputs_j = []

            for n_cand in range(num_mult_choices):

                nQuestionWords = question_lengths[j]

                #############################
                # run text encoder first time
                #############################
                s1_t1, s1_t2, c1_t1, c1_t2 = self.init_hiddens()

                for i in xrange(nQuestionWords):
                    input_question = features_question_j[n_cand, i:i+1]
                    s1_t1, c1_t1 = self.lstm_text_1(input_question, (s1_t1, c1_t1))
                    s1_t2, c1_t2 = self.lstm_text_2(s1_t1, (s1_t2, c1_t2))

                ###########################################
                # run video encoder with spatial attention
                ###########################################
                sV_t1, sV_t2, cV_t1, cV_t2 = self.init_hiddens()

                # record each time t, hidden states, for later temporal attention after text encoding
                hidden_array_1 = []
                hidden_array_2 = []

                for i in xrange(nImg):
                    # lstm
                    sV_t1, cV_t1 = self.lstm_video_1(feature[i:i + 1], (sV_t1, cV_t1))
                    sV_t2, cV_t2 = self.lstm_video_2(sV_t1, (sV_t2, cV_t2))

                    sV_t1_vec = sV_t1.view(sV_t1.size(0), 1, sV_t1.size(1))
                    sV_t2_vec = sV_t2.view(sV_t2.size(0), 1, sV_t2.size(1))

                    hidden_array_1.append(sV_t1_vec)
                    hidden_array_2.append(sV_t2_vec)

                # assume sV_t1 is of size (1,1,hidden)
                sV_l1 = torch.cat(hidden_array_1, dim=1)
                sV_l2 = torch.cat(hidden_array_2, dim=1)

                sV_ll = torch.cat((sV_l1, sV_l2), dim=2)

                #############################
                # run text encoder second time
                #############################
                # here sT_t1, sT_t2 are the last hiddens from video, input to text encoder againa
                input_question = features_question_j[n_cand,0:1]
                # print('input_question size', input_question.size())
                sT_t1, cT_t1 = self.lstm_text_1(input_question, (sV_t1, cV_t1))
                sT_t2, cT_t2 = self.lstm_text_2(sT_t1, (sV_t2, cV_t2))

                for i in xrange(1, nQuestionWords):
                    input_question = features_question_j[n_cand,i:i+1]
                    sT_t1, cT_t1 = self.lstm_text_1(input_question, (sT_t1, cT_t1))
                    sT_t2, cT_t2 = self.lstm_text_2(sT_t1, (sT_t2, cT_t2))
                # print('Text encoding One size', sT_t1.size(), sT_t2.size())

                # here sT_t1, sT_t2 is the last hidden
                sT_t = torch.cat((sT_t1, sT_t2), dim=1)  # should be of size (1,1024)

                final_embed = self.linear_decoder_count_1(sT_t)
                output = self.linear_decoder_count_2(final_embed)
                outputs_j.append(output)

            # output is the score of each multiple choice
            outputs_j = torch.cat(outputs_j, 1)
            outputs.append(outputs_j)

            # for evaluate accuracy, find the max one
            _, mx_idx = torch.max(outputs_j, 1)
            predictions.append(mx_idx)
            # print(outputs_j,mx_idx)

        outputs = torch.cat(outputs, 0)
        predictions = torch.cat(predictions, 0)
        return outputs, predictions

    def accuracy(self, logits, targets):
        correct = torch.sum(logits.eq(targets)).float()
        return correct * 100.0 / targets.size(0)


class TGIFBenchmark(nn.Module):

    def __init__(self, feat_channel, feat_dim, text_embed_size, hidden_size, vocab_size, num_layers, word_matrix,
                 answer_vocab_size=None, max_len=20, dropout=0.2):
        """Set the hyper-parameters and build the layers."""
        super(TGIFBenchmark, self).__init__()

        # text input size
        self.text_embed_size = text_embed_size  # should be 300

        # video input size
        self.feat_channel = feat_channel
        self.feat_dim = feat_dim  # should be 7

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.TpAtt = TemporalAttentionModule(hidden_size * 2, hidden_size)

        self.drop_keep_prob_final_att_vec = nn.Dropout(dropout)

        self.embed = nn.Embedding(vocab_size, text_embed_size)

        self.lstm_text_1 = nn.LSTMCell(text_embed_size, hidden_size)
        self.lstm_text_2 = nn.LSTMCell(hidden_size, hidden_size)

        self.lstm_video_1 = nn.LSTMCell(feat_channel, hidden_size)
        self.lstm_video_2 = nn.LSTMCell(hidden_size, hidden_size)

        self.video_encoder = nn.Linear(feat_channel, hidden_size * 2)

        self.linear_decoder_att = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.linear_decoder_mem = nn.Linear(hidden_size * 2 + hidden_size, hidden_size * 2)
        self.linear_decoder_count_1 = nn.Linear(hidden_size * 2, hidden_size * 2)

        if answer_vocab_size is not None:
            self.linear_decoder_count_2 = nn.Linear(hidden_size * 2, answer_vocab_size)
        else:
            self.linear_decoder_count_2 = nn.Linear(hidden_size * 2, 1)  # Count is regression problem

        self.max_len = max_len

        self.init_weights(word_matrix)

    def init_weights(self, word_matrix):
        """Initialize weights."""
        if word_matrix is None:
            self.embed.weight.data.uniform_(-0.1, 0.1)
        else:
            # init embed from glove
            self.embed.weight.data.copy_(torch.from_numpy(word_matrix))

    def init_hiddens(self):
        s_t = torch.zeros(1, self.hidden_size).cuda()
        s_t2 = torch.zeros(1, self.hidden_size).cuda()
        c_t = torch.zeros(1, self.hidden_size).cuda()
        c_t2 = torch.zeros(1, self.hidden_size).cuda()
        return s_t, s_t2, c_t, c_t2

    def forward(self, data_dict):
        ret = self.forward_action(data_dict)
        return ret

    def forward_action(self, data_dict):

        video_features = data_dict['video_features']
        questions, question_lengths = data_dict['question_words'], data_dict['question_lengths']
        num_mult_choices = data_dict['num_mult_choices']

        outputs = []
        predictions = []
        batch_size = len(questions)  # batch size has to be 1

        for j in range(batch_size):
            features_question_j = self.embed(questions[j])
            vgg, c3d = video_features[0][j], video_features[1][j]
            feature = torch.cat([vgg,c3d], dim=1)
            #print feature.size()

            nImg = vgg.shape[0]
            outputs_j = []

            for n_cand in range(num_mult_choices):

                nQuestionWords = question_lengths[j]

                #############################
                # run text encoder first time
                #############################
                s1_t1, s1_t2, c1_t1, c1_t2 = self.init_hiddens()

                for i in xrange(nQuestionWords):
                    input_question = features_question_j[n_cand,i:i+1]
                    s1_t1, c1_t1 = self.lstm_text_1(input_question, (s1_t1, c1_t1))
                    s1_t2, c1_t2 = self.lstm_text_2(s1_t1, (s1_t2, c1_t2))

                # here s1_t1, s1_t2 is the last hidden
                s1_t = torch.cat((s1_t1, s1_t2), dim=1)  # should be of size (1,1024)

                ###########################################
                # run video encoder with spatial attention
                ###########################################
                sV_t1, sV_t2, cV_t1, cV_t2 = self.init_hiddens()

                # record each time t, hidden states, for later temporal attention after text encoding
                hidden_array_1 = []
                hidden_array_2 = []

                for i in xrange(nImg):
                    # lstm
                    sV_t1, cV_t1 = self.lstm_video_1(feature[i:i + 1], (sV_t1, cV_t1))
                    # print('Video encoding One size', sV_t1.size(), sV_t2.size())  # (1, 512), (1, 512)
                    sV_t2, cV_t2 = self.lstm_video_2(sV_t1, (sV_t2, cV_t2))

                    sV_t1_vec = sV_t1.view(sV_t1.size(0), 1, sV_t1.size(1))
                    sV_t2_vec = sV_t2.view(sV_t2.size(0), 1, sV_t2.size(1))

                    hidden_array_1.append(sV_t1_vec)
                    hidden_array_2.append(sV_t2_vec)

                # assume sV_t1 is of size (1,1,hidden)
                sV_l1 = torch.cat(hidden_array_1, dim=1)
                sV_l2 = torch.cat(hidden_array_2, dim=1)
                # print('sV_l1 sV_l2 size', sV_l1.size(),sV_l2.size()) # should be (1,1,hidden*2)

                sV_ll = torch.cat((sV_l1, sV_l2), dim=2)
                # print('Video encoding all frame size', sV_ll.size()) # should be (1,1,hidden*2)

                #############################
                # run text encoder second time
                #############################
                # here sT_t1, sT_t2 are the last hiddens from video, input to text encoder againa
                input_question = features_question_j[n_cand,0:1]
                # print('input_question size', input_question.size())
                sT_t1, cT_t1 = self.lstm_text_1(input_question, (sV_t1, cV_t1))
                sT_t2, cT_t2 = self.lstm_text_2(sT_t1, (sV_t2, cV_t2))

                for i in xrange(1, nQuestionWords):
                    input_question = features_question_j[n_cand,i:i+1]
                    sT_t1, cT_t1 = self.lstm_text_1(input_question, (sT_t1, cT_t1))
                    sT_t2, cT_t2 = self.lstm_text_2(sT_t1, (sT_t2, cT_t2))
                # print('Text encoding One size', sT_t1.size(), sT_t2.size())

                # here sT_t1, sT_t2 is the last hidden
                sT_t = torch.cat((sT_t1, sT_t2), dim=1)  # should be of size (1,1024)

                #####################
                # temporal attention
                #####################
                vid_att = self.TpAtt(sV_ll, sT_t)
                # print('Temporal attention output', vid_att.size())  # (1, 1024)

                #########################
                # decode the final output
                #########################
                # in section 4.4, Temporal attention
                # compute the attended textual signal

                final_embed = torch.tanh(self.linear_decoder_count_1(vid_att)) * sT_t
                output = self.linear_decoder_count_2(final_embed)
                outputs_j.append(output)

            # output is the score of each multiple choice
            outputs_j = torch.cat(outputs_j, 1)
            # print('Output_j size', outputs_j.size()) # (1,5)
            outputs.append(outputs_j)

            # for evaluate accuracy, find the max one
            _, mx_idx = torch.max(outputs_j, 1)
            predictions.append(mx_idx)
            # print(outputs_j,mx_idx)

        outputs = torch.cat(outputs, 0)
        predictions = torch.cat(predictions, 0)
        return outputs, predictions

    def accuracy(self, logits, targets):
        # print(logits.size(), targets.size(), logits.type(), targets.type())
        # print(logits,targets)
        # targets = targets.int()
        correct = torch.sum(logits.eq(targets)).float()
        # print(correct,targets.size(0))
        # print(correct, targets.size(0), correct * 100.0 / targets.size(0))
        return correct * 100.0 / targets.size(0)


class CoMemory(nn.Module):

    def __init__(self, feat_channel, feat_dim, text_embed_size, hidden_size, vocab_size, num_layers, word_matrix,
                 answer_vocab_size=None, max_len=20, dropout=0.2, mm_version=1, useSpatial=False, useNaive=False,
                 mrmUseOriginFeat=False):
        """Set the hyper-parameters and build the layers."""
        super(CoMemory, self).__init__()

        # text input size
        self.text_embed_size = text_embed_size  # should be 300

        # video input size
        self.feat_channel = feat_channel
        self.feat_dim = feat_dim  # should be 7

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.useNaive = useNaive
        self.mrmUseOriginFeat = mrmUseOriginFeat
        self.useSpatial = useSpatial
        self.mm_version = mm_version

        self.TpAtt_a = TemporalAttentionModule(hidden_size * 2, hidden_size)
        self.TpAtt_m = TemporalAttentionModule(hidden_size * 2, hidden_size)

        if useSpatial:
            self.SpAtt = SpatialAttentionModule(feat_channel, feat_dim, hidden_size)
        else:
            self.video_encoder = nn.Linear(feat_channel, hidden_size)

        self.drop_keep_prob_final_att_vec = nn.Dropout(dropout)
        self.embed = nn.Embedding(vocab_size, text_embed_size)

        self.lstm_text_1 = nn.LSTMCell(text_embed_size, hidden_size)
        self.lstm_text_2 = nn.LSTMCell(hidden_size, hidden_size)

        self.lstm_video_1a = nn.LSTMCell(4096, hidden_size)
        self.lstm_video_2a = nn.LSTMCell(hidden_size, hidden_size)

        self.lstm_video_1m = nn.LSTMCell(4096, hidden_size)
        self.lstm_video_2m = nn.LSTMCell(hidden_size, hidden_size)

        self.lstm_mm_1 = nn.LSTMCell(hidden_size, hidden_size)
        self.lstm_mm_2 = nn.LSTMCell(hidden_size, hidden_size)
        self.linear_decoder_mem = nn.Linear(hidden_size * 2, hidden_size)
        self.hidden_encoder_1 = nn.Linear(hidden_size * 2, hidden_size)
        self.hidden_encoder_2 = nn.Linear(hidden_size * 2, hidden_size)

        self.ma_decoder = nn.Linear(hidden_size * 2 * 3, hidden_size * 2)
        self.mb_decoder = nn.Linear(hidden_size * 2 * 3, hidden_size * 2)

        self.linear_decoder_count_1 = nn.Linear(hidden_size * 2 * 2, hidden_size * 2)
        self.linear_decoder_count_2 = nn.Linear(hidden_size * 2, 1)

        self.max_len = max_len

        self.epm1 = EpisodicMemory(hidden_size * 2)
        self.epm2 = EpisodicMemory(hidden_size * 2)

        self.init_weights(word_matrix)

    def init_weights(self, word_matrix):
        """Initialize weights."""
        if word_matrix is None:
            self.embed.weight.data.uniform_(-0.1, 0.1)
        else:
            # init embed from glove
            self.embed.weight.data.copy_(torch.from_numpy(word_matrix))


    def init_hiddens(self):
        s_t = torch.zeros(1, self.hidden_size).cuda()
        s_t2 = torch.zeros(1, self.hidden_size).cuda()
        c_t = torch.zeros(1, self.hidden_size).cuda()
        c_t2 = torch.zeros(1, self.hidden_size).cuda()
        return s_t, s_t2, c_t, c_t2

    def forward(self, data_dict):
        ret = self.forward_action(data_dict)
        return ret

    def forward_action(self, data_dict):

        video_features = data_dict['video_features']
        questions, question_lengths = data_dict['question_words'], data_dict['question_lengths']
        num_mult_choices = data_dict['num_mult_choices']

        outputs = []
        predictions = []
        batch_size = len(questions)  # batch size has to be 1

        for j in range(batch_size):
            features_question_j = self.embed(questions[j])
            vgg, c3d = video_features[0][j], video_features[1][j]

            nImg = vgg.shape[0]
            outputs_j = []

            for n_cand in range(num_mult_choices):

                # print ' ', n_cand,
                nQuestionWords = question_lengths[j]

                #############################
                # run text encoder first time
                #############################
                s1_t1, s1_t2, c1_t1, c1_t2 = self.init_hiddens()

                for i in xrange(nQuestionWords):
                    input_question = features_question_j[n_cand, i:i + 1]
                    # print input_question.size() # (1,300)
                    s1_t1, c1_t1 = self.lstm_text_1(input_question, (s1_t1, c1_t1))
                    s1_t2, c1_t2 = self.lstm_text_2(s1_t1, (s1_t2, c1_t2))

                # here s1_t1, s1_t2 is the last hidden
                s1_t = torch.cat((s1_t1, s1_t2), dim=1)  # should be of size (1,1024)

                ###########################################
                # run video encoder with spatial attention
                ###########################################
                sV_t1a, sV_t2a, cV_t1a, cV_t2a = s1_t1, s1_t2, c1_t1, c1_t2
                sV_t1m, sV_t2m, cV_t1m, cV_t2m = s1_t1, s1_t2, c1_t1, c1_t2

                # record each time t, hidden states, for later temporal attention after text encoding
                hidden_array_1a = []
                hidden_array_2a = []
                hidden_array_1m = []
                hidden_array_2m = []

                for i in xrange(nImg):
                    feat_att_m = vgg[i:i + 1]
                    feat_att_a = c3d[i:i + 1]
                    # print feat_att_a.size(), feat_att_m.size() # (1, 4096) (1, 4096)

                    sV_t1m, cV_t1m = self.lstm_video_1m(feat_att_m, (sV_t1m, cV_t1m))
                    sV_t2m, cV_t2m = self.lstm_video_2m(sV_t1m, (sV_t2m, cV_t2m))

                    sV_t1a, cV_t1a = self.lstm_video_1a(feat_att_a, (sV_t1a, cV_t1a))
                    sV_t2a, cV_t2a = self.lstm_video_2a(sV_t1a, (sV_t2a, cV_t2a))

                    sV_t1a_vec = sV_t1a.view(sV_t1a.size(0), 1, sV_t1a.size(1))
                    sV_t2a_vec = sV_t2a.view(sV_t2a.size(0), 1, sV_t2a.size(1))

                    hidden_array_1a.append(sV_t1a_vec)
                    hidden_array_2a.append(sV_t2a_vec)

                    sV_t1m_vec = sV_t1m.view(sV_t1m.size(0), 1, sV_t1m.size(1))
                    sV_t2m_vec = sV_t2m.view(sV_t2m.size(0), 1, sV_t2m.size(1))

                    hidden_array_1m.append(sV_t1m_vec)
                    hidden_array_2m.append(sV_t2m_vec)

                sV_l1a = torch.cat(hidden_array_1a, dim=1)
                sV_l2a = torch.cat(hidden_array_2a, dim=1)
                sV_l1m = torch.cat(hidden_array_1m, dim=1)
                sV_l2m = torch.cat(hidden_array_2m, dim=1)

                sV_lla = torch.cat((sV_l1a, sV_l2a), dim=2)
                sV_llm = torch.cat((sV_l1m, sV_l2m), dim=2)

                #############################
                # Add Co-memory network
                #############################
                M1 = torch.cat([sV_t1a, sV_t2a], dim=1)
                M2 = torch.cat([sV_t1m, sV_t2m], dim=1)

                ma = M1.detach()
                mb = M2.detach()

                M1 = M1.view(M1.size(0), 1, M1.size(1))
                M2 = M2.view(M2.size(0), 1, M2.size(1))

                for hop in range(3):
                    mm = ma + mb
                    M1 = self.epm1(sV_lla, mm, M1)
                    M2 = self.epm2(sV_llm, mm, M2)
                    M1 = M1.view(M1.size(0), M1.size(2))
                    M2 = M2.view(M2.size(0), M2.size(2))

                    maq = torch.cat([ma, M1, s1_t], dim=1)
                    mbq = torch.cat([mb, M2, s1_t], dim=1)

                    ma = torch.tanh(self.ma_decoder(maq))
                    mb = torch.tanh(self.mb_decoder(mbq))

                M = torch.cat((ma, mb), dim=1)

                #########################
                # decode the final output
                #########################
                out = torch.tanh(self.linear_decoder_count_1(M))
                output = self.linear_decoder_count_2(out)
                outputs_j.append(output)

            # output is the score of each multiple choice
            outputs_j = torch.cat(outputs_j, 1)
            # print('Output_j size', outputs_j.size()) # (1,5)
            outputs.append(outputs_j)

            # for evaluate accuracy, find the max one
            _, mx_idx = torch.max(outputs_j, 1)
            predictions.append(mx_idx)
            # print(outputs_j,mx_idx)

        outputs = torch.cat(outputs, 0)
        predictions = torch.cat(predictions, 0)
        # print outputs.size(), predictions.size()
        return outputs, predictions


    def accuracy(self, logits, targets):
        correct = torch.sum(logits.eq(targets)).float()
        return correct * 100.0 / targets.size(0)

class AttentionTwoStream(nn.Module):

    def __init__(self, opt, feat_channel, feat_dim, text_embed_size, hidden_size, vocab_size, num_layers, word_matrix,
                    answer_vocab_size=None, max_len=20, dropout=0.2, mm_version=1, useSpatial=False, useNaive=False, mrmUseOriginFeat=False):
        """Set the hyper-parameters and build the layers."""
        super(AttentionTwoStream, self).__init__()
                
        self.opt = opt
        # text input size
        self.text_embed_size = text_embed_size # should be 300
        
        # video input size
        self.feat_channel = feat_channel
        self.feat_dim = feat_dim # should be 7
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.useNaive = useNaive
        self.mrmUseOriginFeat = mrmUseOriginFeat
        self.useSpatial = useSpatial
        self.mm_version = mm_version
        
        self.TpAtt_a = TemporalAttentionModule(hidden_size*2, hidden_size)
        self.TpAtt_m = TemporalAttentionModule(hidden_size*2, hidden_size)

        self.video_encoder = nn.Linear(feat_channel, hidden_size)

        self.drop_keep_prob_final_att_vec = nn.Dropout(dropout)
        self.embed = nn.Embedding(vocab_size, text_embed_size)
                    
        self.lstm_text_1 = nn.LSTMCell(text_embed_size, hidden_size)
        self.lstm_text_2 = nn.LSTMCell(hidden_size, hidden_size)

        self.lstm_video_1a = nn.LSTMCell(4096, hidden_size)
        self.lstm_video_2a = nn.LSTMCell(hidden_size, hidden_size)
        
        self.lstm_video_1m = nn.LSTMCell(4096, hidden_size)
        self.lstm_video_2m = nn.LSTMCell(hidden_size, hidden_size)
        
        
        self.dropout = nn.Dropout(0.2)    
                
        if mm_version==1:
            self.lstm_mm_1 = nn.LSTMCell(hidden_size, hidden_size)
            self.lstm_mm_2 = nn.LSTMCell(hidden_size, hidden_size)
            self.linear_decoder_mem = nn.Linear(hidden_size * 2, hidden_size) 
            self.hidden_encoder_1 = nn.Linear(hidden_size * 2, hidden_size) 
            self.hidden_encoder_2 = nn.Linear(hidden_size * 2, hidden_size) 
        
        else:
            self.gru_mm = nn.GRUCell(hidden_size, hidden_size)
            self.linear_decoder_mem = nn.Linear(hidden_size, hidden_size) 
        
        
        self.linear_decoder_att_a = nn.Linear(hidden_size * 2, hidden_size) 
        self.linear_decoder_att_m = nn.Linear(hidden_size * 2, hidden_size) 

        if answer_vocab_size is not None:
            self.linear_decoder_count_2 = nn.Linear(hidden_size * 2 + hidden_size, answer_vocab_size)
        else:
            self.linear_decoder_count_2 = nn.Linear(hidden_size * 2 + hidden_size, 1)    # Count is regression problem
                
        self.max_len = max_len
        
        if mrmUseOriginFeat:
            self.mrm_vid = MemoryRamTwoStreamModule(hidden_size, hidden_size, max_len)
            self.mrm_txt = MemoryRamModule(text_embed_size, hidden_size, max_len)
        else:
            self.mrm_vid = MemoryRamTwoStreamModule(hidden_size, hidden_size, max_len)
            self.mrm_txt = MemoryRamModule(hidden_size, hidden_size, max_len)

        self.init_weights(word_matrix)
        #####################################################################################################################
        if self.opt.pool_type is not "default":
            activation = 'leaky_relu'
            dropout = 0.2
            choices = {
                "default": None,

                "LinearSum": fusions.LinearSum(
                    input_dims=opt.pool_in_dims, output_dim=opt.pool_out_dim, mm_dim=opt.pool_hidden_dim, #1200
                    activ_input=activation, activ_output=activation,
                    dropout_input=dropout, dropout_pre_lin=dropout, dropout_output=dropout
                ),

                "ConcatMLP": fusions.ConcatMLP(
                    input_dims=opt.pool_in_dims, output_dim=opt.pool_out_dim, dimensions=[opt.pool_hidden_dim, opt.pool_hidden_dim],#[500,500]
                    activation=activation, 
                    dropout=dropout
                ),

                "MCB": fusions.MCB(
                    input_dims=opt.pool_in_dims, output_dim=opt.pool_out_dim, mm_dim=opt.pool_hidden_dim, #16000 ## Not usable with pytorch 1.0 or late apparently
                    activ_output=activation, 
                    dropout_output=dropout
                ),

                "MFH": fusions.MFH(
                    input_dims=opt.pool_in_dims, output_dim=opt.pool_out_dim, factor=2, mm_dim=opt.pool_hidden_dim, #1200
                    activ_input=activation, activ_output=activation, 
                    dropout_input=dropout, dropout_pre_lin=dropout, dropout_output=dropout
                ),

                "MFB": fusions.MFB(
                    input_dims=opt.pool_in_dims, output_dim=opt.pool_out_dim, factor=2, mm_dim=opt.pool_hidden_dim, #1200
                    activ_input=activation, activ_output=activation, normalize=True, 
                    dropout_input=dropout, dropout_pre_norm=dropout, dropout_output=dropout
                ),

                "MLB": fusions.MLB(
                    input_dims=opt.pool_in_dims, output_dim=opt.pool_out_dim, mm_dim=opt.pool_hidden_dim, #1200
                    activ_input=activation, activ_output=activation, normalize=True,
                    dropout_input=dropout, dropout_pre_lin=dropout, dropout_output=dropout
                ),

                "Block": fusions.Block(
                    input_dims=opt.pool_in_dims, output_dim=opt.pool_out_dim, mm_dim=opt.pool_hidden_dim, #1600 NO ACITVATIONS IN HERE CURRENTLY
                    dropout_input=dropout, dropout_pre_lin=dropout, dropout_output=dropout
                ),

                "Tucker": fusions.Tucker(
                    input_dims=opt.pool_in_dims, output_dim=opt.pool_out_dim, mm_dim=opt.pool_hidden_dim, #1600 NO ACITVATIONS IN HERE CURRENTLY
                    normalize=True,
                    dropout_input=dropout, dropout_pre_lin=dropout, dropout_output=dropout
                ),

                "BlockTucker": fusions.BlockTucker(
                    input_dims=opt.pool_in_dims, output_dim=opt.pool_out_dim, mm_dim=opt.pool_hidden_dim, #1600 NO ACITVATIONS IN HERE CURRENTLY
                    dropout_input=dropout, dropout_pre_lin=dropout, dropout_output=dropout
                ),
                
                "Mutan": fusions.Mutan(
                    input_dims=opt.pool_in_dims, output_dim=opt.pool_out_dim, mm_dim=opt.pool_hidden_dim, #1600 NO ACITVATIONS IN HERE CURRENTLY
                    normalize=True,
                    dropout_input=dropout, dropout_pre_lin=dropout, dropout_output=dropout
                )
            }
        blp = choices[self.opt.pool_type]
        if useNaive:
            self.mm_att = MultiModalNaiveModule(hidden_size)
        else:
            self.mm_att = MultiModalAttentionModule(opt, blp, hidden_size)

        #####################################################################################################################

        


    def init_weights(self, word_matrix):
        if word_matrix is None:
            self.embed.weight.data.uniform_(-0.1, 0.1)
        else:
            # init embed from glove
            self.embed.weight.data.copy_(torch.from_numpy(word_matrix))
        
        self.mrm_vid.init_weights()
        self.mrm_txt.init_weights()

    def init_hiddens(self):
        s_t = torch.zeros(1, self.hidden_size).cuda()
        s_t2 = torch.zeros(1, self.hidden_size).cuda()
        c_t = torch.zeros(1, self.hidden_size).cuda()
        c_t2 = torch.zeros(1, self.hidden_size).cuda()
        return s_t,s_t2,c_t,c_t2
    
    
    def mm_module_v1(self,svt_tmp,memory_ram_vid,memory_ram_txt,loop=3):
    
        sm_q1,sm_q2,cm_q1,cm_q2 = self.init_hiddens()
        mm_oo = self.drop_keep_prob_final_att_vec(torch.tanh(self.hidden_encoder_1(svt_tmp)))
        
        for _ in range(loop):
        
            sm_q1, cm_q1 = self.lstm_mm_1(mm_oo, (sm_q1, cm_q1))
            sm_q2, cm_q2 = self.lstm_mm_2(sm_q1, (sm_q2, cm_q2))
        
            mm_o1 = self.mm_att(sm_q2,memory_ram_vid,memory_ram_txt)
            mm_o2 = torch.cat((sm_q2,mm_o1),dim=1)
            mm_oo = self.drop_keep_prob_final_att_vec(torch.tanh(self.hidden_encoder_2(mm_o2)))
        
        smq = torch.cat( (sm_q1,sm_q2), dim=1)
        return smq
    
    
    def mm_module_v2(self,memory_ram_vid,memory_ram_txt,loop=5):
        
        h_t = torch.zeros(1, self.hidden_size).cuda()
                
        for _ in range(loop):
            mm_o = self.mm_att(h_t,memory_ram_vid,memory_ram_txt)
            h_t = self.gru_mm(mm_o, h_t)
        
        return h_t

    def forward(self, data_dict):
        ret = self.forward_action(data_dict)
        return ret

    def forward_action(self, data_dict):
        video_features = data_dict['video_features']
        questions, question_lengths = data_dict['question_words'], data_dict['question_lengths']
        num_mult_choices = data_dict['num_mult_choices']
                
        outputs = []
        predictions = []
        batch_size = len(questions)

        
        for j in range(batch_size):
            features_question_j = self.embed(questions[j])
            vgg, c3d = video_features[0][j], video_features[1][j]

            nImg = vgg.shape[0]
            outputs_j = []

            for n_cand in range(num_mult_choices):
                
                #print ' ', n_cand,
                nQuestionWords = question_lengths[j]

                ################################
                # slice the input image features
                ################################
                #feature = video_features[j,video_features.size(1)-nImg:]
                #print('current video feature size', feature.size())            

            
                #############################             
                # run text encoder first time
                #############################
                s1_t1,s1_t2,c1_t1,c1_t2 = self.init_hiddens()
            
                for i in xrange(nQuestionWords):
                    input_question = features_question_j[n_cand,i:i+1]
                    #print input_question.size() # (1,300)
                    s1_t1, c1_t1 = self.lstm_text_1(input_question, (s1_t1, c1_t1))
                    s1_t2, c1_t2 = self.lstm_text_2(s1_t1, (s1_t2, c1_t2))
            
                # here s1_t1, s1_t2 is the last hidden
                s1_t = torch.cat((s1_t1,s1_t2), dim=1)  # should be of size (1,1024)
            

            
                ###########################################             
                # run video encoder with spatial attention
                ###########################################
                sV_t1a,sV_t2a,cV_t1a,cV_t2a = s1_t1,s1_t2,c1_t1,c1_t2
                sV_t1m,sV_t2m,cV_t1m,cV_t2m = s1_t1,s1_t2,c1_t1,c1_t2
            

                # record each time t, hidden states, for later temporal attention after text encoding
                hidden_array_1a = []
                hidden_array_2a = []
                hidden_array_1m = []
                hidden_array_2m = []
            
            
                for i in xrange(nImg):
                    feat_att_m = vgg[i:i+1]
                    feat_att_a = c3d[i:i+1]
                    #print feat_att_a.size(), feat_att_m.size() # (1, 4096) (1, 4096)

                    sV_t1m, cV_t1m = self.lstm_video_1m(feat_att_m, (sV_t1m, cV_t1m))
                    sV_t2m, cV_t2m = self.lstm_video_2m(sV_t1m, (sV_t2m, cV_t2m))

                    sV_t1a, cV_t1a = self.lstm_video_1a(feat_att_a, (sV_t1a, cV_t1a))
                    sV_t2a, cV_t2a = self.lstm_video_2a(sV_t1a, (sV_t2a, cV_t2a))
                
                
                    sV_t1a_vec = sV_t1a.view(sV_t1a.size(0),1,sV_t1a.size(1))
                    sV_t2a_vec = sV_t2a.view(sV_t2a.size(0),1,sV_t2a.size(1))
                
                    hidden_array_1a.append(sV_t1a_vec)
                    hidden_array_2a.append(sV_t2a_vec)
                
                    sV_t1m_vec = sV_t1m.view(sV_t1m.size(0),1,sV_t1m.size(1))
                    sV_t2m_vec = sV_t2m.view(sV_t2m.size(0),1,sV_t2m.size(1))
                
                    hidden_array_1m.append(sV_t1m_vec)
                    hidden_array_2m.append(sV_t2m_vec)
                
                

                sV_l1a = torch.cat(hidden_array_1a, dim=1)
                sV_l2a = torch.cat(hidden_array_2a, dim=1)
                sV_l1m = torch.cat(hidden_array_1m, dim=1)
                sV_l2m = torch.cat(hidden_array_2m, dim=1)
            
                sV_lla = torch.cat((sV_l1a,sV_l2a), dim=2)
                sV_llm = torch.cat((sV_l1m,sV_l2m), dim=2)
                

            
                #############################             
                # run text encoder second time
                #############################
                sT_t1,sT_t2,cT_t1,cT_t2 = self.init_hiddens()
                sT_t1,sT_t2 = sV_t1a+sV_t1m, sV_t2a+sV_t2m
            
                hidden_array_3 = []
                
                for i in xrange(nQuestionWords):
                    input_question = features_question_j[n_cand,i:i+1]
                    sT_t1, cT_t1 = self.lstm_text_1(input_question, (sT_t1, cT_t1))
                    sT_t2, cT_t2 = self.lstm_text_2(sT_t1, (sT_t2, cT_t2))
                    hidden_array_3.append(sT_t2)
                    
                #print('Text encoding One size', sT_t1.size(), sT_t2.size())

                # here sT_t1, sT_t2 is the last hidden
                sT_t = torch.cat( (sT_t1, sT_t2), dim=1)  # should be of size (1,1024)
                
                #####################
                # temporal attention
                #####################
                vid_att_a = self.TpAtt_a(sV_lla, sT_t)
                vid_att_m = self.TpAtt_m(sV_llm, sT_t)
            

                ################
                # ram memory
                ################
                sT_rl = torch.cat(hidden_array_3, dim=0)

                memory_ram_vid = self.mrm_vid(sV_l2a[0,:,:], sV_l2m[0,:,:], nImg)
                memory_ram_txt = self.mrm_txt(sT_rl, nQuestionWords)

                if self.mm_version==1:
                    svt_tmp = torch.cat((sV_t2a,sV_t2m),dim=1)
                    smq = self.mm_module_v1(svt_tmp,memory_ram_vid,memory_ram_txt)
                elif self.mm_version==2:
                    smq = self.mm_module_v2(memory_ram_vid,memory_ram_txt)


                ######################### 
                # decode the final output
                ######################### 
                
                final_embed_a = torch.tanh(self.linear_decoder_att_a(vid_att_a))
                final_embed_m = torch.tanh(self.linear_decoder_att_m(vid_att_m))
                final_embed_2 = torch.tanh(self.linear_decoder_mem(smq))
                final_embed = torch.cat([final_embed_a, final_embed_m, final_embed_2], dim=1)
            
                
                output = self.linear_decoder_count_2(final_embed)                
                outputs_j.append(output)
        
            
            # output is the score of each multiple choice
            outputs_j = torch.cat(outputs_j,1)
            #print('Output_j size', outputs_j.size()) # (1,5)
            outputs.append(outputs_j)
            
            # for evaluate accuracy, find the max one
            _,mx_idx = torch.max(outputs_j, 1)
            predictions.append(mx_idx)
            #print(outputs_j,mx_idx)
            
        outputs = torch.cat(outputs, 0)
        predictions = torch.cat(predictions, 0)
        #print outputs.size(), predictions.size()
        return outputs, predictions
           
           
    
    def accuracy(self, logits, targets):
        #print(logits.size(), targets.size(), logits.type(), targets.type())
        #print(logits,targets)
        #targets = targets.int()
        correct = torch.sum(logits.eq(targets)).float()
        #print(correct,targets.size(0))
        #print(correct, targets.size(0), correct * 100.0 / targets.size(0))
        return correct * 100.0 / targets.size(0)





################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
class TGIFAttentionTwoStream(nn.Module):

    def __init__(self, opt, task, feat_channel, feat_dim, text_embed_size, hidden_size, vocab_size, num_layers, word_matrix,
                    answer_vocab_size=None, max_len=20, dropout=0.2, mm_version=1, useSpatial=False, useNaive=False, iter_num=3):
        """Set the hyper-parameters and build the layers."""
        super(TGIFAttentionTwoStream, self).__init__()
        
        self.task = task
        
        self.opt = opt
        # text input size
        self.text_embed_size = text_embed_size # should be 300
        
        # video input size
        self.feat_channel = feat_channel
        self.feat_dim = feat_dim # should be 7
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.useNaive = useNaive
        self.useSpatial = useSpatial
        self.mm_version = mm_version
        
        self.TpAtt_a = TemporalAttentionModule(hidden_size*2, hidden_size)
        self.TpAtt_m = TemporalAttentionModule(hidden_size*2, hidden_size)

        if useSpatial:
            self.SpAtt = SpatialAttentionModule(feat_channel, feat_dim, hidden_size)
        else:
            self.video_encoder = nn.Linear(feat_channel, hidden_size) 

        self.drop_keep_prob_final_att_vec = nn.Dropout(dropout)
        self.embed = nn.Embedding(vocab_size, text_embed_size)
        # ####
        # self.bert = BertModel.from_pretrained('bert-base-uncased')
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        # self.bert_fc = nn.Sequential(
        #             nn.Dropout(0.5),
        #             nn.Linear(768, 300),
        #             nn.Tanh(),
        #         )
        # ####
                    
        self.lstm_text_1 = nn.LSTMCell(text_embed_size, hidden_size)
        self.lstm_text_2 = nn.LSTMCell(hidden_size, hidden_size)

        self.lstm_video_1a = nn.LSTMCell(2048, hidden_size)
        self.lstm_video_2a = nn.LSTMCell(hidden_size, hidden_size)
        
        self.lstm_video_1m = nn.LSTMCell(4096, hidden_size)
        self.lstm_video_2m = nn.LSTMCell(hidden_size, hidden_size)
        
        self.iter_num = iter_num
        
        if mm_version==1:
            self.lstm_mm_1 = nn.LSTMCell(hidden_size, hidden_size)
            self.lstm_mm_2 = nn.LSTMCell(hidden_size, hidden_size)
            self.linear_decoder_mem = nn.Linear(hidden_size * 2, hidden_size) 
            self.hidden_encoder_1 = nn.Linear(hidden_size * 2, hidden_size) 
            self.hidden_encoder_2 = nn.Linear(hidden_size * 2, hidden_size) 
        
        else:
            self.gru_mm = nn.GRUCell(hidden_size, hidden_size)
            self.linear_decoder_mem = nn.Linear(hidden_size, hidden_size) 
        

        self.linear_decoder_att_a = nn.Linear(hidden_size * 2, hidden_size) 
        self.linear_decoder_att_m = nn.Linear(hidden_size * 2, hidden_size) 

        
        if answer_vocab_size is not None:
            self.linear_decoder_count_2 = nn.Linear(hidden_size * 2 + hidden_size, answer_vocab_size)
        else:
            self.linear_decoder_count_2 = nn.Linear(hidden_size * 2 + hidden_size, 1)    # Count is regression problem
                
        self.max_len = max_len


        self.mrm_vid = MemoryRamTwoStreamModule(hidden_size, hidden_size, max_len)
        self.mrm_txt = MemoryRamModule(hidden_size, hidden_size, max_len)

        
        self.init_weights(word_matrix)
        #####################################################################################################################
        if self.opt.pool_type is not "default":
            activation = 'leaky_relu'
            dropout = 0.2
            choices = {
                "default": None,

                "LinearSum": fusions.LinearSum(
                    input_dims=opt.pool_in_dims, output_dim=opt.pool_out_dim, mm_dim=opt.pool_hidden_dim, #1200
                    activ_input=activation, activ_output=activation,
                    dropout_input=dropout, dropout_pre_lin=dropout, dropout_output=dropout
                ),

                "ConcatMLP": fusions.ConcatMLP(
                    input_dims=opt.pool_in_dims, output_dim=opt.pool_out_dim, dimensions=[opt.pool_hidden_dim, opt.pool_hidden_dim],#[500,500]
                    activation=activation, 
                    dropout=dropout
                ),

                "MCB": fusions.MCB(
                    input_dims=opt.pool_in_dims, output_dim=opt.pool_out_dim, mm_dim=opt.pool_hidden_dim, #16000 ## Not usable with pytorch 1.0 or late apparently
                    activ_output=activation, 
                    dropout_output=dropout
                ),

                "MFH": fusions.MFH(
                    input_dims=opt.pool_in_dims, output_dim=opt.pool_out_dim, factor=2, mm_dim=opt.pool_hidden_dim, #1200
                    activ_input=activation, activ_output=activation, 
                    dropout_input=dropout, dropout_pre_lin=dropout, dropout_output=dropout
                ),

                "MFB": fusions.MFB(
                    input_dims=opt.pool_in_dims, output_dim=opt.pool_out_dim, factor=2, mm_dim=opt.pool_hidden_dim, #1200
                    activ_input=activation, activ_output=activation, normalize=True, 
                    dropout_input=dropout, dropout_pre_norm=dropout, dropout_output=dropout
                ),

                "MLB": fusions.MLB(
                    input_dims=opt.pool_in_dims, output_dim=opt.pool_out_dim, mm_dim=opt.pool_hidden_dim, #1200
                    activ_input=activation, activ_output=activation, normalize=True,
                    dropout_input=dropout, dropout_pre_lin=dropout, dropout_output=dropout
                ),

                "Block": fusions.Block(
                    input_dims=opt.pool_in_dims, output_dim=opt.pool_out_dim, mm_dim=opt.pool_hidden_dim, #1600 NO ACITVATIONS IN HERE CURRENTLY
                    dropout_input=dropout, dropout_pre_lin=dropout, dropout_output=dropout
                ),

                "Tucker": fusions.Tucker(
                    input_dims=opt.pool_in_dims, output_dim=opt.pool_out_dim, mm_dim=opt.pool_hidden_dim, #1600 NO ACITVATIONS IN HERE CURRENTLY
                    normalize=True,
                    dropout_input=dropout, dropout_pre_lin=dropout, dropout_output=dropout
                ),

                "BlockTucker": fusions.BlockTucker(
                    input_dims=opt.pool_in_dims, output_dim=opt.pool_out_dim, mm_dim=opt.pool_hidden_dim, #1600 NO ACITVATIONS IN HERE CURRENTLY
                    dropout_input=dropout, dropout_pre_lin=dropout, dropout_output=dropout
                ),
                
                "Mutan": fusions.Mutan(
                    input_dims=opt.pool_in_dims, output_dim=opt.pool_out_dim, mm_dim=opt.pool_hidden_dim, #1600 NO ACITVATIONS IN HERE CURRENTLY
                    normalize=True,
                    dropout_input=dropout, dropout_pre_lin=dropout, dropout_output=dropout
                )
            }
        blp = choices[self.opt.pool_type]
        self.mm_att = MultiModalAttentionModule(opt, blp, hidden_size)

        #####################################################################################################################
        


    def init_weights(self, word_matrix):
        """Initialize weights."""
        
        if word_matrix is None:
            self.embed.weight.data.uniform_(-0.1, 0.1)
        else:
            # init embed from glove
            self.embed.weight.data.copy_(torch.from_numpy(word_matrix))
        
        self.mrm_vid.init_weights()
        self.mrm_txt.init_weights()

    def init_hiddens(self):
        s_t = torch.zeros(1, self.hidden_size).cuda()
        s_t2 = torch.zeros(1, self.hidden_size).cuda()
        c_t = torch.zeros(1, self.hidden_size).cuda()
        c_t2 = torch.zeros(1, self.hidden_size).cuda()
        return s_t,s_t2,c_t,c_t2
    
    
    def mm_module_v1(self,svt_tmp,memory_ram_vid,memory_ram_txt,loop=3):

        sm_q1,sm_q2,cm_q1,cm_q2 = self.init_hiddens()
        mm_oo = self.drop_keep_prob_final_att_vec(torch.tanh(self.hidden_encoder_1(svt_tmp)))
        
        for _ in range(loop):
            
            sm_q1, cm_q1 = self.lstm_mm_1(mm_oo, (sm_q1, cm_q1))
            sm_q2, cm_q2 = self.lstm_mm_2(sm_q1, (sm_q2, cm_q2))
        
            mm_o1 = self.mm_att(sm_q2,memory_ram_vid,memory_ram_txt)
            mm_o2 = torch.cat((sm_q2,mm_o1),dim=1)
            mm_oo = self.drop_keep_prob_final_att_vec(torch.tanh(self.hidden_encoder_2(mm_o2)))
        
        smq = torch.cat( (sm_q1,sm_q2), dim=1)
        return smq
    
    
    def mm_module_v2(self,memory_ram_vid,memory_ram_txt,loop=5):
        
        h_t = torch.zeros(1, self.hidden_size).cuda()
                
        for _ in range(loop):
            mm_o = self.mm_att(h_t,memory_ram_vid,memory_ram_txt)
            h_t = self.gru_mm(mm_o, h_t)
        
        return h_t
    

    def forward(self, data_dict, question_type='Count'):
        if question_type=='Count':
            ret = self.forward_count(data_dict)
        elif question_type=='Action':
            ret = self.forward_action(data_dict)
        elif question_type=='Trans':
            ret = self.forward_trans(data_dict)
        else:
            assert question_type=='FrameQA'
            ret = self.forward_frameqa(data_dict)

        return ret

    def forward_count(self, data_dict):
        
        video_features, numImgs = data_dict['video_features'], data_dict['video_lengths'],
        questions, question_lengths = data_dict['question_words'], data_dict['question_lengths']
        answers = data_dict['answers']           
                
        outputs = []
        predictions = []
        bsize = len(questions)
        batch_size = len(questions)  # batch size has to be 1


        cnt = 0
        features_questions = self.embed(questions)
        
        for j in range(batch_size):
        
            nImg = numImgs[j]
            nQuestionWords = question_lengths[j]

            ################################
            # slice the input image features
            ################################
            feature = video_features[j,video_features.size(1)-nImg:]
            #print('current video feature size', feature.size())            

            
            #############################             
            # run text encoder first time
            #############################
            s1_t1,s1_t2,c1_t1,c1_t2 = self.init_hiddens()
            
            for i in xrange(nQuestionWords):
                input_question = features_questions[j,i:i+1]
                s1_t1, c1_t1 = self.lstm_text_1(input_question, (s1_t1, c1_t1))
                s1_t2, c1_t2 = self.lstm_text_2(s1_t1, (s1_t2, c1_t2))
            
            # here s1_t1, s1_t2 is the last hidden
            s1_t = torch.cat( (s1_t1,s1_t2), dim=1)  # should be of size (1,1024)
            
            
            
            
            ###########################################             
            # run video encoder with spatial attention
            ###########################################
            sV_t1a,sV_t2a,cV_t1a,cV_t2a = s1_t1,s1_t2,c1_t1,c1_t2
            sV_t1m,sV_t2m,cV_t1m,cV_t2m = s1_t1,s1_t2,c1_t1,c1_t2
  
                        
            
            # record each time t, hidden states, for later temporal attention after text encoding
            hidden_array_1a = []
            hidden_array_2a = []
            hidden_array_1m = []
            hidden_array_2m = []
            
            
            for i in xrange(nImg):
                
                if self.useSpatial:
                    input_frame = feature[i:i+1]
                    feat_att = self.SpAtt(input_frame, s1_t)
                else:
                    feat_att_m = feature[i:i+1,0,0,:4096]
                    feat_att_a = feature[i:i+1,0,0,4096:]
                
                sV_t1m, cV_t1m = self.lstm_video_1m(feat_att_m, (sV_t1m, cV_t1m))
                sV_t2m, cV_t2m = self.lstm_video_2m(sV_t1m, (sV_t2m, cV_t2m))

                sV_t1a, cV_t1a = self.lstm_video_1a(feat_att_a, (sV_t1a, cV_t1a))
                sV_t2a, cV_t2a = self.lstm_video_2a(sV_t1a, (sV_t2a, cV_t2a))
                
                
                sV_t1a_vec = sV_t1a.view(sV_t1a.size(0),1,sV_t1a.size(1))
                sV_t2a_vec = sV_t2a.view(sV_t2a.size(0),1,sV_t2a.size(1))
                
                hidden_array_1a.append(sV_t1a_vec)
                hidden_array_2a.append(sV_t2a_vec)
                
                sV_t1m_vec = sV_t1m.view(sV_t1m.size(0),1,sV_t1m.size(1))
                sV_t2m_vec = sV_t2m.view(sV_t2m.size(0),1,sV_t2m.size(1))
                
                hidden_array_1m.append(sV_t1m_vec)
                hidden_array_2m.append(sV_t2m_vec)
                

            # assume sV_t1 is of size (1,1,hidden)
            sV_l1a = torch.cat(hidden_array_1a, dim=1)
            sV_l2a = torch.cat(hidden_array_2a, dim=1)
            sV_l1m = torch.cat(hidden_array_1m, dim=1)
            sV_l2m = torch.cat(hidden_array_2m, dim=1)
            
            sV_lla = torch.cat((sV_l1a,sV_l2a), dim=2)
            sV_llm = torch.cat((sV_l1m,sV_l2m), dim=2)



            
            #############################             
            # run text encoder second time
            #############################
            sT_t1,sT_t2,cT_t1,cT_t2 = self.init_hiddens()
            sT_t1,sT_t2 = sV_t1a+sV_t1m, sV_t2a+sV_t2m
            
            hidden_array_3 = []

            # here sT_t1, sT_t2 are the last hiddens from video, input to text encoder again
            for i in xrange(nQuestionWords):
                input_question = features_questions[j,i:i+1]
                sT_t1, cT_t1 = self.lstm_text_1(input_question, (sT_t1, cT_t1))
                sT_t2, cT_t2 = self.lstm_text_2(sT_t1, (sT_t2, cT_t2))
                hidden_array_3.append(sT_t2)
            #print('Text encoding One size', sT_t1.size(), sT_t2.size())

            # here sT_t1, sT_t2 is the last hidden
            sT_t = torch.cat( (sT_t1,sT_t2), dim=1)  # should be of size (1,1024)
            
            #####################
            # temporal attention
            #####################
            vid_att_a = self.TpAtt_a(sV_lla, sT_t)
            vid_att_m = self.TpAtt_m(sV_llm, sT_t)
            
            
            ################
            # ram memory
            ################
            sT_rl = torch.cat(hidden_array_3, dim=0)

            memory_ram_vid = self.mrm_vid(sV_l2a[0,:,:], sV_l2m[0,:,:], nImg)
            memory_ram_txt = self.mrm_txt(sT_rl, nQuestionWords)
                
            if self.mm_version==1:
                svt_tmp = torch.cat((sV_t2a,sV_t2m),dim=1)
                smq = self.mm_module_v1(svt_tmp,memory_ram_vid,memory_ram_txt,self.iter_num)
            elif self.mm_version==2:
                smq = self.mm_module_v2(memory_ram_vid,memory_ram_txt)
                
                
            ######################### 
            # decode the final output
            #########################
            final_embed_a = torch.tanh( self.linear_decoder_att_a(vid_att_a) )
            final_embed_m = torch.tanh( self.linear_decoder_att_m(vid_att_m) )
            final_embed_2 = torch.tanh( self.linear_decoder_mem(smq) )
            final_embed = torch.cat([final_embed_a,final_embed_m,final_embed_2],dim=1)
            
            output = self.linear_decoder_count_2(final_embed)
            prediction = torch.clamp(torch.round(output.detach()), min=1, max=10).int()
            
            outputs.append(output)
            predictions.append(prediction)

    
        outputs = torch.cat(outputs, 0)
        #targets = torch.cat(targets, 0)
        predictions = torch.cat(predictions, 0)
        #print(predictions.size())
        return outputs, answers, predictions
    
    def forward_action(self, data_dict):

        video_features, numImgs = data_dict['video_features'], data_dict['video_lengths'],
        questions, question_lengths = data_dict['candidates'], data_dict['candidate_lengths']
        answers, num_mult_choices = data_dict['answers'], data_dict['num_mult_choices']
                
        outputs = []
        predictions = []
        #print(questions.size())   # (N, 5, 35), 5 multiple choices, each choice is a question of 35 max lengths
        batch_size = len(questions)  

        cnt = 0
        ########
        #Flatten the batch and 5 candidates, process them with bert and FC layer and then recover them
        features_questions = self.embed(questions)
        # questions = questions.view(-1, questions.shape[-1])
        # features_questions = self.bert(questions)[0][11]    # Final bert layer
        # features_questions = self.bert_fc(features_questions)
        # features_questions = features_questions.view(batch_size, 5, questions.shape[-1], 300)
        ########

        
        for j in range(batch_size):
        
            nImg = numImgs[j]
            outputs_j = []
            predictions_j = []
            
            for n_cand in range(num_mult_choices):
                
                #print ' ', n_cand,
                nQuestionWords = question_lengths[j][n_cand]
                #nAnwserWords = candidate_lengths[j][n_cand]
                

                ################################
                # slice the input image features
                ################################
                feature = video_features[j,video_features.size(1)-nImg:]
                #print('current video feature size', feature.size())            

            
                #############################             
                # run text encoder first time
                #############################
                s1_t1,s1_t2,c1_t1,c1_t2 = self.init_hiddens()
            
                for i in xrange(nQuestionWords):
                    input_question = features_questions[j,n_cand,i:i+1]
                    s1_t1, c1_t1 = self.lstm_text_1(input_question, (s1_t1, c1_t1))
                    s1_t2, c1_t2 = self.lstm_text_2(s1_t1, (s1_t2, c1_t2))
            
                # here s1_t1, s1_t2 is the last hidden
                s1_t = torch.cat( (s1_t1,s1_t2), dim=1)  # should be of size (1,1024)
            
                
            
                ###########################################             
                # run video encoder with spatial attention
                ###########################################
                sV_t1a,sV_t2a,cV_t1a,cV_t2a = s1_t1,s1_t2,c1_t1,c1_t2
                sV_t1m,sV_t2m,cV_t1m,cV_t2m = s1_t1,s1_t2,c1_t1,c1_t2
            

                # record each time t, hidden states, for later temporal attention after text encoding
                hidden_array_1a = []
                hidden_array_2a = []
                hidden_array_1m = []
                hidden_array_2m = []
            
            
                for i in xrange(nImg):               
                    if self.useSpatial:
                        input_frame = feature[i:i+1]
                        feat_att = self.SpAtt(input_frame, s1_t)
                    else:
                        feat_att_m = feature[i:i+1,0,0,:4096]
                        feat_att_a = feature[i:i+1,0,0,4096:]

                    sV_t1m, cV_t1m = self.lstm_video_1m(feat_att_m, (sV_t1m, cV_t1m))
                    sV_t2m, cV_t2m = self.lstm_video_2m(sV_t1m, (sV_t2m, cV_t2m))

                    sV_t1a, cV_t1a = self.lstm_video_1a(feat_att_a, (sV_t1a, cV_t1a))
                    sV_t2a, cV_t2a = self.lstm_video_2a(sV_t1a, (sV_t2a, cV_t2a))
                
                
                    sV_t1a_vec = sV_t1a.view(sV_t1a.size(0),1,sV_t1a.size(1))
                    sV_t2a_vec = sV_t2a.view(sV_t2a.size(0),1,sV_t2a.size(1))
                
                    hidden_array_1a.append(sV_t1a_vec)
                    hidden_array_2a.append(sV_t2a_vec)
                
                    sV_t1m_vec = sV_t1m.view(sV_t1m.size(0),1,sV_t1m.size(1))
                    sV_t2m_vec = sV_t2m.view(sV_t2m.size(0),1,sV_t2m.size(1))
                
                    hidden_array_1m.append(sV_t1m_vec)
                    hidden_array_2m.append(sV_t2m_vec)
                
                

                sV_l1a = torch.cat(hidden_array_1a, dim=1)
                sV_l2a = torch.cat(hidden_array_2a, dim=1)
                sV_l1m = torch.cat(hidden_array_1m, dim=1)
                sV_l2m = torch.cat(hidden_array_2m, dim=1)
            
                sV_lla = torch.cat((sV_l1a,sV_l2a), dim=2)
                sV_llm = torch.cat((sV_l1m,sV_l2m), dim=2)
                

            
                #############################             
                # run text encoder second time
                #############################
                sT_t1,sT_t2,cT_t1,cT_t2 = self.init_hiddens()
                sT_t1,sT_t2 = sV_t1a+sV_t1m, sV_t2a+sV_t2m
            
                hidden_array_3 = []
                
                for i in xrange(nQuestionWords):
                    input_question = features_questions[j,n_cand,i:i+1]
                    sT_t1, cT_t1 = self.lstm_text_1(input_question, (sT_t1, cT_t1))
                    sT_t2, cT_t2 = self.lstm_text_2(sT_t1, (sT_t2, cT_t2))
                    hidden_array_3.append(sT_t2)
                    
                #print('Text encoding One size', sT_t1.size(), sT_t2.size())

                # here sT_t1, sT_t2 is the last hidden
                sT_t = torch.cat( (sT_t1,sT_t2), dim=1)  # should be of size (1,1024)
                
                #####################
                # temporal attention
                #####################
                vid_att_a = self.TpAtt_a(sV_lla, sT_t)
                vid_att_m = self.TpAtt_m(sV_llm, sT_t)
            

                ################
                # ram memory
                ################
                sT_rl = torch.cat(hidden_array_3, dim=0)

                memory_ram_vid = self.mrm_vid(sV_l2a[0,:,:], sV_l2m[0,:,:], nImg)
                memory_ram_txt = self.mrm_txt(sT_rl, nQuestionWords)

                if self.mm_version==1:
                    svt_tmp = torch.cat((sV_t2a,sV_t2m),dim=1)
                    smq = self.mm_module_v1(svt_tmp,memory_ram_vid,memory_ram_txt,self.iter_num)
                elif self.mm_version==2:
                    smq = self.mm_module_v2(memory_ram_vid,memory_ram_txt)
                
                
                ######################### 
                # decode the final output
                ######################### 
                
                final_embed_a = torch.tanh( self.linear_decoder_att_a(vid_att_a) )
                final_embed_m = torch.tanh( self.linear_decoder_att_m(vid_att_m) )
                final_embed_2 = torch.tanh( self.linear_decoder_mem(smq) )
                final_embed = torch.cat([final_embed_a,final_embed_m,final_embed_2],dim=1)
            
                
                output = self.linear_decoder_count_2(final_embed)                
                outputs_j.append(output)
        
            
            # output is the score of each multiple choice
            outputs_j = torch.cat(outputs_j,1)
            #print('Output_j size', outputs_j.size()) # (1,5)
            outputs.append(outputs_j)
            
            # for evaluate accuracy, find the max one
            _,mx_idx = torch.max(outputs_j,1)
            predictions.append(mx_idx)
            #print(outputs_j,mx_idx)
            
        outputs = torch.cat(outputs, 0)
        predictions = torch.cat(predictions, 0)
        return outputs, answers, predictions
           
    
    def forward_trans(self, data_dict):

        
        video_features, numImgs = data_dict['video_features'], data_dict['video_lengths'],
        questions, question_lengths = data_dict['candidates'], data_dict['candidate_lengths']
        answers, num_mult_choices = data_dict['answers'], data_dict['num_mult_choices']
                    
        
        outputs = []
        predictions = []
        #print(questions.size())   # (N, 5, 35), 5 multiple choices, each choice is a question of 35 max lengths
        batch_size = len(questions)  

        cnt = 0        
        features_questions = self.embed(questions)

        
        
        for j in range(batch_size):
            nImg = numImgs[j]
            outputs_j = []
            predictions_j = []
            
            for n_cand in range(num_mult_choices):
                
                nQuestionWords = question_lengths[j][n_cand]
                #nAnwserWords = answer_lengths[j]

                ################################
                # slice the input image features
                ################################
                feature = video_features[j,video_features.size(1)-nImg:]
                #print('current video feature size', feature.size())     
            
                #############################             
                # run text encoder first time
                #############################
                s1_t1,s1_t2,c1_t1,c1_t2 = self.init_hiddens()
            
                for i in xrange(nQuestionWords):
                    input_question = features_questions[j,n_cand,i:i+1]
                    s1_t1, c1_t1 = self.lstm_text_1(input_question, (s1_t1, c1_t1))
                    s1_t2, c1_t2 = self.lstm_text_2(s1_t1, (s1_t2, c1_t2))
            
                # here s1_t1, s1_t2 is the last hidden
                s1_t = torch.cat( (s1_t1,s1_t2), dim=1)  # should be of size (1,1024)
            
                
            
                ###########################################             
                # run video encoder with spatial attention
                ###########################################
                sV_t1a,sV_t2a,cV_t1a,cV_t2a = s1_t1,s1_t2,c1_t1,c1_t2
                sV_t1m,sV_t2m,cV_t1m,cV_t2m = s1_t1,s1_t2,c1_t1,c1_t2
            
                # record each time t, hidden states, for later temporal attention after text encoding
                hidden_array_1a = []
                hidden_array_2a = []
                hidden_array_1m = []
                hidden_array_2m = []
            
                for i in xrange(nImg):
                    if self.useSpatial:
                        input_frame = feature[i:i+1]
                        feat_att = self.SpAtt(input_frame, s1_t)
                    else:
                        feat_att_m = feature[i:i+1,0,0,:4096]
                        feat_att_a = feature[i:i+1,0,0,4096:]
                        
                                                
                    # lstm
                    sV_t1m, cV_t1m = self.lstm_video_1m(feat_att_m, (sV_t1m, cV_t1m))
                    sV_t2m, cV_t2m = self.lstm_video_2m(sV_t1m, (sV_t2m, cV_t2m))

                    sV_t1a, cV_t1a = self.lstm_video_1a(feat_att_a, (sV_t1a, cV_t1a))
                    sV_t2a, cV_t2a = self.lstm_video_2a(sV_t1a, (sV_t2a, cV_t2a))
                
                    sV_t1a_vec = sV_t1a.view(sV_t1a.size(0),1,sV_t1a.size(1))
                    sV_t2a_vec = sV_t2a.view(sV_t2a.size(0),1,sV_t2a.size(1))
                
                    hidden_array_1a.append(sV_t1a_vec)
                    hidden_array_2a.append(sV_t2a_vec)
                
                    sV_t1m_vec = sV_t1m.view(sV_t1m.size(0),1,sV_t1m.size(1))
                    sV_t2m_vec = sV_t2m.view(sV_t2m.size(0),1,sV_t2m.size(1))
                
                    hidden_array_1m.append(sV_t1m_vec)
                    hidden_array_2m.append(sV_t2m_vec)
                    
                
                sV_l1a = torch.cat(hidden_array_1a, dim=1)
                sV_l2a = torch.cat(hidden_array_2a, dim=1)
                sV_l1m = torch.cat(hidden_array_1m, dim=1)
                sV_l2m = torch.cat(hidden_array_2m, dim=1)
            
                sV_lla = torch.cat((sV_l1a,sV_l2a), dim=2)
                sV_llm = torch.cat((sV_l1m,sV_l2m), dim=2)

            
                #############################             
                # run text encoder second time
                #############################
                sT_t1,sT_t2,cT_t1,cT_t2 = self.init_hiddens()
                sT_t1,sT_t2 = sV_t1a+sV_t1m, sV_t2a+sV_t2m
                
                hidden_array_3 = []
                
                for i in xrange(nQuestionWords):
                    input_question = features_questions[j,n_cand,i:i+1]
                    sT_t1, cT_t1 = self.lstm_text_1(input_question, (sT_t1, cT_t1))
                    sT_t2, cT_t2 = self.lstm_text_2(sT_t1, (sT_t2, cT_t2))
                    hidden_array_3.append(sT_t2)
                    
                #print('Text encoding One size', sT_t1.size(), sT_t2.size())

                # here sT_t1, sT_t2 is the last hidden
                sT_t = torch.cat( (sT_t1,sT_t2), dim=1)  # should be of size (1,1024)
                
                #####################
                # temporal attention
                #####################
                vid_att_a = self.TpAtt_a(sV_lla, sT_t)
                vid_att_m = self.TpAtt_m(sV_llm, sT_t)


                ################
                # stack memory
                ################
                sT_rl = torch.cat(hidden_array_3, dim=0)

                memory_ram_vid = self.mrm_vid(sV_l2a[0,:,:], sV_l2m[0,:,:], nImg)
                memory_ram_txt = self.mrm_txt(sT_rl, nQuestionWords)

                if self.mm_version==1:
                    svt_tmp = torch.cat((sV_t2a,sV_t2m),dim=1)
                    smq = self.mm_module_v1(svt_tmp,memory_ram_vid,memory_ram_txt,self.iter_num)
                elif self.mm_version==2:
                    smq = self.mm_module_v2(memory_ram_vid,memory_ram_txt)

                                
                ######################### 
                # decode the final output
                ######################### 

                final_embed_a = torch.tanh( self.linear_decoder_att_a(vid_att_a) )
                final_embed_m = torch.tanh( self.linear_decoder_att_m(vid_att_m) )
                final_embed_2 = torch.tanh( self.linear_decoder_mem(smq) )
                final_embed = torch.cat([final_embed_a,final_embed_m,final_embed_2],dim=1)
            

                output = self.linear_decoder_count_2(final_embed)                
                outputs_j.append(output)
        
            
            # output is the score of each multiple choice
            outputs_j = torch.cat(outputs_j,1)
            #print('Output_j size', outputs_j.size()) # (1,5)
            outputs.append(outputs_j)
            
            # for evaluate accuracy, find the max one
            _,mx_idx = torch.max(outputs_j,1)
            predictions.append(mx_idx)
            #print(outputs_j,mx_idx)
            
        outputs = torch.cat(outputs, 0)
        predictions = torch.cat(predictions, 0)
        return outputs, answers, predictions
    
    
    def forward_frameqa(self, data_dict):
        
        video_features, numImgs = data_dict['video_features'], data_dict['video_lengths'],
        questions, question_lengths = data_dict['question_words'], data_dict['question_lengths']
        answers = data_dict['answers']           
                                        
        
        outputs = []
        predictions = []
        bsize = len(questions)
        batch_size = len(questions)  # batch size has to be 1


        cnt = 0
        
        features_questions = self.embed(questions)
        # (64, 35, 300)
        #print('text feature size', features_questions.size())
        
        
        for j in range(batch_size):
        
            nImg = numImgs[j]
            nQuestionWords = question_lengths[j]

            ################################
            # slice the input image features
            ################################
            feature = video_features[j,video_features.size(1)-nImg:]
            #print('current video feature size', feature.size())            

            
            #############################             
            # run text encoder first time
            #############################
            s1_t1,s1_t2,c1_t1,c1_t2 = self.init_hiddens()
            
            for i in xrange(nQuestionWords):
                input_question = features_questions[j,i:i+1]
                s1_t1, c1_t1 = self.lstm_text_1(input_question, (s1_t1, c1_t1))
                s1_t2, c1_t2 = self.lstm_text_2(s1_t1, (s1_t2, c1_t2))
            
            # here s1_t1, s1_t2 is the last hidden
            s1_t = torch.cat( (s1_t1,s1_t2), dim=1)  # should be of size (1,1024)
            
            
            
            ###########################################             
            # run video encoder with spatial attention
            ###########################################
            sV_t1a,sV_t2a,cV_t1a,cV_t2a = s1_t1,s1_t2,c1_t1,c1_t2
            sV_t1m,sV_t2m,cV_t1m,cV_t2m = s1_t1,s1_t2,c1_t1,c1_t2

            # record each time t, hidden states, for later temporal attention after text encoding
            hidden_array_1a = []
            hidden_array_2a = []
            hidden_array_1m = []
            hidden_array_2m = []
            
            for i in xrange(nImg):

                if self.useSpatial:
                    input_frame = feature[i:i+1]
                    feat_att = self.SpAtt(input_frame, s1_t)
                else:
                    feat_att_m = feature[i:i+1,0,0,:4096]
                    feat_att_a = feature[i:i+1,0,0,4096:]
                
                sV_t1m, cV_t1m = self.lstm_video_1m(feat_att_m, (sV_t1m, cV_t1m))
                sV_t2m, cV_t2m = self.lstm_video_2m(sV_t1m, (sV_t2m, cV_t2m))

                sV_t1a, cV_t1a = self.lstm_video_1a(feat_att_a, (sV_t1a, cV_t1a))
                sV_t2a, cV_t2a = self.lstm_video_2a(sV_t1a, (sV_t2a, cV_t2a))
                
                sV_t1a_vec = sV_t1a.view(sV_t1a.size(0),1,sV_t1a.size(1))
                sV_t2a_vec = sV_t2a.view(sV_t2a.size(0),1,sV_t2a.size(1))
            
                hidden_array_1a.append(sV_t1a_vec)
                hidden_array_2a.append(sV_t2a_vec)
            
                sV_t1m_vec = sV_t1m.view(sV_t1m.size(0),1,sV_t1m.size(1))
                sV_t2m_vec = sV_t2m.view(sV_t2m.size(0),1,sV_t2m.size(1))
            
                hidden_array_1m.append(sV_t1m_vec)
                hidden_array_2m.append(sV_t2m_vec)
                

            sV_l1a = torch.cat(hidden_array_1a, dim=1)
            sV_l2a = torch.cat(hidden_array_2a, dim=1)
            sV_l1m = torch.cat(hidden_array_1m, dim=1)
            sV_l2m = torch.cat(hidden_array_2m, dim=1)
        
            sV_lla = torch.cat((sV_l1a,sV_l2a), dim=2)
            sV_llm = torch.cat((sV_l1m,sV_l2m), dim=2)

               
            #############################             
            # run text encoder second time
            #############################
            sT_t1,sT_t2,cT_t1,cT_t2 = self.init_hiddens()
            sT_t1,sT_t2 = sV_t1a+sV_t1m, sV_t2a+sV_t2m
            
            hidden_array_3 = []
            
                
            for i in xrange(nQuestionWords):
                input_question = features_questions[j,i:i+1]
                sT_t1, cT_t1 = self.lstm_text_1(input_question, (sT_t1, cT_t1))
                sT_t2, cT_t2 = self.lstm_text_2(sT_t1, (sT_t2, cT_t2))
                hidden_array_3.append(sT_t2)
            #print('Text encoding One size', sT_t1.size(), sT_t2.size())

            # here sT_t1, sT_t2 is the last hidden
            sT_t = torch.cat( (sT_t1,sT_t2), dim=1)  # should be of size (1,1024)
            
            
            #####################
            # temporal attention
            #####################
            vid_att_a = self.TpAtt_a(sV_lla, sT_t)
            vid_att_m = self.TpAtt_m(sV_llm, sT_t)


            ################
            # ram memory
            ################
            sT_rl = torch.cat(hidden_array_3, dim=0)

            memory_ram_vid = self.mrm_vid(sV_l2a[0,:,:], sV_l2m[0,:,:], nImg)
            memory_ram_txt = self.mrm_txt(sT_rl, nQuestionWords)
                
            if self.mm_version==1:
                svt_tmp = torch.cat((sV_t2a,sV_t2m),dim=1)
                smq = self.mm_module_v1(svt_tmp,memory_ram_vid,memory_ram_txt,self.iter_num)
            elif self.mm_version==2:
                smq = self.mm_module_v2(memory_ram_vid,memory_ram_txt)
                
            ######################### 
            # decode the final output
            ######################### 
            
            final_embed_a = torch.tanh( self.linear_decoder_att_a(vid_att_a) )
            final_embed_m = torch.tanh( self.linear_decoder_att_m(vid_att_m) )
            final_embed_2 = torch.tanh( self.linear_decoder_mem(smq) )
            final_embed = torch.cat([final_embed_a,final_embed_m,final_embed_2],dim=1)
        
                
            output = self.linear_decoder_count_2(final_embed)
            #print('Output size', output.size()) # (1,5)
            outputs.append(output)
            
            _,mx_idx = torch.max(output,1)
            predictions.append(mx_idx)
            #print(output,mx_idx)

    
        outputs = torch.cat(outputs, 0)
        #targets = torch.cat(targets, 0)
        predictions = torch.cat(predictions, 0)
        #print(predictions.size())
        return outputs, answers[:,0], predictions
    

    def accuracy(self, logits, targets):
        correct = torch.sum(logits.eq(targets)).float()
        return correct * 100.0 / targets.size(0)
