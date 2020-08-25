################################################################################
import os, sys
sys.path.insert(1, os.path.expanduser("~/kable_management/blp_paper/tvqa/mystuff"))
import tvqa_dataset
from torch.utils.data import DataLoader

import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from visdom_plotter import VisdomLinePlotter
sys.path.insert(1, os.path.expanduser("~/kable_management/blp_paper/models"))
################################################################################

#from attention_module_lite import *
from attention_module import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    """Main script."""
    torch.manual_seed(2667)

    ################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='train/test')
    parser.add_argument('--save_path', type=str, default=os.path.expanduser('~/kable_management/blp_paper/.results/hme_tvqa/saved_models'),
                        help='path for saving trained models')
    parser.add_argument('--test', type=int, default=0, help='0 | 1')
    parser.add_argument('--jobname', type=str , default='hme_tvqa-default',
                        help='jobname to direct to plotter') 
    parser.add_argument("--pool_type", type=str, default="default", choices=["default", "LinearSum", "ConcatMLP", "MCB", "MFH", "MFB", "MLB", "Block"], help="Which pooling technique to use")
    parser.add_argument("--pool_in_dims", type=int, nargs='+', default=[300,300], help="Input dimensions to pooling layers")
    parser.add_argument("--pool_out_dim", type=int, default=600, help="Output dimension to pooling layers")
    parser.add_argument("--pool_hidden_dim", type=int, default=1500, help="Some pooling types come with a hidden internal dimension")
    
    parser.add_argument("--results_dir_base", type=str, default=os.path.expanduser("~/kable_management/.results/test"))
    parser.add_argument("--log_freq", type=int, default=400, help="print, save training info")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--n_epoch", type=int, default=100, help="number of epochs to run")
    parser.add_argument("--max_es_cnt", type=int, default=3, help="number of epochs to early stop")
    parser.add_argument("--bsz", type=int, default=32, help="mini-batch size")
    parser.add_argument("--test_bsz", type=int, default=100, help="mini-batch size for testing")
    parser.add_argument("--device", type=int, default=0, help="gpu ordinal, -1 indicates cpu")
    parser.add_argument("--no_core_driver", action="store_true",
                                help="hdf5 driver, default use `core` (load into RAM), if specified, use `None`")
    parser.add_argument("--word_count_threshold", type=int, default=2, help="word vocabulary threshold")
    parser.add_argument("--no_glove", action="store_true", help="not use glove vectors")
    parser.add_argument("--no_ts", action="store_true", help="no timestep annotation, use full length feature")
    parser.add_argument("--input_streams", type=str, nargs="+", choices=["vcpt", "sub", "imagenet", "regional"], #added regional support here
                                help="input streams for the model")
    ################ My stuff
    parser.add_argument("--modelname", type=str, default="tvqa_abc", help="name of the model ot use")
    parser.add_argument("--lrtype", type=str, choices=["adam", "cyclic", "radam", "lrelu"], default="adam", help="Kind of learning rate")
    parser.add_argument("--poolnonlin", type=str, choices=["tanh", "relu", "sigmoid", "None", "lrelu"], default="None", help="add nonlinearities to pooling layer")
    parser.add_argument("--pool_dropout", type=float, default=0.0, help="Dropout value for the projections")
    parser.add_argument("--testrun", type=bool, default=False, help="set True to stop writing and visdom")
    parser.add_argument("--topk", type=int, default=1, help="To use instead of max pooling")
    parser.add_argument("--nosub", type=bool, default=False, help="Ignore the sub stream")
    parser.add_argument("--noimg", type=bool, default=False, help="Ignore the imgnet stream")
    parser.add_argument("--bert", type=str, choices=["default", "mine", "multi_choice", "qa"], default=None, help="What kind of BERT model to use")
    parser.add_argument("--reg_feat_path", type=str, default=os.path.expanduser("~/kable_management/data/tvqa/regional_features/100p.h5"),
                                help="regional features")
    parser.add_argument("--my_vcpt", type=bool, default=False, help="Use my extracted visual concepts")
    parser.add_argument("--lanecheck", type=bool, default=False, help="Validation lane checks")
    parser.add_argument("--lanecheck_path", type=str, help="Validation lane check path")
    parser.add_argument("--best_path", type=str, help="Path to best model")
    parser.add_argument("--disable_streams", type=str, default=None, nargs="+", choices=["vcpt", "sub", "imagenet", "regional"], #added regional support here
                                help="disable the input stream from voting in the softmax of model outputs")
    parser.add_argument("--dset", choices=["valid", "test", "train"], default="valid", type=str, help="The dataset to use")
    ########################

    parser.add_argument("--n_layers_cls", type=int, default=1, help="number of layers in classifier")
    parser.add_argument("--hsz1", type=int, default=150, help="hidden size for the first lstm")
    parser.add_argument("--hsz2", type=int, default=300, help="hidden size for the second lstm")
    parser.add_argument("--embedding_size", type=int, default=300, help="word embedding dim")
    parser.add_argument("--max_sub_l", type=int, default=300, help="max length for subtitle")
    parser.add_argument("--max_vcpt_l", type=int, default=300, help="max length for visual concepts")
    parser.add_argument("--max_vid_l", type=int, default=480, help="max length for video feature")
    parser.add_argument("--vocab_size", type=int, default=0, help="vocabulary size")
    parser.add_argument("--no_normalize_v", action="store_true", help="do not normalize video featrue")
    # Data paths
    parser.add_argument("--train_path", type=str, default=os.path.expanduser("~/kable_management/data/tvqa/tvqa_train_processed.json"),
                                help="train set path")
    parser.add_argument("--valid_path", type=str, default=os.path.expanduser("~/kable_management/data/tvqa/tvqa_val_processed.json"),
                                help="valid set path")
    parser.add_argument("--test_path", type=str, default=os.path.expanduser("~/kable_management/data/tvqa/tvqa_test_public_processed.json"),
                                help="test set path")
    parser.add_argument("--glove_path", type=str, default=os.path.expanduser("~/kable_management/data/word_embeddings/glove.6B.300d.txt"),
                                help="GloVe pretrained vector path")
    parser.add_argument("--vcpt_path", type=str, default=os.path.expanduser("~/kable_management/data/tvqa/vcpt_features/py2_det_visual_concepts_hq.pickle"),
                                help="visual concepts feature path")
    parser.add_argument("--vid_feat_path", type=str, default=os.path.expanduser("~/kable_management/data/tvqa/imagenet_features/tvqa_imagenet_pool5_hq.h5"),
                                help="imagenet feature path")
    parser.add_argument("--vid_feat_size", type=int, default=2048,
                                help="visual feature dimension")
    parser.add_argument("--word2idx_path", type=str, default=os.path.expanduser("~/kable_management/data/tvqa/cache/py2_word2idx.pickle"),
                                help="word2idx cache path")
    parser.add_argument("--idx2word_path", type=str, default=os.path.expanduser("~/kable_management/data/tvqa/cache/py2_idx2word.pickle"),
                                help="idx2word cache path")
    parser.add_argument("--vocab_embedding_path", type=str, default=os.path.expanduser("~/kable_management/data/tvqa/cache/py2_vocab_embedding.pickle"),
                                help="vocab_embedding cache path")
    parser.add_argument("--regional_topk", type=int, default=-1, help="Pick top-k scoring regional features across all frames")
    
    args = parser.parse_args()

    plotter = VisdomLinePlotter(env_name="TEST")#args.jobname)
    args.plotter = plotter
    args.normalize_v = not args.no_normalize_v
    args.device = torch.device("cuda:%d" % args.device if args.device >= 0 else "cpu")
    args.with_ts = not args.no_ts
    args.input_streams = [] if args.input_streams is None else args.input_streams
    args.vid_feat_flag = True if "imagenet" in args.input_streams else False
    args.h5driver = None if args.no_core_driver else "core"

    
    #args.dataset = 'msvd_qa'
    args.dataset = 'tvqa'

    args.word_dim = 300
    args.vocab_num = 4000
    args.pretrained_embedding = os.path.expanduser("~/kable_management/data/msvd_qa/word_embedding.npy")
    args.video_feature_dim = 4096
    args.video_feature_num = 480#20
    args.answer_num = 1000
    args.memory_dim = 256
    args.batch_size = 32
    args.reg_coeff = 1e-5
    args.learning_rate = 0.001
    args.preprocess_dir = os.path.expanduser("~/kable_management/data/hme_tvqa/")
    args.log = os.path.expanduser('~/kable_management/blp_paper/.results/msvd/logs')
    args.hidden_size = 512
    args.image_feature_net = 'concat'
    args.layer = 'fc'
    ################################################################################







    #model.load_embedding(dset.vocab_embedding)
    #opt.vocab_size = len(dset.word2idx)


    ##############################################################################
    ##############################################################################
    ##############################################################################
    ##############################################################################
    ##############################################################################
    ##############################################################################
    ##############################################################################
    ##############################################################################
    ### add arguments ### 
    args.max_sequence_length = 480

    args.memory_type = '_mrm2s'
    args.image_feature_net = 'concat'
    args.layer = 'fc'


    
    #args.save_model_path = args.save_path + '%s_%s_%s%s' % (args.task,args.image_feature_net,args.layer,args.memory_type)
    ## Create model directory
    # if not os.path.exists(args.save_model_path):
    #     os.makedirs(args.save_model_path)
    


    
    #############################
    # Parameters
    #############################
    SEQUENCE_LENGTH = args.max_sequence_length
    VOCABULARY_SIZE = train_dataset.n_words
    assert VOCABULARY_SIZE == voc_len
    FEAT_DIM = train_dataset.get_video_feature_dimension()[1:]
    train_iter = train_dataset.batch_iter(args.num_epochs, args.batch_size)


    # Create model directory
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    from embed_loss import MultipleChoiceLoss
    criterion = MultipleChoiceLoss(num_option=5, margin=1, size_average=True).cuda()

    
    
    if args.memory_type=='_mrm2s':
        rnn = TGIFAttentionTwoStream(args, 'Trans', feat_channel, feat_dim, text_embed_size, args.hidden_size,
                             voc_len, args.num_layers, word_matrix, answer_vocab_size = answer_vocab_size, 
                             max_len=args.max_sequence_length)
    else:
        assert 1==2
        
    rnn = rnn.cuda()
    ##############################################################################
    ############################################################################################################################################################
    ############################################################################################################################################################
    ############################################################################################################################################################
    ##############################################################################





# Heres my code
# rnn = TGIFAttentionTwoStream(args, feat_channel, feat_dim, text_embed_size, args.hidden_size,
#                      voc_len, num_layers, word_matrix, answer_vocab_size = answer_vocab_size,
#                      max_len=max_sequence_length)
# #rnn = rnn.cuda()
# rnn = rnn.to(args.device)

# if args.test == 1:
#     rnn.load_state_dict(torch.load(os.path.join(args.save_model_path, 'rnn-3800-vl_0.316.pkl')))



# # loss function
# criterion = nn.CrossEntropyLoss(size_average=True).cuda()
# optimizer = torch.optim.Adam(rnn.parameters(), lr=args.learning_rate)


# iter = 0
# ######################################################################
# best_val_acc = 0.0
# best_test_acc = 0.0
# train_loss = []
# train_acc = []
# ######################################################################
# for epoch in range(0, 10000):
#     #dataset.reset_train()
#     for batch_idx, batch in enumerate(train_loader):
#         tvqa_data, targets, _ = tvqa_dataset.preprocess_inputs(batch, args.max_sub_l, args.max_vcpt_l, args.max_vid_l,
#                                                  device='cpu')#args.device
#         assert( tvqa_data[18].shape[2] == 4096 )
#         data_dict = {}                                         
#         data_dict['num_mult_choices'] = 5
#         data_dict['answers'] = targets #'int32 torch tensor correct answer ids'
#         data_dict['video_features'] = torch.cat( [tvqa_data[18], tvqa_data[16]], dim=2 ).unsqueeze(2).unsqueeze(2)#'(bsz, 35(max), 1, 1, 6144)'
#         data_dict['video_lengths'] = tvqa_data[17].tolist() #'list, (bsz), number of frames, #max is 35'
#         data_dict['candidates'] = tvqa_data[0]#'torch tensor (bsz, 5, 35) (max q length)'
#         data_dict['candidate_lengths'] = tvqa_data[1]#'list of list, lengths of each question+ans'
#         import ipdb; ipdb.set_trace()


################################################################################



















    #dataset = dt.MSVDQA(args.batch_size, args.preprocess_dir)
    dataset = tvqa_dataset.TVQADataset(args, mode="train")
    train_loader = DataLoader(dataset, batch_size=args.bsz, shuffle=True, collate_fn=tvqa_dataset.pad_collate)

    args.memory_type='_mrm2s'
    args.save_model_path = args.save_path + 'model_%s_%s%s' % (args.image_feature_net,args.layer,args.memory_type)
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)



    #############################
    # get video feature dimension
    #############################
    feat_channel = args.video_feature_dim
    feat_dim = 1
    text_embed_size = args.word_dim
    answer_vocab_size = args.answer_num
    voc_len = args.vocab_num
    num_layers = 2
    max_sequence_length = args.video_feature_num
    word_matrix = np.load(args.pretrained_embedding)
    answerset = pd.read_csv(os.path.expanduser('~/kable_management/data/msvd_qa/answer_set.txt'), header=None)[0]


    rnn = TGIFAttentionTwoStream(args, feat_channel, feat_dim, text_embed_size, args.hidden_size,
                         voc_len, num_layers, word_matrix, answer_vocab_size = answer_vocab_size,
                         max_len=max_sequence_length)
    #rnn = rnn.cuda()
    rnn = rnn.to(args.device)

    if args.test == 1:
        rnn.load_state_dict(torch.load(os.path.join(args.save_model_path, 'rnn-3800-vl_0.316.pkl')))



    # loss function
    criterion = nn.CrossEntropyLoss(size_average=True).cuda()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=args.learning_rate)

    
    iter = 0
    ######################################################################
    best_val_acc = 0.0
    best_test_acc = 0.0
    train_loss = []
    train_acc = []
    ######################################################################
    for epoch in range(0, 10000):
        #dataset.reset_train()
        for batch_idx, batch in enumerate(train_loader):
            tvqa_data, targets, _ = tvqa_dataset.preprocess_inputs(batch, args.max_sub_l, args.max_vcpt_l, args.max_vid_l,
                                                     device='cpu')#args.device
            assert( tvqa_data[18].shape[2] == 4096 )
            data_dict = {}                                         
            data_dict['num_mult_choices'] = 5
            data_dict['answers'] = targets #'int32 torch tensor correct answer ids'
            data_dict['video_features'] = torch.cat( [tvqa_data[18], tvqa_data[16]], dim=2 ).unsqueeze(2).unsqueeze(2)#'(bsz, 35(max), 1, 1, 6144)'
            data_dict['video_lengths'] = tvqa_data[17].tolist() #'list, (bsz), number of frames, #max is 35'
            data_dict['candidates'] = tvqa_data[0]#'torch tensor (bsz, 5, 35) (max q length)'
            data_dict['candidate_lengths'] = tvqa_data[1]#'list of list, lengths of each question+ans'
            import ipdb; ipdb.set_trace()

            outputs, predictions = rnn(data_dict)
            targets = data_dict['answers']

            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            import ipdb; ipdb.set_trace()

            acc = rnn.accuracy(predictions, targets)
            print('Train iter %d, loss %.3f, acc %.2f' % (iter,loss.data,acc.item()))
            train_loss.append(loss.item())
            train_acc.append(acc.item())
            ######################################################################
            #args.plotter.plot("accuracy", "train", args.jobname, iter, acc.item())
            #import ipdb; ipdb.set_trace()
            #args.plotter.plot("loss", "train", args.jobname, iter, loss.item())
            ######################################################################


        # while dataset.has_train_batch:

        #     if args.test==0:
        #         vgg, c3d, questions, answers, question_lengths = dataset.get_train_batch()
                
        #         data_dict = getInput(vgg, c3d, questions, answers, question_lengths)
        #         outputs, predictions = rnn(data_dict)
        #         targets = data_dict['answers']

        #         loss = criterion(outputs, targets)

        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #         import ipdb; ipdb.set_trace()

        #         acc = rnn.accuracy(predictions, targets)
        #         print('Train iter %d, loss %.3f, acc %.2f' % (iter,loss.data,acc.item()))
        #         train_loss.append(loss.item())
        #         train_acc.append(acc.item())
        #         ######################################################################
        #         #args.plotter.plot("accuracy", "train", args.jobname, iter, acc.item())
        #         #import ipdb; ipdb.set_trace()
        #         #args.plotter.plot("loss", "train", args.jobname, iter, loss.item())
        #         ######################################################################

        #     if iter % 10==0:# % 1000
        #         rnn.eval()
        #         ######################################################################
        #         train_loss = sum(train_loss) / float(len(train_loss))
        #         train_acc = sum(train_acc) / float(len(train_acc))
        #         args.plotter.plot("accuracy", "train", args.jobname, iter, train_acc)
        #         #import ipdb; ipdb.set_trace()
        #         args.plotter.plot("loss", "train", args.jobname, iter, train_loss)
        #         train_loss = []
        #         train_acc = []
        #         ######################################################################
                

        #         with torch.no_grad():

        #             # val iterate over examples
        #             if args.test == 0:
        #                 correct = 0
        #                 idx = 0
        #                 while dataset.has_val_example:
        #                     if idx % 100 == 0:
        #                         print 'Val iter %d/%d' % (idx,dataset.val_example_total)
        #                     vgg, c3d, questions, answer, question_lengths = dataset.get_val_example()
        #                     data_dict = getInput(vgg, c3d, questions, None, question_lengths, False)
        #                     outputs, predictions = rnn(data_dict)
        #                     prediction = predictions.item()
        #                     idx += 1
        #                     if answerset[prediction] == answer:
        #                         correct += 1

        #                 val_acc = 100.0*correct / dataset.val_example_total
        #                 print correct, dataset.val_example_total
        #                 print('Val iter %d, acc %.3f' % (iter, val_acc))
                        
        #                 ######################################################################
        #                 if best_val_acc < val_acc:
        #                     best_val_acc = val_acc
        #                     args.plotter.text_plot(args.jobname+" val", args.jobname+" val "+str(round(best_val_acc, 4))+" "+str(iter))
        #                 args.plotter.plot("accuracy", "val", args.jobname, iter, val_acc)
        #                 ######################################################################

        #                 dataset.reset_val()
        #                 torch.save(rnn.state_dict(), os.path.join(args.save_model_path,
        #                                                           'rnn-%04d-vl_%.3f.pkl' %(iter,val_acc)))

        #             correct = 0
        #             idx = 0
        #             while dataset.has_test_example:
        #                 if idx % 100 == 0:
        #                     print 'Test iter %d/%d' % (idx,dataset.test_example_total)
        #                 vgg, c3d, questions, answer, question_lengths, _ = dataset.get_test_example()
        #                 data_dict = getInput(vgg, c3d, questions, None, question_lengths, False)
        #                 outputs, predictions = rnn(data_dict)
        #                 prediction = predictions.item()
        #                 idx += 1
        #                 if answerset[prediction] == answer:
        #                     correct += 1

        #             test_acc = 100.0*correct / dataset.test_example_total
        #             print correct, dataset.test_example_total
        #             print('Test iter %d, acc %.3f' % (iter, test_acc))

        #             ######################################################################
        #             if best_test_acc < test_acc:
        #                 best_test_acc = test_acc
        #                 args.plotter.text_plot(args.jobname+" test", args.jobname+" test "+str(round(best_test_acc, 4))+" "+str(iter))
        #             args.plotter.plot("accuracy", "test", args.jobname, iter, test_acc)
        #             ######################################################################

        #             dataset.reset_test()

        #             if args.test == 1:
        #                 exit()


        #         rnn.train()
        #     iter += 1
            


if __name__ == '__main__':
    main()
