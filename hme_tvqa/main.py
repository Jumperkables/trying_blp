import os, sys
#sys.path.insert(1, os.path.expanduser("~/kable_management/projects/trying_blp/tvqa"))
import tvqa_dataset
from torch.utils.data import DataLoader
import argparse
import torch
import torch.nn as nn
import numpy as np
################################################################################
#sys.path.insert(1, os.path.expanduser("~/kable_management/blp_paper/models"))i
sys.path.insert(1, "../models")
sys.path.insert(1, "../tvqa/tools")
################################################################################
from attention_module import *
from util import AverageMeter
from visdom_plotter import VisdomLinePlotter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main(args):
    torch.manual_seed(2667)#1
    
    ### add arguments ### 
    args.vc_dir = os.path.expanduser("~/kable_management/data/tgif_qa/data/Vocabulary")
    args.df_dir = os.path.expanduser("~/kable_management/data/tgif_qa/data/dataset")

    # Dataset
    dataset = tvqa_dataset.TVQADataset(args, mode="train")

    args.max_sequence_length = 50#35
    args.model_name = args.task + args.feat_type

    args.memory_type = '_mrm2s'
    args.image_feature_net = 'concat'
    args.layer = 'fc'


    
    args.save_model_path = args.save_path + '%s_%s_%s%s' % (args.task,args.image_feature_net,args.layer,args.memory_type)
    # Create model directory
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    

    
    
    #############################
    # get video feature dimension
    #############################
    video_feature_dimension = (480,1,1,6144)
    feat_channel = video_feature_dimension[3]
    feat_dim = video_feature_dimension[2]
    text_embed_size = 300
    answer_vocab_size = None
    
    #############################
    # get word vector dimension
    #############################
    word_matrix = dataset.vocab_embedding
    voc_len = len(dataset.word2idx)
    assert text_embed_size==word_matrix.shape[1]
    
    #############################
    # Parameters
    #############################
    SEQUENCE_LENGTH = args.max_sequence_length
    VOCABULARY_SIZE = voc_len
    assert VOCABULARY_SIZE == voc_len
    FEAT_DIM = (1,1,6144)

    # Create model directory
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)


    if args.task=='Count':
        # add L2 loss
        criterion = nn.MSELoss(size_average=True).to(args.device)
    elif args.task in ['Action','Trans']:
        from embed_loss import MultipleChoiceLoss
        criterion = MultipleChoiceLoss(num_option=5, margin=1, size_average=True).to(args.device)
    elif args.task=='FrameQA':
        # add classification loss
        answer_vocab_size = len(train_dataset.ans2idx)
        print('Vocabulary size', answer_vocab_size, VOCABULARY_SIZE)
        criterion = nn.CrossEntropyLoss(size_average=True).to(args.device)
    
    
    if args.memory_type=='_mrm2s':
        rnn = TGIFAttentionTwoStream(args, args.task, feat_channel, feat_dim, text_embed_size, args.hidden_size,
                             voc_len, args.num_layers, word_matrix, answer_vocab_size = answer_vocab_size, 
                             max_len=args.max_sequence_length)
    else:
        assert 1==2
        
    rnn = rnn.to(args.device)
    
    #  to directly test, load pre-trained model, replace with your model to test your model
    if args.test==1:
        if args.task == 'Count':
            rnn.load_state_dict(torch.load('./saved_models/Count_concat_fc_mrm2s/rnn-1300-l3.257-a27.942.pkl'))
        elif args.task == 'Action':
            rnn.load_state_dict(torch.load('./saved_models/Action_concat_fc_mrm2s/rnn-0800-l0.137-a84.663.pkl'))
        elif args.task == 'Trans':
            rnn.load_state_dict(torch.load('./saved_models/Trans_concat_fc_mrm2s/rnn-1500-l0.246-a78.068.pkl'))
        elif args.task == 'FrameQA':
            rnn.load_state_dict(torch.load('./saved_models/FrameQA_concat_fc_mrm2s/rnn-4200-l1.233-a69.361.pkl'))
        else:
            assert 1==2, 'Invalid task'
    
    optimizer = torch.optim.Adam(rnn.parameters(), lr=args.learning_rate)
    
        
    iter = 0

    if args.task in ['Action','Trans']:
        best_val_acc = 0.0
        best_val_iter = 0.0
        best_test_acc = 0.0
        train_acc = []
        train_loss = []

        for epoch in range(0, args.num_epochs):
            dataset.set_mode("train")
            train_loader = DataLoader(dataset, batch_size=args.bsz, shuffle=True, collate_fn=tvqa_dataset.pad_collate, num_workers=args.num_workers)
            for batch_idx, batch in enumerate(train_loader):
                # if(batch_idx<51):
                #     iter+=1
                #     continue
                # else:
                #     pass
                tvqa_data, targets, _ = tvqa_dataset.preprocess_inputs(batch, args.max_sub_l, args.max_vcpt_l, args.max_vid_l,
                                                         device=args.device)
                assert( tvqa_data[18].shape[2] == 4096 )
                data_dict = {}                                         
                data_dict['num_mult_choices'] = 5
                data_dict['answers'] = targets #'int32 torch tensor correct answer ids'
                data_dict['video_features'] = torch.cat( [tvqa_data[18], tvqa_data[16]], dim=2 ).unsqueeze(2).unsqueeze(2)#'(bsz, 35(max), 1, 1, 6144)'
                data_dict['video_lengths'] = tvqa_data[17].tolist() #'list, (bsz), number of frames, #max is 35'
                data_dict['candidates'] = tvqa_data[0]#'torch tensor (bsz, 5, 35) (max q length)'
                data_dict['candidate_lengths'] = tvqa_data[1]#'list of list, lengths of each question+ans'
                # if hard restrict
                if args.hard_restrict:
                    data_dict['video_features'] = data_dict['video_features'][:,:args.max_sequence_length]#'(bsz, 35(max), 1, 1, 6144)'
                    data_dict['video_lengths'] = [i if i<=args.max_sequence_length else args.max_sequence_length for i in data_dict['video_lengths']]
                    data_dict['candidates'] = data_dict['candidates'][:,:,:args.max_sequence_length]
                    for xx in range(data_dict['candidate_lengths'].shape[0]):
                        if(data_dict['video_lengths'][xx]==0):
                            data_dict['video_lengths'][xx] = 1
                        for yy in range(data_dict['candidate_lengths'].shape[1]):
                            if(data_dict['candidate_lengths'][xx][yy]>args.max_sequence_length):
                                data_dict['candidate_lengths'][xx][yy] = args.max_sequence_length
                #import ipdb; ipdb.set_trace()
                #print("boom")
                outputs, targets, predictions = rnn(data_dict, args.task)

                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acc = rnn.accuracy(predictions, targets.long())
                #print("boom")
                print('Train %s iter %d, loss %.3f, acc %.2f' % (args.task,iter,loss.data,acc.item()))
                ######################################################################
                #args.plotter.plot("accuracy", "train", args.jobname, iter, acc.item())
                #args.plotter.plot("loss", "train", args.jobname, iter, int(loss.data))
                train_acc.append(acc.item())
                train_loss.append(loss.item())
                ######################################################################

                if (batch_idx % args.log_freq) == 0:#(args.log_freq-1):
                    dataset.set_mode("valid")
                    valid_loader = DataLoader(dataset, batch_size=args.test_bsz, shuffle=True, collate_fn=tvqa_dataset.pad_collate, num_workers=args.num_workers)
                    rnn.eval()
                    ######################################################################
                    train_loss = sum(train_loss) / float(len(train_loss))
                    train_acc = sum(train_acc) / float(len(train_acc))
                    args.plotter.plot("accuracy", "train", args.jobname+" "+args.task, iter, train_acc)
                    #import ipdb; ipdb.set_trace()
                    args.plotter.plot("loss", "train", args.jobname+" "+args.task, iter, train_loss)
                    train_loss = []
                    train_acc = []
                    ######################################################################
                    with torch.no_grad():
                        if args.test == 0:
                            n_iter = len(valid_loader)
                            losses = []
                            accuracy = []
                            iter_val = 0
                            for batch_idx, batch in enumerate(valid_loader):
                                tvqa_data, targets, _ = tvqa_dataset.preprocess_inputs(batch, args.max_sub_l, args.max_vcpt_l, args.max_vid_l,
                                                         device=args.device)
                                assert( tvqa_data[18].shape[2] == 4096 )
                                data_dict = {}                                         
                                data_dict['num_mult_choices'] = 5
                                data_dict['answers'] = targets #'int32 torch tensor correct answer ids'
                                data_dict['video_features'] = torch.cat( [tvqa_data[18], tvqa_data[16]], dim=2 ).unsqueeze(2).unsqueeze(2)#'(bsz, 35(max), 1, 1, 6144)'
                                data_dict['video_lengths'] = tvqa_data[17].tolist() #'list, (bsz), number of frames, #max is 35'
                                data_dict['candidates'] = tvqa_data[0]#'torch tensor (bsz, 5, 35) (max q length)'
                                data_dict['candidate_lengths'] = tvqa_data[1]#'list of list, lengths of each question+ans'
                                # if hard restrict
                                if args.hard_restrict:
                                    data_dict['video_features'] = data_dict['video_features'][:,:args.max_sequence_length]#'(bsz, 35(max), 1, 1, 6144)'
                                    data_dict['video_lengths'] = [i if i<=args.max_sequence_length else args.max_sequence_length for i in data_dict['video_lengths']]
                                    data_dict['candidates'] = data_dict['candidates'][:,:,:args.max_sequence_length]
                                    for xx in range(data_dict['candidate_lengths'].shape[0]):
                                        if(data_dict['video_lengths'][xx]==0):
                                            data_dict['video_lengths'][xx] = 1
                                        for yy in range(data_dict['candidate_lengths'].shape[1]):
                                            if(data_dict['candidate_lengths'][xx][yy]>args.max_sequence_length):
                                                data_dict['candidate_lengths'][xx][yy] = args.max_sequence_length





                                outputs, targets, predictions = rnn(data_dict, args.task)

                                loss = criterion(outputs, targets)
                                acc = rnn.accuracy(predictions, targets.long())

                                losses.append(loss.item())
                                accuracy.append(acc.item())
                            losses = sum(losses) / n_iter
                            accuracy = sum(accuracy) / n_iter
                            
                            ######################################################################
                            if best_val_acc < accuracy:
                                best_val_iter = iter
                                best_val_acc = accuracy
                                args.plotter.text_plot(args.task+args.jobname+" val", args.task+args.jobname+" val "+str(round(best_val_acc, 4))+" "+str(iter))
                            #args.plotter.plot("loss", "val", args.jobname, iter, losses.avg)
                            args.plotter.plot("accuracy", "val", args.jobname+" "+args.task, iter, accuracy)
                            ######################################################################

                            #print('[Val] iter %d, loss %.3f, acc %.2f, best acc %.3f at iter %d' % (
                            #                iter, losses, accuracy, best_val_acc, best_val_iter))
                            # torch.save(rnn.state_dict(), os.path.join(args.save_model_path, 'rnn-%04d-l%.3f-a%.3f.pkl' % (
                            #                 iter, losses.avg, accuracy.avg)))
                    rnn.train()
                    dataset.set_mode("train")

                iter += 1
    elif args.task=='Count':
        pass
        # best_val_loss = 100.0
        # best_val_iter = 0.0
        # best_test_loss = 100.0
        # ######################################################################
        # best_val_acc = 0.0
        # best_test_acc = 0.0
        # train_acc = []
        # train_loss = []
        # ######################################################################

        # # this is a regression problem, predict a value from 1-10
        # for batch_chunk in train_iter:

        #     if args.test==0:
        #         video_features = torch.from_numpy(batch_chunk['video_features'].astype(np.float32)).cuda()
        #         video_lengths = batch_chunk['video_lengths']
        #         question_words = torch.from_numpy(batch_chunk['question_words'].astype(np.int64)).cuda()
        #         question_lengths = batch_chunk['question_lengths']
        #         answers = torch.from_numpy(batch_chunk['answer'].astype(np.float32)).cuda()

        #         data_dict = {}
        #         data_dict['video_features'] = video_features
        #         data_dict['video_lengths'] = video_lengths
        #         data_dict['question_words'] = question_words
        #         data_dict['question_lengths'] = question_lengths
        #         data_dict['answers'] = answers

        #         outputs, targets, predictions = rnn(data_dict, 'Count')
        #         loss = criterion(outputs, targets)

        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()

        #         acc = rnn.accuracy(predictions, targets.int())
        #         print('Train %s iter %d, loss %.3f, acc %.2f' % (args.task,iter,loss.data,acc.item()))
        #         ######################################################################
        #         #args.plotter.plot("accuracy", "train", args.jobname, iter, acc.item())
        #         #args.plotter.plot("loss", "train", args.jobname, iter, int(loss.data))
        #         train_acc.append(acc.item())
        #         train_loss.append(loss.item())
        #         ######################################################################

        #     if iter % 100==0:
        #         rnn.eval()
        #         ######################################################################
        #         train_loss = sum(train_loss) / float(len(train_loss))
        #         train_acc = sum(train_acc) / float(len(train_acc))
        #         args.plotter.plot("accuracy", "train", args.jobname+" "+args.task, iter, train_acc)
        #         #import ipdb; ipdb.set_trace()
        #         args.plotter.plot("loss", "train", args.jobname+" "+args.task, iter, train_loss)
        #         train_loss = []
        #         train_acc = []
        #         ######################################################################
        #         with torch.no_grad():

        #             if args.test == 0:
        #                 ##### Validation ######
        #                 n_iter = len(val_dataset) / args.batch_size
        #                 losses = AverageMeter()
        #                 accuracy = AverageMeter()

        #                 iter_val = 0
        #                 for batch_chunk in val_dataset.batch_iter(1, args.batch_size, shuffle=False):
        #                     if iter_val % 10 == 0:
        #                         print('%d/%d' % (iter_val, n_iter))

        #                     iter_val += 1

        #                     video_features = torch.from_numpy(batch_chunk['video_features'].astype(np.float32)).cuda()
        #                     video_lengths = batch_chunk['video_lengths']
        #                     question_words = torch.from_numpy(batch_chunk['question_words'].astype(np.int64)).cuda()
        #                     question_lengths = batch_chunk['question_lengths']
        #                     answers = torch.from_numpy(batch_chunk['answer'].astype(np.float32)).cuda()

        #                     # print(question_words)
        #                     data_dict = {}
        #                     data_dict['video_features'] = video_features
        #                     data_dict['video_lengths'] = video_lengths
        #                     data_dict['question_words'] = question_words
        #                     data_dict['question_lengths'] = question_lengths
        #                     data_dict['answers'] = answers

        #                     outputs, targets, predictions = rnn(data_dict, 'Count')
        #                     loss = criterion(outputs, targets)
        #                     acc = rnn.accuracy(predictions, targets.int())

        #                     losses.update(loss.item(), video_features.size(0))
        #                     accuracy.update(acc.item(), video_features.size(0))

        #                 if best_val_loss > losses.avg:
        #                     best_val_loss = losses.avg
        #                     best_val_iter = iter
        #                     args.plotter.text_plot("loss "+args.task+args.jobname+" val", args.task+args.jobname+" val loss "+str(round(best_val_loss, 5))+" "+str(iter))

        #                 print('[Val] iter %d, loss %.3f, acc %.2f, best loss %.3f at iter %d' % (
        #                     iter, losses.avg, accuracy.avg, best_val_loss, best_val_iter))
        #                 ######################################################################
        #                 if best_val_acc < accuracy.avg:
        #                     best_val_acc = accuracy.avg
        #                     args.plotter.text_plot(args.task+args.jobname+" val", args.task+args.jobname+" val "+str(round(best_val_acc, 4))+" "+str(iter))
        #                 args.plotter.plot("loss", "val", args.jobname, iter, losses.avg)
        #                 args.plotter.plot("accuracy", "val", args.jobname+" "+args.task, iter, accuracy.avg)
        #                 ######################################################################

        #                 torch.save(rnn.state_dict(), os.path.join(args.save_model_path, 'rnn-%04d-l%.3f-a%.3f.pkl' % (
        #                             iter, losses.avg, accuracy.avg)))

        #             if 1 == 1:
        #                 ###### Test ######
        #                 n_iter = len(test_dataset) / args.batch_size
        #                 losses = AverageMeter()
        #                 accuracy = AverageMeter()

        #                 iter_test = 0
        #                 for batch_chunk in test_dataset.batch_iter(1, args.batch_size, shuffle=False):
        #                     if iter_test % 10 == 0:
        #                         print('%d/%d' % (iter_test, n_iter))

        #                     iter_test+=1

        #                     video_features = torch.from_numpy(batch_chunk['video_features'].astype(np.float32)).cuda()
        #                     video_lengths = batch_chunk['video_lengths']
        #                     question_words = torch.from_numpy(batch_chunk['question_words'].astype(np.int64)).cuda()
        #                     question_lengths = batch_chunk['question_lengths']
        #                     answers = torch.from_numpy(batch_chunk['answer'].astype(np.float32)).cuda()

        #                     data_dict = {}
        #                     data_dict['video_features'] = video_features
        #                     data_dict['video_lengths'] = video_lengths
        #                     data_dict['question_words'] = question_words
        #                     data_dict['question_lengths'] = question_lengths
        #                     data_dict['answers'] = answers


        #                     outputs, targets, predictions = rnn(data_dict, 'Count')
        #                     loss = criterion(outputs, targets)
        #                     acc = rnn.accuracy(predictions, targets.int())

        #                     losses.update(loss.item(), video_features.size(0))
        #                     accuracy.update(acc.item(), video_features.size(0))

                        
        #                 print('[Test] iter %d, loss %.3f, acc %.2f' % (iter,losses.avg,accuracy.avg))
        #                 ######################################################################
        #                 if best_test_loss > losses.avg:
        #                     best_test_loss = losses.avg
        #                     args.plotter.text_plot("loss "+args.task+args.jobname+" test", args.task+args.jobname+" test loss "+str(round(best_test_loss, 5))+" "+str(iter))

        #                 if best_test_acc < accuracy.avg:
        #                     best_test_acc = accuracy.avg
        #                     args.plotter.text_plot(args.task+args.jobname+" test", args.task+args.jobname+" test "+str(round(best_test_acc, 4))+" "+str(iter))
        #                 args.plotter.plot("loss", "test", args.jobname, iter, losses.avg)
        #                 args.plotter.plot("accuracy", "test", args.jobname+" "+args.task, iter, accuracy.avg)
        #                 ######################################################################
                        
        #                 if args.test==1:
        #                     exit()

        #         rnn.train()

        #     iter += 1


    # elif args.task=='FrameQA':
    #     pass
    #     best_val_acc = 0.0
    #     best_val_iter = 0.0
    #     ####################################################################################
    #     best_test_acc = 0.0
    #     train_loss = []
    #     train_acc = []
    #     ####################################################################################
    
    #     # this is a multiple-choice problem, predict probability of each class
    #     for batch_chunk in train_iter:

    #         if args.test==0:
    #             video_features = torch.from_numpy(batch_chunk['video_features'].astype(np.float32)).cuda()
    #             video_lengths = batch_chunk['video_lengths']
    #             question_words = torch.from_numpy(batch_chunk['question_words'].astype(np.int64)).cuda()
    #             question_lengths = batch_chunk['question_lengths']
    #             answers = torch.from_numpy(batch_chunk['answer'].astype(np.int64)).cuda()

    #             data_dict = {}
    #             data_dict['video_features'] = video_features
    #             data_dict['video_lengths'] = video_lengths
    #             data_dict['question_words'] = question_words
    #             data_dict['question_lengths'] = question_lengths
    #             data_dict['answers'] = answers


    #             outputs, targets, predictions = rnn(data_dict, args.task)

    #             loss = criterion(outputs, targets)

    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()

    #             acc = rnn.accuracy(predictions, targets)
    #             print('Train %s iter %d, loss %.3f, acc %.2f' % (args.task,iter,loss.data,acc.item()))
    #             ######################################################################
    #             #args.plotter.plot("accuracy", "train", args.jobname, iter, acc.item())
    #             #args.plotter.plot("loss", "train", args.jobname, iter, int(loss.data))
    #             train_loss.append(loss.item())
    #             train_acc.append(acc.item())
    #             ######################################################################


    #         if iter % 100==0:
    #             rnn.eval()
    #             ######################################################################
    #             train_loss = sum(train_loss) / float(len(train_loss))
    #             train_acc = sum(train_acc) / float(len(train_acc))
    #             args.plotter.plot("accuracy", "train", args.jobname+" "+args.task, iter, train_acc)
    #             #import ipdb; ipdb.set_trace()
    #             args.plotter.plot("loss", "train", args.jobname+" "+args.task, iter, train_loss)
    #             train_loss = []
    #             train_acc = []
    #             ######################################################################
    #             with torch.no_grad():

    #                 if args.test == 0:
    #                     losses = AverageMeter()
    #                     accuracy = AverageMeter()
    #                     n_iter = len(val_dataset) / args.batch_size

    #                     iter_val = 0
    #                     for batch_chunk in val_dataset.batch_iter(1, args.batch_size, shuffle=False):
    #                         if iter_val % 10 == 0:
    #                             print('%d/%d' % (iter_val, n_iter))

    #                         iter_val += 1

    #                         video_features = torch.from_numpy(batch_chunk['video_features'].astype(np.float32)).cuda()
    #                         video_lengths = batch_chunk['video_lengths']
    #                         question_words = torch.from_numpy(batch_chunk['question_words'].astype(np.int64)).cuda()
    #                         question_lengths = batch_chunk['question_lengths']
    #                         answers = torch.from_numpy(batch_chunk['answer'].astype(np.int64)).cuda()

    #                         data_dict = {}
    #                         data_dict['video_features'] = video_features
    #                         data_dict['video_lengths'] = video_lengths
    #                         data_dict['question_words'] = question_words
    #                         data_dict['question_lengths'] = question_lengths
    #                         data_dict['answers'] = answers

    #                         outputs, targets, predictions = rnn(data_dict, args.task)

    #                         loss = criterion(outputs, targets)
    #                         acc = rnn.accuracy(predictions, targets)

    #                         losses.update(loss.item(), video_features.size(0))
    #                         accuracy.update(acc.item(), video_features.size(0))
                            

    #                     ######################################################################
    #                     if best_val_acc < accuracy.avg:
    #                         best_val_acc = accuracy.avg
    #                         best_val_iter = iter
    #                         args.plotter.text_plot(args.task+args.jobname+" val", str(round(best_val_acc, 4))+" val "+str(args.jobname)+" "+str(iter))
    #                     #args.plotter.plot("loss", "val", args.jobname, iter, losses.avg)
    #                     args.plotter.plot("accuracy", "val", args.jobname+" "+args.task, iter, accuracy.avg)
    #                     ######################################################################

    #                     print('[Val] iter %d, loss %.3f, acc %.2f, best acc %.3f at iter %d' % (
    #                                     iter, losses.avg, accuracy.avg, best_val_acc, best_val_iter))
    #                     torch.save(rnn.state_dict(), os.path.join(args.save_model_path, 'rnn-%04d-l%.3f-a%.3f.pkl' % (
    #                                     iter, losses.avg, accuracy.avg)))

    #                 if 1==1:
    #                     losses = AverageMeter()
    #                     accuracy = AverageMeter()
    #                     n_iter = len(test_dataset) / args.batch_size

    #                     iter_test = 0
    #                     for batch_chunk in test_dataset.batch_iter(1, args.batch_size, shuffle=False):
    #                         if iter_test % 10 == 0:
    #                             print('%d/%d' % (iter_test, n_iter))

    #                         iter_test+=1

    #                         video_features = torch.from_numpy(batch_chunk['video_features'].astype(np.float32)).cuda()
    #                         video_lengths = batch_chunk['video_lengths']
    #                         question_words = torch.from_numpy(batch_chunk['question_words'].astype(np.int64)).cuda()
    #                         question_lengths = batch_chunk['question_lengths']
    #                         answers = torch.from_numpy(batch_chunk['answer'].astype(np.int64)).cuda()

    #                         data_dict = {}
    #                         data_dict['video_features'] = video_features
    #                         data_dict['video_lengths'] = video_lengths
    #                         data_dict['question_words'] = question_words
    #                         data_dict['question_lengths'] = question_lengths
    #                         data_dict['answers'] = answers

    #                         outputs, targets, predictions = rnn(data_dict, args.task)

    #                         loss = criterion(outputs, targets)
    #                         acc = rnn.accuracy(predictions, targets)

    #                         losses.update(loss.item(), video_features.size(0))
    #                         accuracy.update(acc.item(), video_features.size(0))
    #                         #break

                        
    #                     ######################################################################
    #                     #import ipdb; ipdb.set_trace()
    #                     if best_test_acc < accuracy.avg:
    #                         best_test_acc = accuracy.avg
    #                         best_val_iter = iter
    #                         args.plotter.text_plot(args.task+args.jobname+" test", args.task+args.jobname+" test "+str(round(best_test_acc, 4))+" "+str(iter))
    #                     #args.plotter.plot("loss", "test", args.jobname, iter, losses.avg)
    #                     args.plotter.plot("accuracy", "test", args.jobname+" "+args.task, iter, accuracy.avg)
    #                     ######################################################################

    #                     print('[Test] iter %d, loss %.3f, acc %.2f' % (iter,losses.avg,accuracy.avg))
    #                     if args.test==1:
    #                         exit()

    #             rnn.train()

    #         iter += 1

                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=int , default=0,
                        help='1-directly test, 0-normal training')
    parser.add_argument('--task', type=str , default='Count',
                         help='[Count, Action, FrameQA, Trans]')
    parser.add_argument('--feat_type', type=str , default='SpTp', 
                         help='[C3D, Resnet, Concat, Tp, Sp, SpTp]')
    parser.add_argument('--save_path', type=str, default=os.path.expanduser('~/kable_management/blp_paper/.results/tgif/tgifsaved_models'),
                        help='path for saving trained models')
    parser.add_argument('--hidden_size', type=int , default=512 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=2,
                        help='number of layers in lstm')                        
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--momentum', default=0.9, type=float,  help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')

    #######
    #######
    parser.add_argument('--mode', default='train', help='train/test')
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
    parser.add_argument("--hard_restrict", action="store_true", help="strictly enforce the max length frame")
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

    plotter = VisdomLinePlotter(env_name=args.jobname)
    args.plotter = plotter
    args.normalize_v = not args.no_normalize_v
    args.device = torch.device("cuda:%d" % args.device if args.device >= 0 else "cpu")
    args.with_ts = not args.no_ts
    args.input_streams = [] if args.input_streams is None else args.input_streams
    args.vid_feat_flag = True if "imagenet" in args.input_streams else False
    args.h5driver = None if args.no_core_driver else "core"

    plotter = VisdomLinePlotter(env_name=args.jobname)
    args.plotter = plotter
    print(args)
    main(args)
