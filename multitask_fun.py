import os
import theano.tensor as T
import theano
import numpy
import random as R
import cPickle
from collections import OrderedDict
import theano.sandbox.rng_mrg as RNG_MRG
import theano.sandbox.rng_mrg as RNG_MRG
from eric_tools.mlp_utils import *

cos_dist    =   lambda a,b : numpy.dot(a,b) / (((a**2).sum()**0.5) * ((b**2).sum()**0.5))
enum_pairs  =   lambda n : reduce(lambda x,y : x+y, [[(i,j) for j in range(i+1,n)] for i in range(n-1)], [])

def experiment(state, channel=None):
    if __name__ == "__main__":
        state.no_multitask      =   True      
      
        state.n_hid             =   1000
        state.n_hid_last        =   100 # last layer
        state.n_layers          =   3
        state.sparse_init       =   True
       
        state.batch_size        =   10
        state.learning_rate     =   0.1
        state.adptv_decay       =   1.
        state.momentum          =   0.95
        
        state.input_dropout     =   0.1
        state.dropout           =   0.5
        state.hidden_noise_post =   0.
        state.hidden_noise_pre  =   0.

        state.fun_weight_cost   =   0.5

        state.L1_weights        =   1e-5
        state.L2_weights        =   0

        state.L1_output         =   0
        state.L2_output         =   0

        state.L1_hiddens        =   0

        state.log_normalized    =   False
        state.activation        =   'rectifier'

        state.base_data_path    =   '/data/lisa/exp/viaugaut/mq/data'
        
    print state

    #----------------------------------------------------------------------
    #                        Load/split the data
    #----------------------------------------------------------------------
    import numpy
    # Load data    
    
    #data_path   =   '/data/lisa/exp/viaugaut/mq/data/pack13/skew_kurt/user_attr/log_normalized'
    #data_path   =   '/data/lisa/exp/viaugaut/mq/data/pack11/base_attr/log_normalized/'
    #data_path   =   os.path.join(state.base_data_path, 'pack11/base_attr/log_normalized/')
    data_path   =   os.path.join(state.base_data_path, 'pack13/skew_kurt/user_attr/log_normalized')


    #diff_attr_size  =   207
    diff_attr_size  =   9999999999999

    train_X     =   theano.shared(cast32(numpy.load(os.path.join(data_path, 'train_X.npy'))))
    valid_X     =   theano.shared(cast32(numpy.load(os.path.join(data_path, 'valid_X.npy'))))
    test_X      =   theano.shared(cast32(numpy.load(os.path.join(data_path, 'test_X.npy'))))
   
    data_path   =   os.path.join(state.base_data_path, 'pack13/skew_kurt/user_attr/log_normalized')

    train_Y     =   numpy.load(os.path.join(data_path, 'train_Y.npy'))
    valid_Y     =   numpy.load(os.path.join(data_path, 'valid_Y.npy'))
    test_Y     =   numpy.load(os.path.join(data_path, 'test_Y.npy'))

    # normalize dcount
    max_death   =   max(train_Y.max(axis=0)[1], valid_Y.max(axis=0)[1], test_Y.max(axis=0)[1])
    train_Y[:,1]    /=  max_death
    valid_Y[:,1]    /=  max_death
    test_Y[:,1]     /=  max_death

    if state.no_multitask:
        train_Y =   train_Y[:, -1:]
        valid_Y =   valid_Y[:, -1:]
        test_Y =   test_Y[:, -1:]

    train_Y     =   theano.shared(cast32(train_Y))
    valid_Y     =   theano.shared(cast32(valid_Y))
    test_Y     =   theano.shared(cast32(test_Y))

    #----------------------------------------------------------------------
    #                           Create the model
    #----------------------------------------------------------------------
    numpy.random.seed(1)

    # TODO add option to init sparse weights

    X   =   T.fmatrix()    
    Y   =   T.fmatrix()
    index   =   T.lscalar()

    n_layers    =   state.n_layers
    n_hid       =   state.n_hid
    n_hid_last  =   state.n_hid_last
    sizes       =   [train_X.get_value().shape[1]] + (n_layers-1) * [n_hid] + [n_hid_last] + [train_Y.get_value().shape[1]]

    if not state.sparse_init:  
        weights =   [get_shared_weights(sizes[i], sizes[i+1]) for i in range(n_layers)]
    else:
        weights =   [get_sparse_shared_weights(sizes[i], sizes[i+1], n_nonzeros=15) for i in range(n_layers)]
    
    weights +=  [get_shared_weights(sizes[-2], sizes[-1])]
    biases  =   [get_shared_bias(sizes[i+1], 'b') for i in range(n_layers+1)]

    L1_H        =   theano.shared(cast32(state.L1_hiddens))
    L1_hiddens  =   theano.shared(cast32(0))
   
    L1_W    =   theano.shared(cast32(state.L1_weights))
    L2_W    =   theano.shared(cast32(state.L2_weights))
    L1_O    =   theano.shared(cast32(state.L1_output))
    L2_O    =   theano.shared(cast32(state.L2_output))

    all_coeff   =   (L1_W, L2_W, L1_O, L2_O)
    L1  =   lambda x:abs(x).sum()
    L2  =   lambda x:(x**2).sum()

    weight_decay    =   numpy.sum([L1_W * L1(w) for w in weights[:-1]]) + \
                        numpy.sum([L2_W * L2(w) for w in weights[:-1]]) + \
                        L1_O * L1(weights[-1]) + \
                        L2_O * L2(weights[-1])
   
    if state.activation == 'rectifier':
        act_fun =   rectifier
    elif state.activation == 'tanh':
        act_fun =   lambda x:T.tanh(x)
    elif state.activation == 'sigmoid':
        act_fun =   lambda x:T.nnet.sigmoid(x)

    # f prop with dropout
    if state.input_dropout:
        train_out =   dropout(X, p=state.input_dropout)
    dropped_out_layers  =   []
    
    for i in range(n_layers+1):
        # Compute preactivation
        act     =   T.dot(train_out, weights[i]) + biases[i]

        # if layer is not output or bef
        if i < (n_layers-1) or (i==(n_layers-1) and n_layers==1):
        #if i < n_layers:
            if state.hidden_noise_pre:
                act     =   add_gaussian_noise(act, std = state.hidden_noise_pre)

            train_out   =   act_fun(act)

            if state.dropout:
                train_out   =   dropout(train_out, p=state.dropout)
                dropped_out_layers.append(i)

            if state.hidden_noise_post:
                train_out   =   add_gaussian_noise(train_out, std = state.hidden_noise_post)
            L1_hiddens  +=  (L1_H * abs(train_out).sum())

        elif i == (n_layers-1):
            train_out   =   act_fun(act)
        
        elif i == n_layers:
            train_out =   T.nnet.sigmoid(act)
    
    # f prop withOUT dropout
    test_out =   X
    if state.input_dropout:
        test_out    *=  cast32(1 - state.input_dropout)
    for i in range(n_layers+1):
        act     =   T.dot(test_out, weights[i]) + biases[i]
        if i != n_layers:
            test_out =   act_fun(act)
            if i in dropped_out_layers:
                test_out    *=  cast32(1 - state.dropout)
        elif i == n_layers:
            test_out =   T.nnet.sigmoid(act)
   
    train_fun_out   =   train_out[:,-1]
    test_fun_out    =   test_out[:,-1]
  
    multitask_cost  =   T.mean(T.nnet.binary_crossentropy(train_out, Y))
    fun_cost        =   T.mean(T.nnet.binary_crossentropy(train_fun_out, Y[:,-1]))
    
    reg_cost        =   L1_hiddens + weight_decay

    # FUNCTION COMPILING
    
    params          =   weights + biases 

    lr              =   theano.shared(cast32(state.learning_rate))
    momentum        =   theano.shared(cast32(state.momentum))

    # multitask momentum gradient
    gradient        =   T.grad(multitask_cost + reg_cost, params)
    gradient_buffer =   [theano.shared(numpy.zeros(x.get_value().shape, dtype='float32')) for x in params]

    m_gradient      =   [momentum * gb + (cast32(1) - momentum) * g for (gb, g) in zip(gradient_buffer, gradient)]

    g_updates       =   [(p, p - lr * mg) for (p, mg) in zip(params, m_gradient)]
    b_updates       =   zip(gradient_buffer, m_gradient)

    updates         =   OrderedDict(g_updates + b_updates)

    f_learn =   theano.function(inputs  =   [index],
                                updates =   updates,
                                givens  =   {X  :   train_X[index * state.batch_size:(index+1) * state.batch_size],
                                             Y  :   train_Y[index * state.batch_size:(index+1) * state.batch_size]},
                                outputs =   T.mean(T.nnet.binary_crossentropy(test_out, Y), axis=0))

    f_cost_v =  theano.function(inputs  =   [index],
                                givens  =   {X  :   valid_X[index * state.batch_size:(index+1) * state.batch_size],
                                             Y  :   valid_Y[index * state.batch_size:(index+1) * state.batch_size]},
                                outputs =   T.mean(T.nnet.binary_crossentropy(test_out, Y), axis=0))

    f_cost_t =  theano.function(inputs  =   [index],
                                givens  =   {X  :   test_X[index * state.batch_size:(index+1) * state.batch_size],
                                             Y  :   test_Y[index * state.batch_size:(index+1) * state.batch_size]},
                                outputs =   T.mean(T.nnet.binary_crossentropy(test_out, Y), axis=0))

    
    # TRAINING PRELIMINARIES
    train_costs     =   []
    valid_costs     =   []
    test_costs      =   []

    best_params     =   []
    best_VALID_MULTI     =  numpy.inf
    best_VALID_FINE      =  numpy.inf

    previous_valid  =   None
    best_params     =   [x.get_value(borrow=False) for x in params]

    STOP            =   False
    FINETUNE        =   False

    K_multitask     =   0
    K_finetune      =   0
    EARLY_STOP_VAL  =   best_VALID_MULTI
    ES_THRESH       =   15

    epoch_count     =   0

    batch_size      =   state.batch_size

    # DA TRAIN LOOP
    R.seed(1)
    while not STOP:
        epoch_count     +=1
        epoch_cost      = []
    
        learn   =   f_learn
        if FINETUNE:
            print 'finetuning'
            learn   =   f_learn_fun
        
        # Train        
        for i in range(train_X.get_value().shape[0] // batch_size):
            epoch_cost.append(learn(i))
            
        epoch_cost  =   numpy.vstack(epoch_cost).mean(axis=0)
        train_costs.append(epoch_cost)

        # Valid
        valid_cost  =   []
        for i in range(valid_X.get_value().shape[0] // batch_size):
            valid_cost.append(f_cost_v(i))
        valid_cost  =   numpy.vstack(valid_cost).mean(axis=0)
        valid_costs.append(valid_cost)

        # Test
        test_cost  =   []
        for i in range(test_X.get_value().shape[0] // batch_size):
            test_cost.append(f_cost_t(i))
        test_cost  =   numpy.vstack(test_cost).mean(axis=0)
        test_costs.append(test_cost)

        # Print
        print epoch_count, 'Train    :',train_costs[-1],' Valid  :',valid_costs[-1],' Test   :', test_costs[-1], ' LR    :', lr.get_value(),

        # (early) stopping
        if epoch_count>1000 or numpy.isnan(numpy.mean(valid_costs[-1])) or STOP:
            print 'Done training!'
            break

        print 'Fix finetuning'
        if not FINETUNE:
            valid_multi =   numpy.mean(valid_costs[-1])
            if (valid_multi < best_VALID_MULTI):
                print 'best!'
                K_multitask         =   0
                best_VALID_MULTI    =   valid_multi                
                best_params         =   [x.get_value(borrow=False) for x in params]
            else:
                print
                K_multitask         +=  1
                #[p.set_value(previous_p) for (p, previous_p) in zip(params, best_params)]
                lr.set_value(cast32(state.adptv_decay) * lr.get_value())

            if (K_multitask >= ES_THRESH):
                FINETUNE    =   True                
                print 'Beginning finetuning!'
                [p.set_value(previous_p) for (p, previous_p) in zip(params, best_params)]

        elif FINETUNE and not STOP:
            valid_fine  =   valid_costs[-1][-1]
            if (valid_fine < best_VALID_FINE):
                print 'best!'
                K_finetune          =   0 
                best_VALID_FINE     =   valid_fine
                best_params         =   [x.get_value(borrow=False) for x in params]
            else:
                print
                K_finetune          +=  1
                #[p.set_value(previous_p) for (p, previous_p) in zip(params, best_params)]
                lr.set_value(cast32(state.adptv_decay) * lr.get_value())

            if (K_finetune >= ES_THRESH):
                STOP    =   True
    
    train_costs =   [list(x) for x in train_costs]
    valid_costs =   [list(x) for x in valid_costs]
    test_costs  =   [list(x) for x in test_costs]

    state.train_costs   =   train_costs
    state.valid_costs   =   valid_costs
    state.test_costs    =   test_costs

    numpy.save('params.npy', numpy.array(best_params)) 

    # save final train, valid, test predictions

    f_pred = theano.function(inputs = [X], outputs = test_fun_out)
    
    train_pred  =   numpy.concatenate([f_pred(train_X.get_value()[i*100:(i+1)*100]) for i in range(len(train_X.get_value())/100 + 1)])
    valid_pred  =   numpy.concatenate([f_pred(valid_X.get_value()[i*100:(i+1)*100]) for i in range(len(valid_X.get_value())/100 + 1)])
    test_pred   =   numpy.concatenate([f_pred(test_X.get_value()[i*100:(i+1)*100]) for i in range(len(test_X.get_value())/100 + 1)])
   
    numpy.save('train_predictions.npy', train_pred)
    numpy.save('valid_predictions.npy', valid_pred)
    numpy.save('test_predictions.npy', test_pred)

    # Network statistics

    return channel.COMPLETE

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    experiment(args)
    print 'done'

