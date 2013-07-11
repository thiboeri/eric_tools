import numpy, os
from eric_tools import mDA
from eric_tools.mlp_utils import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import argparse

def load_data():
    # load data
    #data_path = '/data/lisa/exp/viaugaut/mq/data/pack13/skew_kurt/user_attr/log_normalized'
    data_path = '/data/lisa/data/ubi/hyperquest/data/pack14/'

    #train_X     =   cast32(numpy.load(os.path.join(data_path, 'train_X.npy')))
    #valid_X     =   cast32(numpy.load(os.path.join(data_path, 'valid_X.npy')))
    #test_X      =   cast32(numpy.load(os.path.join(data_path, 'test_X.npy')))

    train_X     =   cast32(numpy.load(os.path.join(data_path, 'train_diff_cached_X.npy')))
    valid_X     =   cast32(numpy.load(os.path.join(data_path, 'valid_diff_cached_X.npy')))
    test_X      =   cast32(numpy.load(os.path.join(data_path, 'test_diff_cached_X.npy')))

    train_X = train_X.T[train_X.std(axis=0)!=0].T
    valid_X = valid_X.T[train_X.std(axis=0)!=0].T
    test_X = test_X.T[train_X.std(axis=0)!=0].T


    train_Y     =   numpy.load(os.path.join(data_path, 'train_diff_cached_Y.npy'))
    valid_Y     =   numpy.load(os.path.join(data_path, 'valid_diff_cached_Y.npy'))
    test_Y     =   numpy.load(os.path.join(data_path, 'test_diff_cached_Y.npy'))
    
    return (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y)

def experiment(state, channel):

    if __name__ == "__main__":
        state.n_layers = 1
        state.corruption = 0.25
       
        state.dimension_boost = False
        
        state.n_estimators      = 25
        state.max_depth         = None
        state.min_samples_split = 1
        state.min_samples_leaf  = 1
        state.min_density       = 0.1
        state.max_features      = "auto"
        state.bootstrap         = True

    (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = load_data()

    if state.dimension_boost:
        # choice of random matrix...
        #random_matrix = numpy.random.normal(size=(train_X.shape[1], dimension_boost))
        print 'Boosting dimension through random matrix'

    # choose regressor, init HP
    cls = RandomForestRegressor(n_estimators = state.n_estimators,
                                max_depth = state.max_depth,
                                min_samples_split = state.min_samples_split,
                                min_samples_leaf = state.min_samples_leaf,
                                min_density = state.min_density,
                                max_features = state.max_features,
                                bootstrap = state.bootstrap)

    # smDA
    print 'Computing stacked mDA'
    n_layers = state.n_layers
    corruption = state.corruption

    W, h, Z = 0, train_X, [valid_X, test_X]

    for i in range(n_layers):
        W, h, Z = mDA.mDA(h, corruption, Z)

    # Train regressor
    print 'training classifier'
    cls.fit(h, train_Y)

    train_pred = numpy.clip(0.99, 0.01, cls.predict(h)).flatten()
    valid_pred = numpy.clip(0.99, 0.01, cls.predict(Z[0])).flatten()
    test_pred = numpy.clip(0.99, 0.01, cls.predict(Z[1])).flatten()

    print 'train ce : ', binary_xce(train_pred, train_Y.flatten()).mean()
    print 'valid ce : ', binary_xce(valid_pred, valid_Y.flatten()).mean()
    print 'test  ce : ', binary_xce(test_pred, test_Y.flatten()).mean()

    state.train_costs = [binary_xce(train_pred, train_Y.flatten()).mean()]
    state.valid_costs = [binary_xce(valid_pred, valid_Y.flatten()).mean()]
    state.test_costs = [binary_xce(test_pred, test_Y.flatten()).mean()]

    return channel.COMPLETE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    args = parser.parse_args()
    experiment(args, None)
