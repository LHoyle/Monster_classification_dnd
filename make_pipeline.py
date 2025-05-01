#!/usr/bin/env python3

################################################################
#
# These custom functions help with constructing common pipelines.
# They make use of my_args, and object that has been configured
# by the argparse module to match user requests.
#
from pipeline_elements import *
import sklearn.impute
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
import sklearn.tree

def make_numerical_predictor_params(my_args):
    params = { 
        "features__numerical__numerical-features-only__do_predictors" : [ True ],
        "features__numerical__numerical-features-only__do_numerical" : [ True ],
    }
    if my_args.numerical_missing_strategy:
        params["features__numerical__missing-data__strategy"] = [ 'median', 'mean', 'most_frequent' ]
    if my_args.use_polynomial_features:
        params["features__numerical__polynomial-features__degree"] = [ 2 ] # [ 1, 2, 3 ]

    return params

def make_categorical_predictor_params(my_args):
    params = { 
        "features__categorical__categorical-features-only__do_predictors" : [ True ],
        "features__categorical__categorical-features-only__do_numerical" : [ False ],
        "features__categorical__encode-category-bits__categories": [ 'auto' ],
        "features__categorical__encode-category-bits__handle_unknown": [ 'ignore' ],
    }
    if my_args.categorical_missing_strategy:
        params["features__categorical__missing-data__strategy"] = [ 'most_frequent' ]
    return params

def make_predictor_params(my_args):
    p1 = make_numerical_predictor_params(my_args)
    p2 = make_categorical_predictor_params(my_args)
    p1.update(p2)
    return p1

def make_tree_params(my_args):
    tree_params = {
        # "model__criterion": [ "entropy" ], # [ "entropy", "gini" ],
        "model__criterion":  [ "entropy", "gini" ],
        
        "model__splitter": [ "best" ], # [ "best", "random" ],
        # "model__splitter":  [ "best", "random" ],
        
        "model__max_depth": [ 1, 2, 3, 4, None ],

        "model__min_samples_split": [ 2 ], # [ 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64 ],
        # "model__min_samples_split":  [ 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64,2 ],

        "model__min_samples_leaf":  [ 1 ],  # [ 0.01, 0.02, 0.04, 0.1 ],
        # "model__min_samples_leaf":   [ 0.01, 0.02, 0.04, 0.1,1 ],

        # "model__max_features":  [ None ], # [ "sqrt", "log2", None ],
        "model__max_features":  [ "sqrt", "log2", None ],
        
        "model__max_leaf_nodes": [ None ], # [ 2, 4, 8, 16, 32, 64, None ],
        # "model__max_leaf_nodes": [ 2, 4, 8, 16, 32, 64, None ],
        
        "model__min_impurity_decrease": [ 0.0 ], # [ 0.0, 0.01, 0.02, 0.04, 0.1, 0.2 ],
        # "model__min_impurity_decrease":  [ 0.0, 0.01, 0.02, 0.04, 0.1, 0.2 ],
    }
    return tree_params

def make_forest_params(my_args):
    forest_params = {
        "model__criterion": [ "gini" ], # [ "entropy", "gini", "log_loss" ],
        # "model__criterion":  [ "entropy", "gini", "log_loss" ],
        
        "model__max_depth": [ 1, 2, 3, 4, None ],

        "model__min_samples_split": [ 2 ], # [ 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64 ],
        # "model__min_samples_split":  [ 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64,2 ],

        "model__min_samples_leaf":  [ 1 ],  # [ 0.01, 0.02, 0.04, 0.1 ],
        # "model__min_samples_leaf":   [ 0.01, 0.02, 0.04, 0.1,1 ],

        "model__min_weight_fraction_leaf":  [ 0.1 ],  # [ 0.1, 0.2, 0.3, 0.4, 0.5 ],
        # "model__min_weight_fraction_leaf":  [ 1 ],  # [ 0.1, 0.2, 0.3, 0.4, 0.5 ],

        "model__max_features":  [ None ], # [ "sqrt", "log2", None ],
        # "model__max_features":  [ "sqrt", "log2", None ],

        
        "model__max_leaf_nodes": [ None ], # [ 2, 4, 8, 16, 32, 64, None ],
        # "model__max_leaf_nodes": [ 2, 4, 8, 16, 32, 64, None ],
        
        "model__min_impurity_decrease": [ 0.0 ], # [ 0.0, 0.01, 0.02, 0.04, 0.1, 0.2 ],
        # "model__min_impurity_decrease":  [ 0.0, 0.01, 0.02, 0.04, 0.1, 0.2 ],
    }
    return forest_params


def make_boost_params(my_args):
    boost_params = {
        #"model__loss":  ['log_loss'], #['log_loss', 'exponential'],
        "model__loss":  ['log_loss', 'exponential'],

        #can be from 0.0 to inf
        "model__learning_rate": [ 0.1], # [ 0.0, 0.1,1 ], 
        # "model__learning_rate":  [ 0.0, 0.1,1 ],
        
        #can be from 0.0 to 1.0
        "model__subsample": [ 0.25, 1.0 ], # [ 0.0, 0.25, 0.50, 0.75, 1.0 ]
        # "model__subsample": [ 0.0, 0.25, 0.50, 0.75, 1.0 ],
        
        #"model__criterion": [ "friedman_mse" ], # [ "friedman_mse", "squared_error" ],
        "model__criterion":  [ "friedman_mse", "squared_error" ],        

        "model__min_samples_split": [ 2 ], # [ 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64 ],
        # "model__min_samples_split":  [ 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64,2 ],

        "model__min_samples_leaf":  [ 1 ],  # [ 0.01, 0.02, 0.04, 0.1 ],
        # "model__min_samples_leaf":   [ 0.01, 0.02, 0.04, 0.1,1 ],

        "model__min_weight_fraction_leaf":  [ 0.0 ],  # [  0.0, 0.1, 0.2, 0.3, 0.4, 0.5 ],
        # "model__min_weight_fraction_leaf":   [ 0.0,0.1, 0.2, 0.3, 0.4, 0.5 ],
       
        "model__max_depth": [ 1, 2, 3, 4, None ],

        #can be from 0.0 to inf
        "model__min_impurity_decrease": [ 0.0 ], # [ 0.0, 0.01, 0.02, 0.04, 0.1, 0.2 ],
        # "model__min_impurity_decrease":  [ 0.0, 0.01, 0.02, 0.04, 0.1, 0.2 ],
       

        #"model__max_features":  [ None ], # [ "sqrt", "log2", None ],
        "model__max_features":  [ "sqrt", "log2", None ],
        
        "model__max_leaf_nodes": [ None ], # [ 2, 4, 8, 16, 32, 64, None ],
        # "model__max_leaf_nodes": [ 2, 4, 8, 16, 32, 64, None ],
        
    }
    return boost_params


def make_SVM_params(my_args):
    SVM_params = {
        "model__kernel":  [ 'rbf'], #[ 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' ],
        # "model__kernel":  [ 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' ],

        #must be non negative
        "model__degree":  [3], #[ 0,1,2,3,4 ],
        # "model__degree":  [ 0,1,2,3,4 ],

        "model__gamma":  ['scale'], #['scale', 'auto],
        # "model__gamma":  ['scale', 'auto],
        
        #float? only siginifant in poly and sigmoid.
        "model__coef0":  [0.0], #[ 0.0, 0.25, 0.50, 0.75, 1.0 ],
        # "model__coef0":  [ 0.0, 0.25, 0.50, 0.75, 1.0 ],

        #tolerance float, default=1e-3
        "model__tol":  [0.001], #[ 0.001, 0.002, 0.003, 0.004, 0.005 ],
        # "model__tol":  [ 0.001, 0.002, 0.003, 0.004, 0.005 ],

        #C, regularization parameter, must be positive: float, default=1.0
        "model__C":  [1.0], #[ 0.0, 0.25, 0.50, 0.75, 1.0 ],
        # "model__C":  [ 0.0, 0.25, 0.50, 0.75, 1.0 ],

        #epsilon, 
        # "Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty 
        # is associated in the training loss function with points predicted within a distance epsilon
        #  from the actual value. Must be non-negative."
        #  must be positive: float, default=1.0
        # "model__epsilon":  [0.1], #[ 0.1, 0.2, 0.3, 0.4, 0.5 ],
        # "model__epsilon":  [ 0.1, 0.2, 0.3, 0.4, 0.5 ],
        #max iterations
        "model__max_iter":  [-1], #[ 10,100,1000,-1 ],
        # "model__max_iter":  [ 10,100,1000,-1 ],
    }
    return SVM_params


def make_linear_params(my_args):
    linear_params = {
        #regularization strenght, must be a positive float 
        #float default 1.0
        "model__alpha":  [0.1], #[ 0.1, 0.2, 0.3, 0.4, 0.5 ],
        # "model__alpha":  [ 0.1, 0.2, 0.3, 0.4, 0.5 ],

        "model__fit_intercept":  [True], #[ True,False ],
        # "model__fit_intercept":  [ True,False],
        
        "model__copy_X":  [True], #[ True,False ],
        # "model__copy_X":  [ True,False],

        # "model__max_iter":  [-1], #[ 10,100,1000,-1 ],
        # # "model__max_iter":  [ 10,100,1000,-1 ],

        #tolerance float, default=1e-3
        "model__tol":  [0.001], #[ 0.001, 0.002, 0.003, 0.004, 0.005 ],
        # "model__tol":  [ 0.001, 0.002, 0.003, 0.004, 0.005 ],

        "model__solver":  ['auto'], #[ 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs' ],
        # "model__solver":  [ 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs' ],
        
    }
    return linear_params

def make_SGD_params(my_args):
    SGD_params = {
        "model__loss":  ['hinge'], #[ 'hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive' ],
        # "model__loss":  [ 'hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive' ],
        
        "model__penalty":  ['l2'], #[ 'l2', 'l1', 'elasticnet', None ],
        # "model__penalty":  [ 'l2', 'l1', 'elasticnet', None ],

        #regularization strenght, must be a positive float
        # float default 0.0001 
        "model__alpha":  [0.0001], #[ 0.0001, 0.0002, 0.0003, 0.0004, 0.0005 ],
        # "model__alpha":  [ 0.0001, 0.0002, 0.0003, 0.0004, 0.0005 ],

        #regularization strenght, must be a positive float
        # float default 0.15 must be between 0.0 and 1.0 
        "model__alpha":  [0.15], #[ 0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 1.0 ],
        # "model__alpha":  #[ 0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 1.0 ],

        #unlike the others cannot be -1, so 1000 is the default maximum
        "model__max_iter":  [1000], #[ 100,1000,10000,100000, 1000000 ],
        # "model__max_iter":  [ 100,1000,10000,100000, 1000000 ],

        #tolerance float, default=1e-3
        "model__tol":  [0.001], #[ 0.001, 0.002, 0.003, 0.004, 0.005 ],
        # "model__tol":  [ 0.001, 0.002, 0.003, 0.004, 0.005 ],
        
    }
    return SGD_params



def make_fit_params(my_args):
    params = make_predictor_params(my_args)
    if my_args.model_type == "SGD":
        model_params = make_SGD_params(my_args)
    elif my_args.model_type == "linear":
        model_params = make_linear_params(my_args)
    elif my_args.model_type == "SVM":
        model_params = make_SVM_params(my_args)
    elif my_args.model_type == "boost":
        model_params = make_boost_params(my_args)
    elif my_args.model_type == "forest":
        model_params = make_forest_params(my_args)
    elif my_args.model_type == "tree":
        model_params = make_tree_params(my_args)
    else:
        raise Exception("Unknown model type: {} [SGD, linear, SVM, boost, forest]".format(my_args.model_type))

    params.update(model_params)
    return params

def make_numerical_feature_pipeline(my_args):
    items = []

    items.append(("numerical-features-only", DataFrameSelector(do_predictors=True, do_numerical=True)))

    if my_args.numerical_missing_strategy:
        items.append(("missing-data", sklearn.impute.SimpleImputer(strategy=my_args.numerical_missing_strategy)))
    if my_args.use_polynomial_features:
        items.append(("polynomial-features", sklearn.preprocessing.PolynomialFeatures(degree=my_args.use_polynomial_features)))
    if my_args.use_scaler:
        items.append(("scaler", sklearn.preprocessing.StandardScaler()))
    items.append(("noop", PipelineNoop()))
    
    numerical_pipeline = sklearn.pipeline.Pipeline(items)
    return numerical_pipeline


def make_categorical_feature_pipeline(my_args):
    items = []
    
    items.append(("categorical-features-only", DataFrameSelector(do_predictors=True, do_numerical=False)))

    if my_args.categorical_missing_strategy:
        items.append(("missing-data", sklearn.impute.SimpleImputer(strategy=my_args.categorical_missing_strategy)))
    items.append(("encode-category-bits", sklearn.preprocessing.OneHotEncoder(categories='auto', handle_unknown='ignore')))

    categorical_pipeline = sklearn.pipeline.Pipeline(items)
    return categorical_pipeline

def make_feature_pipeline(my_args):
    """
    Numerical features and categorical features are usually preprocessed
    differently. We split them out here, preprocess them, then merge
    the preprocessed features into one group again.
    """
    items = []

    items.append(("numerical", make_numerical_feature_pipeline(my_args)))
    items.append(("categorical", make_categorical_feature_pipeline(my_args)))
    pipeline = sklearn.pipeline.FeatureUnion(transformer_list=items)
    return pipeline


def make_fit_pipeline_regression(my_args):
    """
    These are all regression models.
    """
    items = []
    items.append(("features", make_feature_pipeline(my_args)))
    if my_args.model_type == "SGD":
        items.append(("model", sklearn.linear_model.SGDRegressor(max_iter=10000, n_iter_no_change=100, penalty=None))) # verbose=3, 
    elif my_args.model_type == "linear":
        items.append(("model", sklearn.linear_model.LinearRegression()))
    elif my_args.model_type == "SVM":
        items.append(("model", sklearn.svm.SVR()))
    elif my_args.model_type == "boost":
        items.append(("model", sklearn.ensemble.GradientBoostingRegressor()))
    elif my_args.model_type == "forest":
        items.append(("model", sklearn.ensemble.RandomForestRegressor()))
    elif my_args.model_type == "tree":
        items.append(("model", sklearn.tree.DecisionTreeRegressor()))
    else:
        raise Exception("Unknown model type: {} [SGD, linear, SVM, boost, forest]".format(my_args.model_type))

    return sklearn.pipeline.Pipeline(items)

def make_fit_pipeline_classification(my_args):
    """
    These are all classification models.
    """
    items = []
    items.append(("features", make_feature_pipeline(my_args)))
    if my_args.model_type == "SGD":
        items.append(("model", sklearn.linear_model.SGDClassifier(max_iter=10000, n_iter_no_change=100, penalty=None))) # verbose=3, 
    elif my_args.model_type == "linear":
        items.append(("model", sklearn.linear_model.RidgeClassifier()))
    elif my_args.model_type == "SVM":
        items.append(("model", sklearn.svm.SVC(probability=True)))
    elif my_args.model_type == "boost":
        items.append(("model", sklearn.ensemble.GradientBoostingClassifier()))
    elif my_args.model_type == "forest":
        items.append(("model", sklearn.ensemble.RandomForestClassifier()))
    elif my_args.model_type == "tree":
        items.append(("model", sklearn.tree.DecisionTreeClassifier()))
    else:
        raise Exception("Unknown model type: {} [SGD, linear, SVM, boost, forest]".format(my_args.model_type))

    return sklearn.pipeline.Pipeline(items)

def make_fit_pipeline(my_args):
    return make_fit_pipeline_classification(my_args)
