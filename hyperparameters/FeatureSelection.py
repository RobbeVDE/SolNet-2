from optuna.trial import TrialState
import logging
import sys
import optuna

def feature_selection(trial, features):
    """
    Function that gives updated features, using Optuna HP optimization
    note: if sin or cos then both should be included, so next one is included and next step is just skipped and put aggain to false
    """
    sel_features = []
    prev_sincos = False
    for i,feat in enumerate(features):
        sel = trial.suggest_catagorical(feat,[True, False])
        if ~prev_sincos:
            sel_features.append(sel)
        else:
            prev_sincos = False
        
        if ("_sin" in feat) or ("_cos" in feat):
            sel_features.append(sel)
            prev_sincos = True

    
    return sel_features

                
            
            