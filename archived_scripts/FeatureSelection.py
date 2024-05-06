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
    for feat in features:
        if ("_sin" in feat) or ("_cos" in feat):
            if not prev_sincos:
                sel = trial.suggest_categorical(feat,[True, False])
                sel_features.append(sel)
                sel_features.append(sel)
                prev_sincos = True
            else:
                prev_sincos = False
        
        else:
            sel = trial.suggest_categorical(feat,[True, False])
            sel_features.append(sel)
    
    return sel_features

                
            
            