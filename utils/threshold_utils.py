import numpy as np
import pandas as pd

def adj_threshold(s_test,y_pred_proba,threshold_women, threshold_men=0.5):
    y_proba_adj = np.where(((s_test ==0)&(y_pred_proba>threshold_women))|
                        ((s_test ==1)&(y_pred_proba>threshold_men)), 
                        1,0)
    return y_proba_adj

def find_recall(mask, y_proba):
    total = sum(mask)
    correct = sum((y_proba ==1)&mask)
    recall = correct/total if total>0 else 0 
    return total, correct, recall


def eval_subgroups(y_test,y_proba_adj,s_test,womenthreshold): 
    #set up masks per class
    mask_highinc_women = (s_test ==0)&(y_test==1)
    mask_lowinc_women = (s_test ==0)&(y_test==0)
    mask_highinc_men = (s_test ==1)&(y_test==1)
    mask_lowinc_men = (s_test ==1)&(y_test==0)

    #Target(High Income Women)
    hiw_total, hiw_correct, hiw_recall = find_recall(mask_highinc_women,y_proba_adj)
    hiw_pred_pos = sum((y_proba_adj==1)&(s_test == 0))
    hiw_precision = hiw_correct/hiw_pred_pos if hiw_pred_pos >0 else 0
    hiw_f1 = 2*(hiw_precision*hiw_recall)/(hiw_recall + hiw_precision) if (hiw_recall + hiw_precision)>0 else 0
    #Comparison (High Income Men)
    him_total, him_correct, him_recall = find_recall(mask_highinc_men,y_proba_adj)
    #Low income women 
    liw_total, liw_correct, liw_specific = find_recall(mask_lowinc_women,y_proba_adj)
    #Low income men 
    lim_total, lim_correct, lim_specific = find_recall(mask_lowinc_men,y_proba_adj)

    #Overall Accuracy 
    accuracy = sum(y_proba_adj == y_test)/len(y_test)

    return {
        'women_threshold': womenthreshold,
        'hiw_recall': hiw_recall,
        'hiw_precision': hiw_precision,
        'hiw_f1': hiw_f1,
        'hiw_total': hiw_total,
        'hiw_correct': hiw_correct,
        'him_recall': him_recall,
        'liw_specificity': liw_specific,
        'lim_specificity': lim_specific,
        'overall_accuracy': accuracy
    }