import numpy as np
import pandas as pd
import math
import os
from shutil import rmtree
import sys
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn import metrics
from sklearn.decomposition import PCA
from lime import lime_tabular
from lime import submodular_pick
from matplotlib import pyplot as plt
from math import sqrt
from datetime import datetime as dt

from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

plt.style.use("seaborn")
pd.set_option('display.max_columns', 300)  # or 1000
pd.set_option('display.max_rows', 10)  # or 1000
pd.set_option('display.max_colwidth', 300)  # or 199

'''
###################################################################
LOAD EVENT LOG FILE
###################################################################
'''


def expSettings(expName):
    if os.path.exists('results/' + expName):
        rmtree('results/' + expName)
    os.mkdir('results/' + expName)
    msg = 'Experiment name: %s' % expName
    print(msg)
    return msg


# Load event log data -- ok
def loadEventLog(ds_name, column_caseId="", column_activity="", column_label="", sep=","):
    ds = pd.read_csv('data/' + ds_name.split('_')[1] + '.csv', sep=sep)
    if column_caseId != "" and column_activity != "" and column_label != "":
        ds = ds.rename(columns={column_caseId: 'caseId', column_activity: 'activity', column_label: 'label'})
    else:
        ds = ds.rename(columns={0: 'caseId', 1: 'activity'})
        ds['label'] = '-'
    msg = '====> Log data loaded... \n DS shape: %s' % str(ds.shape)
    print(msg)
    return ds, msg


# Load trace enconding matrix -- ok
def loadTraceEncodingMatrix(ds_name, sep=","):
    ds = pd.read_csv('data/' + ds_name + '_traceEncodMatrix.csv', sep=sep)
    msg = '====> Traces dataset loaded... \n DS shape: %s' % str(ds.shape)
    print(msg)
    return ds, msg


'''
###################################################################
DESCRIBE THE LOG
###################################################################
'''


# Check loops -- ok
def checkLoops(ds):
    data = ds.loc[:, ['caseId', 'activity']]
    actLoop = data.groupby(data.columns.tolist()).size().reset_index().rename(columns={0: 'loop'})
    print('Loop max', actLoop['loop'].max())
    print(actLoop[actLoop['loop'] > 1])
    return actLoop


# Check if any activity is correlated with the label -- ok
def checkCorrelation(encodMatrix, activity, ds):
    ds[activity] = ds[activity].astype(bool)
    print('is Activity == label?', ds[activity].equals(ds['label']))


# # Find a list of activities in a trace
# # Parameters: trace (lista)
# # Returns: traceSemLoop (lista)
# def findActivityInTraces(traceList, actFinded):
#     ans = False
#     traceList = [ans := True for v in traceList if v in actFinded]
#     return ans


'''
###################################################################
# PRE-PROCESSING LOG
###################################################################
'''


# # Delete out loops
# # Parameters: trace (lista)
# # Returns: traceSemLoop (lista)
# def dropLoopsInTraces(traceList):
#     prev = object()
#     traceList = [prev := v for v in traceList if prev != v]
#     return traceList

# # Transform the log into dataset of traces
# def f_Traces(x):
#     return pd.Series(dict(trace='%s' % ','.join(x['activity']),
#                           nrEvents=x['activity'].count(),
#                           target=int(x[target].mean())))
#
#
# # simple indexing, considering trace positions
# # Input: dataset of traces (dataframe)
# # Output: dataset of activity per position in trace (dataframe)
# def simpleIndexingNoLoops(ds):
#     ds = ds.groupby('caseId').apply(f_Traces)
#     ds_new = pd.DataFrame(columns=['caseId', 'trace', 'nrEvents', target])
#     for item in ds.iterrows():
#         noduplicates = dropLoopsInTraces(item[1][0].split(','))
#         ds_new.loc[len(ds_new)] = [int(item[0]), ','.join(noduplicates), len(noduplicates), int(item[1][2])]
#
#     ds_new[['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13']] = ds_new['trace'].str.split(',', expand=True)
#     ds_new = ds_new.drop(['caseId', 'trace', 'nrEvents'], axis=1)
#     ds_new.to_csv('results/' + expName + '_traces.csv')
#     print("Data preprocessed... SIMPLE-INDEXING ENCODING")
#     return ds_new


# frequency indexing, considering frequency of occurrence of an activity in the trace -- ok
def frequencyIndexingWithLoops(ds):
    ds['val'] = 1
    ds = ds.groupby(['caseId', 'label', 'activity'])['val'].sum().unstack(fill_value=0)
    ds.reset_index(level='label', inplace=True)
    ds.reset_index(level='caseId', inplace=True)
    ds.drop(['caseId'], axis=1, inplace=True)
    ds.to_csv('results/%s/%s_traceEncodMatrix.csv' % (expName, expName))
    msg = '====> Data encoding with: Frequency-indexing \n DS shape: %s (traces)' % str(ds.shape)
    print(msg)
    return ds, msg


'''
###################################################################
ENCODING DATASET FOR ML TRAINEE
###################################################################
'''


# Apply one-hot encoding and split data -- ok
def encodingAndSplitData(ds, splitType, expName):
    #####----> mejorar con https://www3.cs.stonybrook.edu/~cse634/lecture_notes/07testing.pdf
    X = pd.get_dummies(ds.drop(columns='label'))
    y = ds['label']
    # y = pd.get_dummies(ds['label'])

    if splitType == '80_20':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Set the random_state parameter to 42 for getting the same split
    elif splitType == 'Resubstitution':
        X_train = X
        X_test = X
        y_train = y
        y_test = y

    X_train.to_csv('results/%s/%s_XTrain.csv' % (expName, expName))
    X_test.to_csv('results/%s/%s_XTest.csv' % (expName, expName))
    y_train.to_csv('results/%s/%s_yTrain.csv' % (expName, expName))
    y_test.to_csv('results/%s/%s_yTest.csv' % (expName, expName))
    msg = '====> Data splitted... \n splitType: %s \n X_train: %s \n y_train: %s \n X_test: %s \n y_test: %s' % (splitType, X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    print(msg)
    return X_train, y_train, X_test, y_test, msg


'''
###################################################################
ML TREINEE
###################################################################
'''


def chooseParameters(X_train, y_train, X_test, y_test, dict_weights):
    # Busca em Grid (grade) de hiperparâmetros avaliados
    param_grid = ParameterGrid(
        {'n_estimators': [50, 100, 150],  # 'max_features': [5, 7, 9],
         'max_depth': [None, 3, 10, 20],
         'criterion': ['gini', 'entropy']
         }
    )
    # Loop para ajustar um modelo com cada combinação de hiperparâmetros
    resultados = {'params': [], 'oob_accuracy': [], 'accuracy': []}

    for params in param_grid:
        modelo = RandomForestClassifier(
            oob_score=True,  # True, #n_jobs=-1,
            random_state=42,
            class_weight=dict_weights,
            **params
        )
        modelo.fit(X_train, y_train)
        resultados['params'].append(params)
        resultados['oob_accuracy'].append(modelo.oob_score_)
        resultados['accuracy'].append(modelo.score(X_train, y_train))
        print(f"Modelo: {params} \u2713")

    # Resultados
    resultados = pd.DataFrame(resultados)
    resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
    resultados = resultados.sort_values('oob_accuracy', ascending=False)
    resultados = resultados.drop(columns='params')
    resultados.head(10)
    msg = '====> Choose better model... \n ' + resultados.head(10).to_string()
    return resultados.head(1), msg


# Train um RandomForestClassifier from ScikitLearn -- ok
def trainRFModel(X_train, y_train, X_test, y_test):
    X_train.to_csv('results/%s/%s_XTrain.csv' % (expName, expName))
    X_test.to_csv('results/%s/%s_XTest.csv' % (expName, expName))
    y_train.to_csv('results/%s/%s_yTrain.csv' % (expName, expName))
    y_test.to_csv('results/%s/%s_yTest.csv' % (expName, expName))

    #class_names = ['0', '1', '2']
    class_names = ['0', '1']

    # escolher parametros
    # = {0: 0.77, 1: 0.91, 2: 1.62}
    dict_weights = {0: 0.5, 1: 0.5}
    bestParameter, msg = chooseParameters(X_train, y_train, X_test, y_test, dict_weights)
    msg = msg + '\n ====> Final parameters... \n ' + bestParameter.to_string() + '\n oob_score = True, random_state = 42, class_weight=' + str(dict_weights)

    # treinar o modelo
    model = RandomForestClassifier(n_estimators=int(bestParameter['n_estimators']), criterion=bestParameter['criterion'].iloc[0], oob_score=True, random_state=42, class_weight=dict_weights)
    model.fit(X_train, y_train)
    msg = msg + '\n ====> Model trained...'

    # avaliar o modelo
    model_score = model.score(X_test, y_test) # Return the mean accuracy on the given test data and label
    model_obbScore = model.oob_score_ # Score of the training dataset obtained using an out-of-bag estimate. This attribute exists only when oob_score is True.
    msg = msg + '\n model_score: %s \n model_obbScore:  %s' % (model_score, model_obbScore)

    # Metrics_accuracy
    y_pred = model.predict(X_test)
    pd.DataFrame(y_pred).to_csv('results/%s/%s_yPred.csv' % (expName, expName))

    acc = metrics.accuracy_score(y_test, y_pred)
    matrix = metrics.confusion_matrix(y_test, y_pred)
    report = metrics.classification_report(y_test, y_pred)
    matrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    mcm = metrics.multilabel_confusion_matrix(y_test, y_pred)
    f1score = metrics.f1_score(y_test, y_pred, average='weighted')
    epsilon = 1 - f1score
    # matrix_display.plot()
    # plt.show()

    msg = msg + ' -> Cross tab \n' + str(pd.crosstab(y_train, y_pred, rownames=['Actual labels'], colnames=['Predicted labels']))
    msg = msg + '\n -> Metrics SKlearn \n accuracy:  %s \n matrix: %s \n report: %s \n multilabel matrix %s\n f1score_weighted %s  \n epsilon %s' % (acc, matrix, report, mcm, f1score, epsilon)

    # x_y_ypred = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1, ignore_index=False)
    # x_y_ypred = pd.concat([pd.DataFrame(x_y_ypred), pd.DataFrame(y_pred)], axis=1, ignore_index=False)
    # x_y_ypred.to_csv('results/%s/%s_x_y_ypred.csv' % (expName, expName))
    # xy_test = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test, columns=['y_test']), pd.DataFrame(y_pred, columns=['y_pred'])], axis=1)
    # xy_test.to_csv('results/%s/%s_XyTest.csv' % (expName, expName))

    X_test[(y_test == 0) & (y_pred == y_test)].to_csv('results/%s/%s_XTest_predOK_class0.csv' % (expName, expName))
    X_test[(y_test == 1) & (y_pred == y_test)].to_csv('results/%s/%s_XTest_predOK_class1.csv' % (expName, expName))
    X_test[(y_test == 2) & (y_pred == y_test)].to_csv('results/%s/%s_XTest_predOK_class2.csv' % (expName, expName))
    print(msg)
    return model, msg


'''
###################################################################
INTERPRETABLE MODELS
###################################################################
'''

#
# # Apply Lime for an instance, Input: CaseId (int), Output: LimeExp (lime object)
# def applyLimeIn(points, nrFeatures):
#     explainer = lime_tabular.LimeTabularExplainer(
#         training_data=np.array(X_train),
#         feature_names=X_train.columns,
#         class_names=class_names,
#         mode='classification')
#
#     exp_points = []
#     exp_list = []
#
#     for idx in points:
#         exp = explainer.explain_instance(data_row=X_train.loc[idx], predict_fn=model.predict_proba, num_features=nrFeatures)
#         exp_list = exp.as_list()
#         exp_list.append(round(model.predict_proba([X_train.loc[idx]])[0, 0], 2))
#         exp_list.append(round(model.predict_proba([X_train.loc[idx]])[0, 1], 2))
#         exp_list.append(ds.loc[idx, target])  # exp_list.append(ds_new.loc[idx, target])
#         exp_list.append(idx)
#         exp_points.append(exp_list)
#     fileName = "results/" + expName + "_explanations.csv"
#     with open(fileName, "w") as file:
#         for row in exp_points:
#             file.write("%s\n" % ';'.join(str(col) for col in row))
#     print("Explanations file created... " + fileName)

#
# # Apply SP-Lime
# def applySPLimeIn(sampleSize, nrFeatures, nrExplanations):
#     explainer = lime_tabular.LimeTabularExplainer(
#         training_data=np.array(X_train),
#         feature_names=X_train.columns,
#         class_names=class_names,
#         mode='classification')
#
#     training_data = np.array(X_train)
#     sp_obj = submodular_pick.SubmodularPick(explainer, data=training_data, predict_fn=model.predict_proba, sample_size=sampleSize, num_features=nrFeatures, num_exps_desired=nrExplanations)
#     exp_points = sp_obj.sp_explanations[9].as_list()
#     fileName = "results/" + expName + "_explSPLIME.csv"
#     with open(fileName, "w") as file:
#         for row in exp_points:
#             file.write("%s\n" % ';'.join(str(col) for col in row))
#     print("Explanations with SP-LIME file created... " + fileName)
#

'''
###################################################################
PLOTTING RESULTS
###################################################################
'''


# # Plot results from file generated by the LIME module, Input: Number of figures per row (int), Experiment Name (str)
# def plotLimeResults(plotsPerRow, expName):
#     i, j = 0, 0
#     fileName = "results/" + expName + "_explLime.csv"
#     with open(fileName, "r") as file:
#         exp_points = list(file)
#     exp_points = [x.rstrip() for x in exp_points]
#     exp_points = [list(x.split(';')) for x in exp_points]
#
#     plt.rcParams['xtick.labelsize'] = 16
#     plt.rcParams['ytick.labelsize'] = 16
#     # Graph multiplot
#     fig, axs = plt.subplots(nrows=math.ceil(len(exp_points) / plotsPerRow), ncols=plotsPerRow, constrained_layout=True, figsize=(20, 20))  # #squeeze=False, you can force the result to be a 2D-array, independant of the number or arrangement of the subplots
#     fig.suptitle('Experiment: %s' % expName, fontsize=20)  # title for entire figure
#
#     exp_list = []
#     for exp_list in exp_points:
#         exp_list = [x.strip('(') for x in exp_list]
#         exp_list = [x.strip(')') for x in exp_list]
#         exp_list = [x.split(', ') for x in exp_list]
#         names = [x[0].strip("\'") for x in exp_list[:-4]]  # Y
#         names = [n.rpartition('.0')[0] for n in names]
#         vals = [round(float(x[1]), 3) for x in exp_list[:-4]]  # X
#         pos = np.arange(len(exp_list) - 4) + .5
#         prob0 = exp_list[len(exp_list) - 4][0]
#         prob1 = exp_list[len(exp_list) - 3][0]
#         y_target = exp_list[len(exp_list) - 2][0]
#         idx = exp_list[len(exp_list) - 1][0]
#         vals.reverse()
#         names.reverse()
#         colors = ['green' if x > 0 else 'red' for x in vals]
#         axs[i][j].set_title('Case Id: %s\nProbability for class 0 = %s \nProbability for class 1 = %s\n Target/Y: %s' % (idx,
#                                                                                                                          str(prob0),
#                                                                                                                          str(prob1),
#                                                                                                                          y_target))
#         axs[i][j].barh(pos, vals, align='center', color=colors)
#         axs[i][j].set_yticks(pos, names)
#         j += 1
#         if j % plotsPerRow == 0:
#             i += 1
#             j = 0
#
#     plt.show()
#     fig.savefig(str("results/" + expName + "_plotLime.jpg"), bbox_inches='tight')
#     print('Figure saved...' + "results/" + expName + "_plotLime.jpg")

def formatColumns(listColumnsName):
    names = []
    for name in listColumnsName:
        if '.00' in name:
            name = name.replace('.00', '')
        if '0 <' in name:
            name = name.replace('0 <', '')
        if '> 0' in name:
            name = name.replace('> 0', '= 1')
        if '<' in name:
            name = name.replace('<', '')
        if ' ' in name:
            name = name.replace(' ', '')
        names.append(name)
    return names


# plot LIME-SP results in a a global way --ok
def plotLimeSPResults(expName, className, sp_obj):
    # Plot all explanations
    # [explainer.as_pyplot_figure(label=explainer.available_labels()[0]) for explainer in sp_obj.sp_explanations];
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 15

    i = 0
    for explainer in sp_obj.sp_explanations:
        fig = explainer.as_pyplot_figure(label=explainer.available_labels()[0])
        fig.savefig('results/%s/%s_SPLime_Class%s_Fig%s.jpg' % (expName, expName, className, str(i)), bbox_inches='tight')
        i += 1

    # Make it into a dataframe SP-LIME
    W_pick = pd.DataFrame([dict(this.as_list(this.available_labels()[0])) for this in sp_obj.sp_explanations]) #.fillna(0)

    # Getting SP predictions
    W_pick['label'] = [this.available_labels()[0] for this in sp_obj.sp_explanations]
    W_pick.columns = formatColumns(W_pick.columns)
    W_pick.sort_index(axis=1, inplace=True)
    #W_pick.to_csv('results/%s/%s_pickWmatrixSPLime_Class%s.csv' % (expName, expName, className))

    # Making a dataframe of all the explanations of sampled points SIMPLE - LIME
    W = pd.DataFrame([dict(this.as_list(this.available_labels()[0])) for this in sp_obj.explanations]) #.fillna(0)
    W['prediction'] = [this.available_labels()[0] for this in sp_obj.explanations]
    W.columns = formatColumns(W.columns)
    W.sort_index(axis=1, inplace=True)
    W.to_csv('results/%s/%s_fullWmatrixSPLime_Class%s.csv' % (expName, expName, className))
    print('====> SPLime results saved...')
    return W_pick


def preProcessLog(ds, expName):
    msg = '------------------------------- \n ====> Pre Processing steps\n'
    ds['activity'] = ds['activity'].replace({'W_Afhandelen leads': 'W_Fixing_incoming_lead',
                                             'W_Completeren aanvraag': 'W_Filling_info_application',
                                             'W_Valideren aanvraag': 'W_Assessing_application',
                                             'W_Nabellen offertes': 'W_Calling_after_sent_offers',
                                             'W_Nabellen incomplete dossiers': 'W_Calling_to_add_info_app',
                                             'W_Beoordelen fraude': 'W_Assess_fraude',
                                             'W_Wijzigen contractgegevens': 'W_Change_contact_details',
                                             'A_ACCEPTED': 'A_Accepted',
                                             'A_ACTIVATED': 'A_Activated',
                                             'A_FINALIZED': 'A_Finalized',
                                             'A_PREACCEPTED': 'A_Preaccepted',
                                             'A_PARTLYSUBMITTED': 'A_Partly_submetted',
                                             'A_SUBMITTED': 'A_Submitted',
                                             'O_CANCELLED': 'O_Cancelled',
                                             'O_ACCEPTED': 'O_Accepted',
                                             'O_CREATED': 'O_Created',
                                             'O_DECLINED': 'O_Declined',
                                             'A_REGISTERED': 'A_Registered',
                                             'O_SELECTED': 'O_Selected',
                                             'O_SENT': 'O_Sent',
                                             'O_SENT_BACK': 'O_Sent_back'})
    msg = msg + '-> activities renamed\n'

    # rotular traces
    caseIdDeclined = ds[(ds['activity'] == 'A_DECLINED')]['caseId']
    caseIdApproveded = ds[(ds['activity'] == 'A_APPROVED')]['caseId']
    caseIdCancelled = ds[(ds['activity'] == 'A_CANCELLED')]['caseId']
    ds['label'] = '-'
    msg = msg + '-> classes were defined\n'

    for item in caseIdDeclined:
        ds.loc[ds.caseId == item, 'label'] = 0  # 27
    msg = msg + 'Class 0:' + str(ds.loc[ds.label == 0].shape[0])+'\n'
    for item in caseIdApproveded:
        ds.loc[ds.caseId == item, 'label'] = 1  # 23
    msg = msg + 'Class 1:' + str(ds.loc[ds.label == 1].shape[0])+'\n'
    for item in caseIdCancelled:
        ds.loc[ds.caseId == item, 'label'] = 2  # 13
    msg = msg + 'Class 2:' + str(ds.loc[ds.label == 2].shape[0])+'\n'

    # drop incomplete traces
    ds.drop(ds[ds.label == '-'].index, inplace=True)
    msg = msg + '-> Incompleted traces were dropped \n DS shape %s (events)\n' % str(ds.shape)

    # encoding
    ds, msg2 = frequencyIndexingWithLoops(ds)
    msg = msg + '-> Frequency Indexing applied\n' + msg2

    # reduced frequencies
    ds[ds.iloc[:, 1:] > 0] = 1
    msg = msg + '-> Frequencies reduced to 1'+'\n'

    # drop duplicates
    ds.drop_duplicates(keep="first", inplace=True)
    msg = msg + '-> Duplicates were dropped \n DS shape %s unique traces\n' % str(ds.shape)

    # drop correlated activities
    ds.drop(['A_APPROVED'], axis=1, inplace=True)
    ds.drop(['A_DECLINED'], axis=1, inplace=True)
    ds.drop(['A_CANCELLED'], axis=1, inplace=True)
    msg = msg + '-> correlated activities were dropped: \nA_APPROVED \nA_DECLINED \nA_CANCELLED \n'

    msg = msg + '-> sequence activities were dropped:'
    # drop sequency activities
    # ds.drop(['A_PARTLYSUBMITTED'], axis=1, inplace=True)
    # ds.drop(['A_SUBMITTED'], axis=1, inplace=True)
    for col in ds.columns:
        if ds[col].is_monotonic:
            msg = msg + '\n' + col
            del ds[col]

    msg = msg + '\n -> ds columns: ' + str(ds.columns) + '\n Nro. Columns: ' + str(ds.columns.shape)
    ds.to_csv('results/%s/%s_preprocessedDS.csv' % (expName, expName))
    return ds, msg

'''
###################################################################
MAIN
###################################################################
'''

if __name__ == '__main__':
    expName = 'expCOOPIS_BPIC12_v2'
    class_names = [0, 1, 2]
    msgSettings = ['========== EXPERIMENT SETUP ==========', 'Date: ' + str(dt.now())]
    #
    msg = expSettings(expName)

    ds, mgs = loadEventLog(ds_name=expName, column_caseId='Case ID', column_activity='Activity', column_label='label', sep=',')
    msgSettings.append(msg)

    ds, msg = preProcessLog(ds, expName)
    msgSettings.append(msg)

    # SPLIT DATASET
    X_train, y_train, X_test, y_test, msg = encodingAndSplitData(ds=ds, splitType='Resubstitution', expName=expName)
    msgSettings.append(msg)

    # X_train_full = pd.read_csv('results/Article 2/exp26_BPIC12_testbies_4_XTrain.csv', sep=',', index_col=[0])  # .values
    # y_train_full = pd.read_csv('results/Article 2/exp26_BPIC12_testbies_4_yTrain.csv', sep=',')['label']
    # model, msg = trainRFModel(X_train_full, y_train_full, X_train_full, y_train_full)
    model, msg = trainRFModel(X_train, y_train, X_test, y_test)
    msgSettings.append(msg)

    msg = '--------------------\n ====> Lime....\n'
    expNameClass0 = expName + 'class_0'
    msgSettings.append(msg)

    X_train = pd.read_csv('results/%s/%s_XTest_predOK_class0.csv' % (expName, expName), sep=',').iloc[:,1:]
    msgSettings.append('clase 0')

    # SP-LIME
    start = dt.now()
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns,
        class_names=class_names,
        mode='classification')

    sp_obj = submodular_pick.SubmodularPick(data=np.array(X_train), explainer=explainer, num_exps_desired=5, predict_fn=model.predict_proba, sample_size=20, num_features=10, top_labels=3)
    msg = "Elapsed time: %s" % timedelta(seconds=round((dt.now() - start).seconds))
    msgSettings.append(msg)
    print(msg)
    W0 = plotLimeSPResults(expName, str('CANCELED_0'), sp_obj)
    W0.sort_index(axis=1, inplace=True)
    W0.to_csv('results/%s/%s_Wpick_CLASE0.csv' % (expName, expName))

    ################
    X_train = pd.read_csv('results/%s/%s_XTest_predOK_class1.csv' % (expName, expName), sep=',').iloc[:,1:]
    msgSettings.append('clase 1')

    # SP-LIME
    start = dt.now()
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns,
        class_names=class_names,
        mode='classification')

    sp_obj = submodular_pick.SubmodularPick(data=np.array(X_train), explainer=explainer, num_exps_desired=5, predict_fn=model.predict_proba, sample_size=20, num_features=10, top_labels=3)
    msg = "Elapsed time: %s" % timedelta(seconds=round((dt.now() - start).seconds))
    msgSettings.append(msg)
    print(msg)
    W1 = plotLimeSPResults(expName, str('APPROVED_1'), sp_obj)
    W1.sort_index(axis=1, inplace=True)
    W1.to_csv('results/%s/%s_Wpick_CLASE1.csv' % (expName, expName))

    ################
    X_train = pd.read_csv('results/%s/%s_XTest_predOK_class2.csv' % (expName, expName), sep=',').iloc[:,1:]
    msgSettings.append('clase 2')

    # SP-LIME
    start = dt.now()
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns,
        class_names=class_names,
        mode='classification')

    sp_obj = submodular_pick.SubmodularPick(data=np.array(X_train), explainer=explainer, num_exps_desired=5, predict_fn=model.predict_proba, sample_size=20, num_features=10, top_labels=3)
    msg = "Elapsed time: %s" % timedelta(seconds=round((dt.now() - start).seconds))
    msgSettings.append(msg)
    print(msg)
    W2 = plotLimeSPResults(expName, str('DECLINED_2'), sp_obj)
    W2.sort_index(axis=1, inplace=True)
    W2.to_csv('results/%s/%s_Wpick_CLASE2.csv' % (expName, expName))

    Wfull = pd.DataFrame()
    Wfull = Wfull.append(W0, ignore_index=True, sort=True)
    Wfull = Wfull.append(W1, ignore_index=True, sort=True)
    Wfull = Wfull.append(W2, ignore_index=True, sort=True)
    Wfull.rename(columns={'label': 'zlabel'}, inplace=True)
    Wfull.sort_index(axis=1, inplace=True)
    Wfull.to_csv('results/%s/%s_Wpick_Completo012.csv' % (expName, expName))

    with open('results/%s/%s_setup.txt' % (expName, expName), "w") as file:
        file.write('\n'.join(msgSettings))
    file.close()


