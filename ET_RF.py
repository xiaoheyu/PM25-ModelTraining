import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
import scikitplot as skplt
import os
import pickle
import glob
# import pyart
import xarray as xr
# import seaborn as sns
import math
from sklearn.model_selection import RandomizedSearchCV
import shap
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.metrics import mean_squared_error
from math import sqrt
import sys
import shap
from sklearn.tree import export_graphviz
import random
from subprocess import call
import joblib
import pydot
from skopt import BayesSearchCV
from sklearn.preprocessing import MinMaxScaler



def model_scatterplot(predictionFile,outname):
        rmse = mean_squared_error(predictionFile['Value'], predictionFile['Predictions'],squared=False)
        R = predictionFile['Value'].corr(predictionFile['Predictions'])
        print(outname)
        print("rmse = %f" %(rmse))
        print('Correlation Coefficient %f \n' %R)

        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.01

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing+0.02, width, 0.2]
        rect_histy = [left + width + spacing+0.03, bottom, 0.2, height]

        plt.figure(figsize=(10, 10))
        ax_scatter = plt.axes(rect_scatter)

        ax_histx = plt.axes(rect_histx)
        ax_histx.tick_params(direction='in', labeltop=False,labelsize = 20)
        ax_histy = plt.axes(rect_histy)
        ax_histy.tick_params(direction='in', labelleft=False,labelright=True,labelsize = 20)

        # fig, ax_scatter = plt.subplots(figsize=[15, 10])

        smoothed = lowess(predictionFile['Predictions'],predictionFile['Value'])
        ax_scatter.scatter(predictionFile['Predictions'], predictionFile['Value'], s=5, edgecolors = 'k', facecolors = 'none')
        # ax_scatter.plot(smoothed[:,0],smoothed[:,1],color = 'r')
        ax_scatter.set_ylabel('Observed Values', fontsize=20)
        xticks = np.arange(0, 150, 15)
        yticks = np.arange(0, 150, 15)

#         xticks = np.linspace(0, np.abs([predictionFile['Predictions'],predictionFile['Value']]).max(), 10)
#         yticks = np.linspace(0, np.abs([predictionFile['Predictions'],predictionFile['Value']]).max(), 10)

#         ax_scatter.set_ylim(ax_scatter.get_ylim())
        ax_scatter.set_ylim(0,150)
        ax_scatter.set_xlabel('Fitted Values',fontsize=20)
#         ax_scatter.set_xlim(ax_scatter.get_xlim())
        ax_scatter.set_xlim(0,150)
        ax_histx.set_title('Observed vs. Fitted n=%d RMSE=%.3f R=%.3f' %(len(predictionFile),rmse,R),fontsize=30)
        ax_scatter.tick_params(labelsize = 20)
        # ax_scatter.plot([min(predictions_train_valid['Value']),max(predictions_train_valid['Value'])],[0,0],color = 'k',linestyle = ':', alpha = .3)
        ax_scatter.set_xticks(xticks)
        ax_scatter.set_yticks(yticks)


        binwidth = 1
        lim = np.ceil(np.abs([predictionFile['Predictions'],predictionFile['Value']]).max() / binwidth) * binwidth
        bins = np.arange(-lim, lim + binwidth, binwidth)

        ax_histy.hist(predictionFile['Value'], bins=bins, orientation='horizontal')
        ax_histy.set_ylim(ax_scatter.get_ylim())

        ax_histx.hist(predictionFile['Predictions'], bins=bins, orientation='vertical')
        ax_histx.set_xlim(ax_scatter.get_xlim())

        plt.savefig(os.path.join(Outdir,outname + '_Observed_vs_Fitted.jpg'), bbox_inches='tight',dpi=300)
        

def residual_plot(predictionFile,outname):
        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.01

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing+0.02, width, 0.2]
        rect_histy = [left + width + spacing, bottom, 0.2, height]

        plt.figure(figsize=(10, 10))
        ax_scatter = plt.axes(rect_scatter)

        ax_histx = plt.axes(rect_histx)
        ax_histx.tick_params(direction='in', labeltop=False,labelsize = 20)
        ax_histy = plt.axes(rect_histy)
        ax_histy.tick_params(direction='in', labelleft=False,labelright=True,labelsize = 20)

        # fig, ax_scatter = plt.subplots(figsize=[15, 10])

#         smoothed = lowess(predictionFile['Predictions'],predictionFile['Value'])
        ax_scatter.scatter(predictionFile['Predictions'], predictionFile['Residuals'], s=5, edgecolors = 'k', facecolors = 'none')
        # ax_scatter.plot(smoothed[:,0],smoothed[:,1],color = 'r')
        ax_scatter.set_ylabel('Residuals', fontsize=20)
        ax_scatter.set_xlabel('Fitted Values',fontsize=20)
        ax_histx.set_title('Residuals vs Fitted n=%d' %(len(predictionFile)),fontsize=30)
        ax_scatter.tick_params(labelsize = 20)
        # ax_scatter.plot([min(predictions_train_valid['Value']),max(predictions_train_valid['Value'])],[0,0],color = 'k',linestyle = ':', alpha = .3)



        binwidth = 1
        lim = np.ceil(np.abs([predictionFile['Predictions'],predictionFile['Value']]).max() / binwidth) * binwidth
        bins = np.arange(-lim, lim + binwidth, binwidth)

        ax_histy.hist(predictionFile['Residuals'], bins=bins, orientation='horizontal')
        ax_histy.set_ylim(ax_scatter.get_ylim())

        ax_histx.hist(predictionFile['Predictions'], bins=bins, orientation='vertical')
        ax_histx.set_xlim(ax_scatter.get_xlim())
        plt.savefig(os.path.join(Outdir,outname +'_Residual.jpg'), bbox_inches='tight', dpi=300)
        
        
def make_prediction(modelName, trainset, testset,train,test):
        predictions_train = pd.DataFrame(modelName.predict(trainset),columns=["Predictions"])
        predictions_test = pd.DataFrame(modelName.predict(testset),columns=["Predictions"])

        predictions_train_valid = pd.concat([train["Value"].reset_index(drop=True), predictions_train.reset_index(drop=True)],axis=1)
        predictions_test_valid = pd.concat([test["Value"].reset_index(drop=True), predictions_test.reset_index(drop=True)],axis=1)

        predictions_train_valid['Residuals'] = predictions_train_valid['Value']  - predictions_train_valid['Predictions']
        predictions_test_valid['Residuals'] = predictions_test_valid['Value']  - predictions_test_valid['Predictions']
        return predictions_train_valid, predictions_test_valid

def plot_nadistr(model):
    is_NaN = df_model.isnull()
    row_has_NaN = is_NaN.any(axis=1)
    rows_with_NaN = df_model[row_has_NaN]
    fig = plt.figure(figsize=(10,8))
    plt.ion()
#     plt.show()
    plt.hist(rows_with_NaN["Value"], bins=range(0, 60, 1))
    plt.close(fig)
    fig.savefig(os.path.join(Outdir,"distribution_na.png"), bbox_inches='tight', dpi=300)
    
    
    
def plot_distr(df_model, var_name="Value"):
    fig = plt.figure()
    plt.ion()
#     plt.show()
    plt.hist(df_model[var_name], bins=range(0, 60, 1))
    plt.close(fig)
    fig.savefig(os.path.join(Outdir,"distribution.png"), bbox_inches='tight', dpi=300)
    
    

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy


def importance_plot(model,features,outname):
    important_features_dict = {}
    for idx, val in enumerate(model.feature_importances_):
        important_features_dict[idx] = val

    important_features_list = sorted(important_features_dict,
                                     key=important_features_dict.get,
                                     reverse=False)
    rankfeature =[]
    rankvalue = []
    for i in important_features_list:
        rankvalue.append(model.feature_importances_[i])
        rankfeature.append(features[i])

    fig = plt.figure(figsize=(12,8))
    plt.ion()
#     plt.show()
    barlist= plt.barh(rankfeature, rankvalue)
    for i in range(3):
        barlist[-(i+1)].set_color('r')
    for i in range(3,6):
        barlist[-(i+1)].set_color('orange')
    plt.close(fig)
    fig.savefig(os.path.join(Outdir,outname+".png"), bbox_inches='tight', dpi=300)

        
        
def abs_shap(df_shap,data_x,outname):
    #import matplotlib as plt
    # Make a copy of the input data
    shap_v = pd.DataFrame(df_shap)
    feature_list = data_x.columns
    shap_v.columns = feature_list
    df_v = data_x.copy().reset_index().drop('index',axis=1)
    
    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i],df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list),pd.Series(corr_list)],axis=1).fillna(0)
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns  = ['Variable','Corr']
    corr_df['Sign'] = np.where(corr_df['Corr']>0,'red','blue')
    
    # Plot it
    shap_abs = np.abs(shap_v)
    k=pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable','SHAP_abs']
    k2 = k.merge(corr_df,left_on = 'Variable',right_on='Variable',how='inner')
    k2 = k2.sort_values(by='SHAP_abs',ascending = True)
    colorlist = k2['Sign']
    fig = plt.figure()
    plt.ion()
    k2.plot.barh(x='Variable',y='SHAP_abs',color = colorlist, figsize=(5,6),legend=False)
#     ax.set_xlabel("SHAP Value (Red = Positive Impact)")
    plt.close(fig)
    fig.savefig(os.path.join(Outdir,outname+".png"), bbox_inches='tight', dpi=300)
    
    
    
def shap_plot(model,data_x,outname):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data_x)
    shap.summary_plot(shap_values, data_x, plot_type="bar")
    fig = plt.figure()
    plt.ion()
#     plt.show()
    shap.summary_plot(shap_values, data_x)
    plt.close(fig)
    fig.savefig(os.path.join(Outdir,outname+".png"), bbox_inches='tight', dpi=300)

    return shap_values



def show_tree(name,model,predictors,train_Y,estimatorindex,Outdir):

    estimator = model.estimators_[estimatorindex]
    dotfn = os.path.join(Outdir,name + str(estimatorindex)+"tree.dot")
    pngfn=os.path.join(Outdir,name + str(estimatorindex)+"tree.png")
    # Export as dot file
    fig = plt.figure()
    plt.ion()
    export_graphviz(estimator, out_file=dotfn, 
                    feature_names = predictors,
                    class_names = train_Y,
                    rounded = True, proportion = False, 
                    precision = 2, filled = True)
    # Convert to png using system command (requires Graphviz)
#     call(['dot', '-Tpng',dotfn , '-o', pngfn, '-Gdpi=400'])
    (graph,) = pydot.graph_from_dot_file(dotfn)
    graph.write_png(pngfn)
    plt.close(fig)

########################################################################################################
## Reading File
#-----------------------------------------------------------------------------------
Outdir = "CA_all_1deg"
if not os.path.exists(Outdir):
    os.makedirs(Outdir)
Filefolder = "flowMatched"
# Filename = "AQS_Radar2ele01_ECMWFLANDFIX_Pop001_Cover_Soil_Glim_Lith4_GEBCO_16_20_fix.csv"
# Filename = "testinghead1000Z01.csv"
# Filename = "AQS_Radar2ele01_ECMWFLANDFIX_Pop001_Cover_Soil_Glim_Lith4_GEBCO_GOES2_Z01_SA_17_20_fix.csv"
Filename = "AQS_CA_Radar2ele01_ECMWFLANDFIX_Pop001_Cover_Soil_Glim_Lith4_GEBCO_GOES2_Z01_17_20_BL_SA_fix_dropna_grid1deg.csv"


# Filename ="/home/xiaohe/Documents/NEXRAD/Model/Prediction/Mdl_10000.csv"
#BayesSearchSV parameters setting
n_iter = 20
n_jobs = 6
n_points = 2
cv = 3
# Use the random grid to search for best hyperparameters
reg = ExtraTreesRegressor(random_state=1)
#-----------------------------------------------------------------------------------

print("Reading file")
# AirNow_Radar_ECMWF_df = pd.read_csv(os.path.join(Filefolder,Filename))
AirNow_Radar_ECMWF_df = pd.read_csv(Filename)

# sys.stdout = open(os.path.join(Outdir,"report.csv"), "w")
print(Outdir)
## Variables definition
columns_useful = ['Latitude', 'Longitude',
       'Value', 'UTC',
       'z01reflectivity', 'z01velocity',
       'z01spectrum_width', 'z01differential_phase',
       'z01differential_reflectivity', 'z01cross_correlation_ratio', 
        'u10', 'v10', 'd2m', 't2m', 'lai_hv', 'lai_lv', 'sp', 'sro', 'tp', 
       'popden', 'landcover', 'soil', 'glim', 'gebco','AOD','DQF','SZA','SAA']

# predictors = [
#         'GROUNDreflectivity','GROUNDvelocity', 'GROUNDspectrum_width', 
#         'GROUNDdifferential_phase','GROUNDdifferential_reflectivity', 
#         'GROUNDcross_correlation_ratio',
#        '1KMreflectivity', '1KMvelocity', '1KMspectrum_width',
#        '1KMdifferential_phase', '1KMdifferential_reflectivity',
#        '1KMcross_correlation_ratio', 'STDreflectivity', 'STDvelocity',
#        'STDspectrum_width', 'STDdifferential_phase',
#        'STDdifferential_reflectivity', 'STDcross_correlation_ratio', 'u10',
#        'v10', 'd2m', 't2m', 'lai_hv', 'lai_lv', 'sp', 'sro', 'tp', 
#        'popden', 'landcover', 'soil', 'glim', 'lith4', 'gebco']


predictors = [
       'z01reflectivity', 'z01velocity',
       'z01spectrum_width', 'z01differential_phase',
       'z01differential_reflectivity', 'z01cross_correlation_ratio', 
       'u10','v10', 'd2m', 't2m', 'lai_hv', 'lai_lv', 'sp', 'sro', 'tp', 
       'popden', 'landcover', 'soil', 'glim', 'gebco','AOD','SZA','SAA']


predictors_nonexrad = [ 'u10','v10','d2m', 't2m', 'lai_hv', 'lai_lv', 'sp', 'sro', 
                       'tp', 'popden', 'landcover', 'soil', 'glim',  'gebco','AOD','SZA','SAA']

predictors_noecmwf = [ 'z01reflectivity', 'z01velocity',
       'z01spectrum_width', 'z01differential_phase',
       'z01differential_reflectivity', 'z01cross_correlation_ratio', 
       'popden', 'landcover', 'soil', 'glim',  'gebco','AOD','SZA','SAA']


predictors_noaod = [
       'z01reflectivity', 'z01velocity',
       'z01spectrum_width', 'z01differential_phase',
       'z01differential_reflectivity', 'z01cross_correlation_ratio', 
       'u10','v10', 'd2m', 't2m', 'lai_hv', 'lai_lv', 'sp', 'sro', 'tp', 
       'popden', 'landcover', 'soil', 'glim', 'gebco']

df_model = pd.DataFrame(AirNow_Radar_ECMWF_df, columns = columns_useful)
df_model.describe().to_csv(os.path.join(Outdir,"describe.csv"))

plot_distr(df_model,"Value")
plot_nadistr(df_model)

## If the missing values are MAR, Drop rows with negative PM2.5 Values
df_model = df_model[df_model['Value'] > 0]
df_model = df_model[df_model['DQF'] < 2]
# df_model.loc[df_model['Value'] < 0,"Value"] = 0
## Filter out extreme cases
# df_model = df_model[df_model['Value'] < 150]

# ## Random split to resuce data size for tesing prupose
# df_model, df_model_large = train_test_split(df_model, test_size=0.999, random_state=1)

#Drop rows with NaN value in any columns
df_model = df_model.dropna()
df_model.reset_index(drop=True)

df_model["uv10"]= np.sqrt(df_model["u10"].values*df_model["u10"].values +df_model["v10"].values*df_model["v10"].values)

df_model["Month"] = df_model["UTC"].str.split("-", n = 2, expand = True)[1].astype(int)

# Update predictors for uv10
predictors = [
       'z01reflectivity', 'z01velocity', 'Month',
       'z01spectrum_width', 'z01differential_phase',
       'z01differential_reflectivity', 'z01cross_correlation_ratio', 
       'uv10', 'd2m', 't2m', 'lai_hv', 'lai_lv', 'sp', 'sro', 'tp', 
       'popden', 'landcover', 'soil', 'glim',  'gebco','AOD','SZA','SAA']


predictors_nonexrad = [ 'uv10', 'd2m', 't2m', 'lai_hv', 'lai_lv', 'sp', 'sro', 'tp', 'Month',
       'popden', 'landcover', 'soil', 'glim', 'gebco','AOD','SZA','SAA']


predictors_noecmwf = [ 'z01reflectivity', 'z01velocity',
       'z01spectrum_width', 'z01differential_phase',
       'z01differential_reflectivity', 'z01cross_correlation_ratio', 'Month',
       'popden', 'landcover', 'soil', 'glim',  'gebco','AOD','SZA','SAA']


predictors_noaod = [
       'z01reflectivity', 'z01velocity',
       'z01spectrum_width', 'z01differential_phase',
       'z01differential_reflectivity', 'z01cross_correlation_ratio', 
       'uv10', 'd2m', 't2m', 'lai_hv', 'lai_lv', 'sp', 'sro', 'tp', 
       'popden', 'landcover', 'soil', 'glim', 'gebco']


########################################################################
# # PCA of NEXRAD parameters
# scaler = MinMaxScaler()
# NEXRAD_raw = pd.DataFrame(df_model,columns=[
#        'z01reflectivity', 'z01velocity',
#        'z01spectrum_width', 'z01differential_phase',
#        'z01differential_reflectivity', 'z01cross_correlation_ratio'])

# NEXRAD_rescaled = scaler.fit_transform(NEXRAD_raw)
# pca = PCA(n_components=6)
# pca.fit(NEXRAD_rescaled)
# NEXRAD_pca = pca.transform(NEXRAD_rescaled)
# print("PCA 6")
# for i in range(6):
#     df_model["NEXRADf"+str(i+1)] = NEXRAD_pca[:,i]
    
    
    
# predictors = [
#        'NEXRADf1', 
# #     'NEXRADf2',
# #        'NEXRADf3', 'NEXRADf4',
# #        'NEXRADf5', 'NEXRADf6', 
#        'uv10', 'd2m', 't2m', 'lai_hv', 'lai_lv', 'sp', 'sro', 'tp', 'Month',
#        'popden', 'landcover', 'soil', 'glim', 'lith4', 'gebco','AOD']



# predictors_noecmwf = [
#        'NEXRADf1', 
# #         'NEXRADf2',
# #        'NEXRADf3', 'NEXRADf4',
# #        'NEXRADf5', 'NEXRADf6',  'Month',
#        'popden', 'landcover', 'soil', 'glim', 'lith4', 'gebco','AOD']
########################################################################

train, test = train_test_split(df_model, test_size=0.2)
print("Total observation %d, training observation %d, testing observation %d\n" %(len(df_model),len(train),len(test)))

## Feature and label separation
train_X = pd.DataFrame(train, columns = predictors)
train_Y = pd.DataFrame(train, columns = ['Value'])
test_X = pd.DataFrame(test, columns = predictors)
test_Y = pd.DataFrame(test, columns = ['Value'])

## Save the training and testing date for residual analysis
train_X.to_csv(os.path.join(Outdir,"train_x.csv"))
train_Y.to_csv(os.path.join(Outdir,"train_y.csv"))
test_X.to_csv(os.path.join(Outdir,"test_x.csv"))
test_Y.to_csv(os.path.join(Outdir,"test_y.csv"))

## NEXRAD with VAR
train_NOECMWF_X = pd.DataFrame(train, columns = predictors_noecmwf)
test_NOECMWF_X = pd.DataFrame(test, columns = predictors_noecmwf )
## ECMWF with Var
train_NONEXRAD_X = pd.DataFrame(train, columns = predictors_nonexrad)
test_NONEXRAD_X = pd.DataFrame(test, columns = predictors_nonexrad)

## No AOD
train_NOAOD_X = pd.DataFrame(train, columns = predictors_noaod)
test_NOAOD_X = pd.DataFrame(test, columns = predictors_noaod)


## Base model training and vaidation, change model
#------------------------------------------------------------------------------------
ETreg = ExtraTreesRegressor(n_estimators=100, random_state=1).fit(train_X, np.ravel(train_Y))
# joblib.dump(ETreg, os.path.join(Outdir,"ETreg_random.joblib"))
predicted_train_valid, predicted_test_valid = make_prediction(ETreg,train_X,test_X,train,test)
residual_plot(predicted_train_valid,"ETreg_random_train") 
model_scatterplot(predicted_train_valid,"ETreg_random_train")
residual_plot(predicted_test_valid,"ETreg_random_test") 
model_scatterplot(predicted_test_valid,"ETreg_random_test")
## Importance rank plot
importance_plot(ETreg,predictors,outname="Importance_ETreg_random")
# predicted_train_valid.to_csv("Predicted train ETreg_random.csv",index=False)
predicted_test_valid.to_csv("Predicted test ETreg_random.csv",index=False)



ETreg_NEXRAD = ExtraTreesRegressor(n_estimators=100, random_state=1).fit(train_NOECMWF_X, np.ravel(train_Y))
# joblib.dump(ETreg_NEXRAD, os.path.join(Outdir,"ETreg_NEXRAD.joblib"))
predicted_train_valid, predicted_test_valid = make_prediction(ETreg_NEXRAD,train_X,test_X,train,test)
residual_plot(predicted_train_valid,"ETreg_NEXRAD_train") 
model_scatterplot(predicted_train_valid,"ETreg_NEXRAD_train")
residual_plot(predicted_test_valid,"ETreg_NEXRAD_test") 
model_scatterplot(predicted_test_valid,"ETreg_NEXRAD_test")
importance_plot(ETreg_NEXRAD,predictors,outname="Importance_ETreg_NEXRAD")

# predicted_train_valid.to_csv("Predicted train ETreg_NEXRAD.csv",index=False)
predicted_test_valid.to_csv("Predicted test ETreg_NEXRAD.csv",index=False)

ETreg_ECMWF =  ExtraTreesRegressor(n_estimators=100, random_state=1).fit(train_NONEXRAD_X, np.ravel(train_Y))
# joblib.dump(ETreg_ECMWF, os.path.join(Outdir,"ETreg_ECMWF.joblib"))
predicted_train_valid, predicted_test_valid = make_prediction(ETreg_ECMWF,train_NONEXRAD_X,test_NONEXRAD_X,train,test)
residual_plot(predicted_train_valid,"ETreg_ECMWF_train") 
model_scatterplot(predicted_train_valid,"ETreg_ECMWF_train")
residual_plot(predicted_test_valid,"ETreg_ECMWF_test") 
model_scatterplot(predicted_test_valid,"ETreg_ECMWF_test")
importance_plot(ETreg_ECMWF,predictors,outname="Importance_ETreg_ECMWF")
# predicted_train_valid.to_csv("Predicted train ETreg_ECMWF.csv",index=False)
predicted_test_valid.to_csv("Predicted test ETreg_ECMWF.csv",index=False)

ETreg_NOAOD =  ExtraTreesRegressor(n_estimators=200, random_state=1).fit(train_NOAOD_X, np.ravel(train_Y))
# joblib.dump(ETreg_NOAOD, os.path.join(Outdir,"ETreg_NOAOD.joblib"))
predicted_train_valid, predicted_test_valid = make_prediction(ETreg_NOAOD,train_NOAOD_X,test_NOAOD_X,train,test)
residual_plot(predicted_train_valid,"ETreg_NOAOD_train") 
model_scatterplot(predicted_train_valid,"ETreg_NOAOD_train")
residual_plot(predicted_test_valid,"ETreg_NOAOD_test") 
model_scatterplot(predicted_test_valid,"ETreg_NOAOD_test")
importance_plot(ETreg_NOAOD,predictors,outname="Importance_ETreg_NOAOD")
# predicted_train_valid.to_csv("Predicted train ETreg_NOAOD.csv",index=False)
predicted_test_valid.to_csv("Predicted test ETreg_NOAOD.csv",index=False)
#-----------------------------------------------------------------------------------
print("Base model performance (n_estimators=200, random_state=1):")
print("NEXRAD-ECMWF Train", ETreg.score(train_X, np.ravel(train_Y)))
print("NEXRAD-ECMWF Test", ETreg.score(test_X, np.ravel(test_Y)))
print("NEXRAD Train ", ETreg_NEXRAD.score(train_NOECMWF_X, np.ravel(train_Y)))
print("NEXRAD Test", ETreg_NEXRAD.score(test_NOECMWF_X, np.ravel(test_Y)))
print("ECMWF Train ", ETreg_ECMWF.score(train_NONEXRAD_X, np.ravel(train_Y)))
print("ECMWF Test " , ETreg_ECMWF.score(test_NONEXRAD_X, np.ravel(test_Y)))
print("ENOAOD Train ", ETreg_NOAOD.score(train_NOAOD_X, np.ravel(train_Y)))
print("NOAOD Test " , ETreg_NOAOD.score(test_NOAOD_X, np.ravel(test_Y)))

print("\n")

# ## Baysian search optimization
# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 400, stop = 800, num = 3)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# # max_depth = [5,10,20,40]
# max_depth =[]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 3, 4, 5, 10, 15]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 3, 4, 5, 10, 15]
# # Method of selecting samples for training each tree
# bootstrap = [True,False]
# # Create the random grid

# bay_space = {
#                'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# print(bay_space,end="\n")



# # ETreg_random = RandomizedSearchCV(estimator = reg, param_distributions = random_grid, n_iter =200, cv = 3, verbose=1, random_state=1, n_jobs = 12).fit(train_X, np.ravel(train_Y))


# # log-uniform: understand as search over p = exp(x) by varying x
# ETreg_random = BayesSearchCV(
#     reg,
#     bay_space,
#     n_iter = n_iter,
#     n_jobs = n_jobs,
#     n_points = n_points,
#     cv=cv
# ).fit(train_X, np.ravel(train_Y))

# joblib.dump(ETreg_random.best_estimator_, os.path.join(Outdir,"ETreg_random.joblib"))
# print("best estimator for ETreg_random:",ETreg_random.best_params_)
# print("Optimized model performance:")
# print("NEXRAD-ECMWF Train", ETreg_random.best_estimator_.score(train_X, np.ravel(train_Y)))
# print("NEXRAD-ECMWF Test", ETreg_random.best_estimator_.score(test_X, np.ravel(test_Y)))
# print("\n")

# #Base model and optimized model accuracy comparison
# print("Optimized model and base model comparison:")
# base_accuracy = evaluate(ETreg, test_X, np.ravel(test_Y))
# best_random = ETreg_random.best_estimator_
# random_accuracy = evaluate(best_random, test_X, np.ravel(test_Y))
# print('ETreg Improvement of {:0.2f}%.\n'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))



# ## Importance rank plot
# importance_plot(ETreg_random.best_estimator_,predictors,outname="Importance_ETreg_random")

## Residual Analysis
# predicted_train_valid, predicted_test_valid = make_prediction(ETreg_random.best_estimator_,train_X,test_X,train,test)
# residual_plot(predicted_train_valid,"ETreg_random_train") 
# model_scatterplot(predicted_train_valid,"ETreg_random_train")
# residual_plot(predicted_test_valid,"ETreg_random_test") 
# model_scatterplot(predicted_test_valid,"ETreg_random_test")

# print("\n")
# # ## Visulization tree
# # index = np.random.choice(ETreg_random.best_params_["n_estimators"],5,replace=False)
# # for i in index:
# #     show_tree("ETreg_random",ETreg_random.best_estimator_,predictors,train_Y, i,Outdir)

# ###################################################################################################

# # ETreg_NEXRAD_random = RandomizedSearchCV(estimator = reg, param_distributions = random_grid, n_iter =200, cv =3 , verbose=1, random_state=1, n_jobs = 12).fit(train_NOECMWF_X, np.ravel(train_Y))

# # log-uniform: understand as search over p = exp(x) by varying x
# ETreg_NEXRAD_random  = BayesSearchCV(
#     reg,
#     bay_space,
#     n_iter = n_iter,
#     n_jobs = n_jobs,
#     n_points = n_points,
#     cv=cv
# ).fit(train_NOECMWF_X, np.ravel(train_Y))

# joblib.dump(ETreg_NEXRAD_random.best_estimator_, os.path.join(Outdir,"ETreg_NEXRAD_random.joblib"))

# print("best estimator for ETreg_NEXRAD_random:",ETreg_NEXRAD_random.best_params_)

# print("Optimized model performance:")
# print("NEXRAD Train ", ETreg_NEXRAD_random.best_estimator_.score(train_NOECMWF_X, np.ravel(train_Y)))
# print("NEXRAD Test", ETreg_NEXRAD_random.best_estimator_.score(test_NOECMWF_X, np.ravel(test_Y)))
# print("\n")

# base_accuracy = evaluate(ETreg_NEXRAD, test_NOECMWF_X, np.ravel(test_Y))
# best_random = ETreg_NEXRAD_random.best_estimator_
# random_accuracy = evaluate(best_random, test_NOECMWF_X, np.ravel(test_Y))
# print('ETreg_NEXRAD_random Improvement of {:0.2f}%.\n'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))



# importance_plot(ETreg_NEXRAD_random.best_estimator_,predictors_noecmwf,outname="Importance_ETreg_NEXRAD_random")

# predicted_train_valid, predicted_test_valid = make_prediction(ETreg_NEXRAD_random.best_estimator_,train_NOECMWF_X,test_NOECMWF_X,train,test)
# residual_plot(predicted_train_valid,"ETreg_NEXRAD_random_train") 
# model_scatterplot(predicted_train_valid,"ETreg_NEXRAD_random_train")
# residual_plot(predicted_test_valid,"ETreg_NEXRAD_random_test") 
# model_scatterplot(predicted_test_valid,"ETreg_NEXRAD_random_test")

# print("\n")
# # ## Visulization tree
# # index = np.random.choice(ETreg_NEXRAD_random.best_params_["n_estimators"],5,replace=False)
# # for i in index:
# #     show_tree("ETreg_NEXRAD_random",ETreg_NEXRAD_random.best_estimator_,predictors_noecmwf,train_Y, i,Outdir)

# ######################################################################################################


# # ETreg_ECMWF_random = RandomizedSearchCV(estimator = reg, param_distributions = random_grid, n_iter =200, cv = 3, verbose=1, random_state=1, n_jobs = 12).fit(train_NONEXRAD_X, np.ravel(train_Y))

# ETreg_ECMWF_random = BayesSearchCV(
#     reg,
#     bay_space,
#     n_iter = n_iter,
#     n_jobs = n_jobs,
#     n_points = n_points,
#     cv=cv
# ).fit(train_NONEXRAD_X, np.ravel(train_Y))

# joblib.dump(ETreg_ECMWF_random.best_estimator_, os.path.join(Outdir,"ETreg_ECMWF_random.joblib"))

      
# print("best estimator for: ETreg_ECMWF_random",ETreg_ECMWF_random.best_params_)

# print("Optimized model performance:")
# print("ECMWF Train ", ETreg_ECMWF_random.best_estimator_.score(train_NONEXRAD_X, np.ravel(train_Y)))
# print("ECMWF Test " , ETreg_ECMWF_random.best_estimator_.score(test_NONEXRAD_X, np.ravel(test_Y)))
# print("\n")

# base_accuracy = evaluate(ETreg_ECMWF, test_NONEXRAD_X, np.ravel(test_Y))
# best_random = ETreg_ECMWF_random.best_estimator_
# random_accuracy = evaluate(best_random, test_NONEXRAD_X, np.ravel(test_Y))
# print('ETreg_ECMWF_random Improvement of {:0.2f}%.\n'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))



# importance_plot(ETreg_ECMWF_random.best_estimator_,predictors_nonexrad,outname="Importance_ETreg_ECMWF_random")

# predicted_train_valid, predicted_test_valid = make_prediction(ETreg_ECMWF_random.best_estimator_,train_NONEXRAD_X,test_NONEXRAD_X,train,test)
# residual_plot(predicted_train_valid,"ETreg_ECMWF_random_train") 
# model_scatterplot(predicted_train_valid,"ETreg_ECMWF_random_train")
# residual_plot(predicted_test_valid,"ETreg_ECMWF_random_test") 
# model_scatterplot(predicted_test_valid,"ETreg_ECMWF_random_test")

# # ## Visulization tree
# # index = np.random.choice(ETreg_ECMWF_random.best_params_["n_estimators"],5,replace=False)
# # for i in index:
# #     show_tree("ETreg_ECMWF_random",ETreg_ECMWF_random.best_estimator_,predictors_nonexrad,train_Y, i,Outdir)

# # ## Shap value plot
# # shap_values = shap_plot(ETreg_random.best_estimator_,train_X,outname="Shap_ETreg_random")
# # # abs_shap(shap_values,train_X,outname="absShap_ETreg_random") 

# # shap_values = shap_plot(ETreg_NEXRAD_random.best_estimator_,train_NOECMWF_X,outname="Shap_ETreg_NEXRAD_random")
# # # abs_shap(shap_values,train_NOECMWF_X,outname="absShap_ETreg_NEXRAD_random") 

# # shap_values = shap_plot(ETreg_ECMWF_random.best_estimator_,train_NONEXRAD_X,outname="Shap_ETreg_ECMWF_radnom")
# # # abs_shap(shap_values,train_NONEXRAD_X,outname="absShap_ETreg_ECMWF_radnom") 


print("Done")
    
# sys.stdout.close()