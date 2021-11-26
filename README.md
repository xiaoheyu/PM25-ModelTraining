# Project Overview 
#### Please cite and refer to the following paper for details.
## PM2.5 Modeling and Historical Reconstruction over the Continental USA Utilizing GOES-16 AOD https://doi.org/10.3390/rs13234788 
#### There are 4 stages for the entire project.
#### Stage 1 (Code available in repo PM25-DataSource): Data download and visulization, including ECMWF, PBLH, GOES16-AOD, Landcover, Soil order, Population density, Elevation, and Lithology.
#### Stage 2 (Code available in repo PM25-DataMaching): Once data are downloaded, all variables will be macthed with PM2.5 groupd observation. A macthed data table is generated for machine learning model training.
#### Stage 3 (): Machine learning moldeing training
#### Stage 4 (Code available in repo PM25-Estimation-Reconstrcution or PM25-Estimation-Reconstrcution-Local ): Once model and data are ready, this code make pm25 estimations and generate visulization results.

# PM25-ModelTraining
Code in this repo establishes machine learning model for pm2.5 estiamtion, including data pre-processing, model training, hyper-parameter optimization and result visulizaiton.
Datatable generated from Stage 2 is requred to run this code. 
