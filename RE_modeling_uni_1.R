## ----setup-----------------------------------------------------------------------------------------------------------------

setwd(normalizePath('../Data/Prep Data'))


## --------------------------------------------------------------------------------------------------------------------------
library(tidyverse)
library(mlr)
library(mlrCPO) 
library(parallelMap) 
library(skimr) 
library(ggplot2) 
library(psych) 

source("../../Code/udf_custom_ranger.R")
source("../../Code/udf_drop_constants.R")


## --------------------------------------------------------------------------------------------------------------------------
n_iter <- 50
cv_inner <- 5
cv_outer <- 10
nested <- 5

## --------------------------------------------------------------------------------------------------------------------------
load('feature_sets.RData')

df_uni_1_clean <- df_uni_1_clean %>% select(-net_dm_closeness)

# assign feature matrix
df_feat <- df_uni_1_clean

# drop user_id column 
df_feat$user_id <- NULL

# subset features based on institutional data only
df_feat_inst <- df_feat %>% select(starts_with("inst_"), status)

# subset all engagement features (app and community engagement features)
df_feat_e <- df_feat %>% select(starts_with("ce_"), starts_with("ae_"), status)


## --------------------------------------------------------------------------------------------------------------------------
task_inst = makeClassifTask("RetentionInst", df_feat_inst, 
                               target = "status", 
                               positive = "Transferred",
                               fixup.data = "no", 
                               check.data = FALSE)

task_e = makeClassifTask("RetentionEngagement", df_feat_e, 
                            target = "status", 
                            positive = "Transferred",
                            fixup.data = "no",
                            check.data = FALSE)

task_all = makeClassifTask("RetentionAll", df_feat,
                              target = "status",
                              positive = "Transferred",
                              fixup.data = "no",
                              check.data = FALSE)


## --------------------------------------------------------------------------------------------------------------------------
# set up imputation method for Random Forest (2*max value)
my.imputeMax = function() {
  makeImputeMethod(learn = function(data, target, col) 2 * max(data[[col]], na.rm = TRUE), impute = mlr:::simpleImpute)
}

# Pre-processing Random Forest 
cpoRFLearner = cpoFixFactors(fix.factors.prediction = TRUE) %>>% #factor levels are kept the same in training and prediction
  cpoDropMostlyConstants(ratio = 0.1, affect.type = "numeric") %>>% #drop all numerical columns that are 90% or more constant, i.e. columns with values deviating only up to 10% from the mode
  cpoDropConstants(ignore.na = TRUE, affect.type = c("ordered", "factor")) %>>% #drop all factor columns that contain only one value
  cpoImputeAll(classes = list(numeric = my.imputeMax(), integer = my.imputeMax(), factor = imputeMode())) %>>% #imputation of missing feature values: numeric features by 2*max to give RF a chance of splitting meaningfully and factors by mode
  cpoSmote() #perform “Synthetic Minority Oversampling TEchnique” sample generation to handle class imbalance:  NULL sets rate to the ratio <majority prevalence> / <minority prevalence>

# Pre-processing LGM GLM
cpoLinLearner = cpoFixFactors(fix.factors.prediction = TRUE) %>>%
  cpoDropMostlyConstants(ratio = 0.1, affect.type = "numeric") %>>%
  cpoDropConstants(ignore.na = TRUE, affect.type = c("ordered", "factor")) %>>%
  cpoImputeAll(classes = list(numeric = imputeMedian(), integer = imputeMedian(), factor = imputeMode())) %>>% #imputation of missing feature values: numeric features by median to make more robust against outliers and factors by mode
  cpoSmote()
# info for cvglmnet: factors automatically get converted to dummy columns, ordered factors to integer


## --------------------------------------------------------------------------------------------------------------------------
# Learner Random Forest
lrnRF = makeLearner("classif.ranger.pow", num.trees = 1000, importance = "permutation", predict.type = "prob", id = "RF")

# attach the pre-processing pipeline to the Learner
lrnRF = cpoRFLearner %>>% lrnRF #fuse learner with pre-processing  (removal of constant features, imputation, oversampling)

# create tunable set of parameters
params_RF = makeParamSet(
  makeNumericParam("mtry.power", lower = 0, upper = 1), #number of randomly selected predictor variables per split, this uses mtry round(p^mtry.pow) to stay flexible for different numbers of featues p; general recommendation: sqrt(p), (Friedman, Hastie, & Tibshirani, 2009)
  makeIntegerParam("min.node.size", lower = 1, upper = 5), #size of (number of observation in) terminal nodes
  makeIntegerParam("sw.rate", lower = 2, upper = 15), #times of synthetic minority instances over the original number of minority instances
  makeIntegerParam("sw.nn", lower = 1, upper = 10) #number of nearest neighbors used for sampling new values
)

# define inner resampling and optimization strategy
tunectrlRF = makeTuneControlRandom(maxit = n_iter) #for tuning with 40 iterations
innerRF = makeResampleDesc("CV", iters = cv_inner,  stratify = TRUE) #specify resampling strategy, 5-fold-cross-validation

# combine to learner model #all measures are returned, optimization happens for the first, f1
learner_RF = makeTuneWrapper(learner = lrnRF, 
                             resampling = innerRF, 
                             par.set = params_RF, 
                             control = tunectrlRF, 
                             show.info = TRUE, 
                             measures = list(f1, auc, tpr, tnr)) 

learner_RF$id = "Random Forest"


## --------------------------------------------------------------------------------------------------------------------------
# Turn off parameter checking, still shows warning
configureMlr(on.par.without.desc = "warn") # The parameter standardize.response = TRUE is not registered in the learner’s parameter set in mlr; By turning off the parameter checking, the parameter setting will be passed to the underlying function; warning will still be shown

# Learner GLM
lrnGLM = makeLearner("classif.cvglmnet", standardize = TRUE, standardize.response = TRUE, predict.type = "prob", id = "GLM")

# Turn on parameter checking again
configureMlr(on.par.without.desc = "stop")

# attach the pre-processing pipeline to the Learner
lrnGLM = cpoLinLearner %>>% lrnGLM

# create tunable set of parameters
paramsGLM = makeParamSet(
  makeDiscreteParam("s", values = c("lambda.1se", "lambda.min")), #“one-standard-error” rule (Friedman, Hastie, & Tibshirani, 2009)
  makeNumericParam("alpha", lower = 0, upper = 1), #the elastic-net mixing parameter, controls the distribution
  makeIntegerParam("sw.rate", lower = 2, upper = 15), #times of synthetic minority instances over the original number of majority instances
  makeIntegerParam("sw.nn", lower = 1, upper = 10) #number of nearest neighbors used for sampling new values
  )

# define inner resampling and optimization strategy
tunectrlGLM = makeTuneControlRandom(maxit = n_iter) #for tuning with 40 iterations
innerGLM = makeResampleDesc("CV", iters = cv_inner, stratify = TRUE) #specify resampling strategy, 5-fold-cross-validation  

# combine to learner model
learner_GLM = makeTuneWrapper(learner = lrnGLM, resampling = innerGLM, par.set = paramsGLM, control = tunectrlGLM, show.info = TRUE, measures = list(f1, auc, tpr, tnr))
learner_GLM$id = "GLM eNet"


## --------------------------------------------------------------------------------------------------------------------------
# Create featureless learner
lrn_Featureless = makeLearner("classif.featureless", predict.type = "prob")
lrn_Featureless$id = "Featureless"

# List of learners to be compared
lrns = list(lrn_Featureless, #featureless learner as baseline, constantly predicting majority class
            learner_RF, #TuneWrapper, functions as learner
            learner_GLM) #TuneWrapper, functions as learner

# Lists of all tasks
tsks = list(task_inst, task_e, task_all)

# List of measures for model assessment
msrs = list(f1, auc, tpr, tnr, timetrain)

# outer resampling loop
outer_res = makeResampleDesc("RepCV", reps = nested, folds = cv_outer, stratify = TRUE) #stratified cross-validation, 5x10 RepCV; no manual instantiation is necessary; it is instantiated automatically once for each Task so that the same instance is used for all learners applied to a single task
outer_simple = makeResampleDesc("CV", iters = cv_outer, stratify = TRUE) #10-fold CV for simpler resampling


## --------------------------------------------------------------------------------------------------------------------------
# set seed again for the benchmarking of the uni 2 models - otherwise, the results will not be reproducible

set.seed(123, kind = "L'Ecuyer")
parallelStartSocket(parallel::detectCores(), level = "mlr.resample")
parallelExport("trainLearner.classif.ranger.pow", "predictLearner.classif.ranger.pow", level = "mlr.resample") #needed to make custom learner work in parallel
parallelLibrary("mlrCPO")
parallelSource("../../Code/udf_drop_constants.R", level = "mlr.resample") #needed to make pre-processing work in parallel

# Uni 1
bmr_uni_1 = mlr::benchmark(learners = lrns, tasks = tsks, resamplings = outer_simple, measures = msrs, show.info = TRUE)

parallelStop()


## --------------------------------------------------------------------------------------------------------------------------
save(bmr_uni_1, file = "../../Results/benchmark_uni_1.RData")


## --------------------------------------------------------------------------------------------------------------------------
set.seed(123, kind = "L'Ecuyer")

#parallelStartSocket(parallel::detectCores(), level = "mlr.resample")
#parallelExport("trainLearner.classif.ranger.pow", "predictLearner.classif.ranger.pow", level = "mlr.resample")
#parallelLibrary("mlrCPO")
#parallelSource("../../Code/udf_drop_constants.R", level = "mlr.resample")
                   
parallelStartSocket(parallel::detectCores(), level = "mlr.tuneParams")
parallelExport("trainLearner.classif.ranger.pow", "predictLearner.classif.ranger.pow", level = "mlr.tuneParams")
parallelLibrary("mlrCPO")
parallelSource("../../Code/udf_drop_constants.R", level = "mlr.tuneParams")
                   

# RF
model_rf_inst_uni_1 = train(learner_RF, task_inst)
model_rf_e_uni_1 = train(learner_RF, task_e)
model_rf_all_uni_1 = train(learner_RF, task_all)

# GLM
model_glm_inst_uni_1 = train(learner_GLM, task_inst)
model_glm_e_uni_1 = train(learner_GLM, task_e)
model_glm_all_uni_1 = train(learner_GLM, task_all)

parallelStop()


## --------------------------------------------------------------------------------------------------------------------------
save(model_rf_inst_uni_1, model_rf_e_uni_1, model_rf_all_uni_1,
     model_glm_inst_uni_1, model_glm_e_uni_1, model_glm_all_uni_1, 
     file = "../../Results/models_trained_uni_1.RData")

