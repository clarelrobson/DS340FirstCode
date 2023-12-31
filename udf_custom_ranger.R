#As we have different sizes of features subsets, we need an individual value for tuning `mtry`. The upper `mtry` level should be oriented on the total number of predictors - usually the square root of the number of predictors is a good rule of thumb. A custom learner from the mlr extralearners enables us to use the same paramSet for all models (without having to create separate ones with different upper levels of `mtry`). The custom learner uses `mtry.pow`, which sets mtry to `round(p^mtry.pow)`.
#This function is copied from https://github.com/mlr-org/mlr-extralearner/blob/master/R/RLearner_classif_ranger_mtry_pow.R out of the mlr extralearners repository.



# Learner definition
makeRLearner.classif.ranger.pow = function() {
  makeRLearnerClassif(
    cl = "classif.ranger.pow",
    package = "ranger",
    par.set = makeParamSet(
      makeIntegerLearnerParam(id = "num.trees", lower = 1L, default = 500L),
      makeNumericLearnerParam(id = "mtry.power", lower = 0, upper = 1),
      makeIntegerLearnerParam(id = "min.node.size", lower = 1L),
      makeLogicalLearnerParam(id = "replace", default = TRUE),
      makeNumericLearnerParam(id = "sample.fraction", lower = 0L, upper = 1L),
      makeNumericVectorLearnerParam(id = "split.select.weights", lower = 0, upper = 1),
      makeUntypedLearnerParam(id = "always.split.variables"),
      makeDiscreteLearnerParam("respect.unordered.factors", values = c("ignore", "order", "partition"), default = "ignore"),
      makeDiscreteLearnerParam(id = "importance", values = c("none", "impurity", "permutation"), default = "none", tunable = FALSE),
      makeLogicalLearnerParam(id = "write.forest", default = TRUE, tunable = FALSE),
      makeLogicalLearnerParam(id = "scale.permutation.importance", default = FALSE, requires = quote(importance == "permutation"), tunable = FALSE),
      makeIntegerLearnerParam(id = "num.threads", lower = 1L, when = "both", tunable = FALSE),
      makeLogicalLearnerParam(id = "save.memory", default = FALSE, tunable = FALSE),
      makeLogicalLearnerParam(id = "verbose", default = TRUE, when = "both", tunable = FALSE),
      makeIntegerLearnerParam(id = "seed", when = "both", tunable = FALSE),
      makeDiscreteLearnerParam(id = "splitrule", values = c("gini", "extratrees"), default = "gini"),
      makeIntegerLearnerParam(id = "num.random.splits", lower = 1L, default = 1L, requires = quote(splitrule == "extratrees")),
      makeLogicalLearnerParam(id = "keep.inbag", default = FALSE, tunable = FALSE)
    ),
    par.vals = list(num.threads = 1L, verbose = FALSE, respect.unordered.factors = "order"),
    properties = c("twoclass", "multiclass", "prob", "numerics", "factors", "ordered", "featimp", "weights", "oobpreds"),
    name = "Random Forests",
    short.name = "ranger.pow",
    note = " Mtry is set to `round(p^mtry.pow)`
      By default, internal parallelization is switched off (`num.threads = 1`), `verbose` output is disabled,
      `respect.unordered.factors` is set to `order` for all splitrules.
      If predict.type='prob' we set 'probability=TRUE' in ranger.",
    callees = "ranger"
  )
}

# Training
trainLearner.classif.ranger.pow = function(.learner, .task, .subset, .weights = NULL, mtry.power, ...) {
  tn = getTaskTargetNames(.task)
  
  if(!missing(mtry.power)) {
    p = getTaskNFeats(.task)
    mtry = round(p^mtry.power)
  } else {
    mtry = NULL
  }
  ranger::ranger(formula = NULL, dependent.variable = tn, data = getTaskData(.task, .subset),
                 probability = (.learner$predict.type == "prob"), case.weights = .weights, mtry = mtry, ...)
}

# Prediction
predictLearner.classif.ranger.pow = function(.learner, .model, .newdata, ...) {
  p = predict(object = .model$learner.model, data = .newdata, ...)
  return(p$predictions)
}

# Out-of-bag prediction
getOOBPredsLearner.classif.ranger.pow = function(.learner, .model) {
  getLearnerModel(.model, more.unwrap = TRUE)$predictions
}

# Feature importance
getFeatureImportanceLearner.classif.ranger.pow = function(.learner, .model, ...) {
  has.fiv = .learner$par.vals$importance
  if (is.null(has.fiv) || has.fiv == "none") {
    stop("You must set the learners parameter value for importance to
         'impurity' or 'permutation' to compute feature importance")
  }
  mod = getLearnerModel(.model, more.unwrap = TRUE)
  ranger::importance(mod)
}