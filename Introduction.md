Introduction
================
Francis Kyalo

## Introduction

CARET stands for `classification and regression training`

The package contains tools for:

> data splitting

> pre-processing

> feature selection

> model tuning using resampling

> variable importance estimation

## Pre-processing

Caret assumes that all of the data are numeric (i.e. factors have been
converted to dummy variables via `model.matrix`, `dummyVars` or other
means).

### Creating dummy variables

The function `dummyVars` can be used to generate a complete (less than
full rank parameterized) set of dummy variables from one or more
factors. The function takes a formula and a data set and outputs an
object that can be used to create the dummy variables using the predict
method.

For example:

    dummies <- dummyVars(survived ~ ., data = etitanic)
    head(predict(dummies, newdata = etitanic))

> Note there is no intercept and each factor has a dummy variable for
> each level, so this parameterization may not be useful for some model
> functions, such as `lm`.

### Zero- and Near Zero-Variance Predictors

In some situations, the data generating mechanism can create predictors
that only have a single unique value (i.e. a “zero-variance predictor”).
For many models (excluding tree-based models), this may cause the model
to crash or the fit to be unstable.

Similarly, predictors might have only a handful of unique values that
occur with very low frequencies.

The concern here that these predictors may become zero-variance
predictors when the data are split into cross-validation/bootstrap
sub-samples or that a few samples may have an undue influence on the
model. These “near-zero-variance” predictors may need to be identified
and eliminated prior to modeling.

The `nearZeroVar` function can be used to identify near zero-variance
variables (the `saveMetrics` argument can be used to show the details
and usually defaults to `FALSE`)

for example:

    nzv <- nearZeroVar(mdrrDescr, saveMetrics= TRUE)
    nzv[nzv$nzv,][1:10,]

    ##        freqRatio percentUnique zeroVar  nzv
    ## nTB     23.00000     0.3787879   FALSE TRUE
    ## nBR    131.00000     0.3787879   FALSE TRUE
    ## nI     527.00000     0.3787879   FALSE TRUE
    ## nR03   527.00000     0.3787879   FALSE TRUE
    ## nR08   527.00000     0.3787879   FALSE TRUE
    ## nR11    21.78261     0.5681818   FALSE TRUE
    ## nR12    57.66667     0.3787879   FALSE TRUE
    ## D.Dr03 527.00000     0.3787879   FALSE TRUE
    ## D.Dr07 123.50000     5.8712121   FALSE TRUE
    ## D.Dr08 527.00000     0.3787879   FALSE TRUE

### Identifying Correlated Predictors

While there are some models that thrive on correlated predictors (such
as pls), other models may benefit from reducing the level of correlation
between the predictors.

Given a correlation matrix, the `findCorrelation` function uses the
following algorithm to flag predictors for removal:

    highlyCorDescr <- findCorrelation(descrCor, cutoff = .75)
    filteredDescr <- filteredDescr[,-highlyCorDescr]

### Linear Dependencies

The function `findLinearCombos` uses the QR decomposition of a matrix to
enumerate sets of linear combinations (if they exist). For example,
consider the following matrix that is could have been produced by a
less-than-full-rank parameterizations of a two-way experimental layout:

    ltfrDesign <- matrix(0, nrow=6, ncol=6)
    ltfrDesign[,1] <- c(1, 1, 1, 1, 1, 1)
    ltfrDesign[,2] <- c(1, 1, 1, 0, 0, 0)
    ltfrDesign[,3] <- c(0, 0, 0, 1, 1, 1)
    ltfrDesign[,4] <- c(1, 0, 0, 1, 0, 0)
    ltfrDesign[,5] <- c(0, 1, 0, 0, 1, 0)
    ltfrDesign[,6] <- c(0, 0, 1, 0, 0, 1)

Note that columns two and three add up to the first column. Similarly,
columns four, five and six add up the first column. findLinearCombos
will return a list that enumerates these dependencies. For each linear
combination, it will incrementally remove columns from the matrix and
test to see if the dependencies have been resolved. findLinearCombos
will also return a vector of column positions can be removed to
eliminate the linear dependencies:

    comboInfo <- findLinearCombos(ltfrDesign)
    comboInfo

    ## $linearCombos
    ## $linearCombos[[1]]
    ## [1] 3 1 2
    ## 
    ## $linearCombos[[2]]
    ## [1] 6 1 4 5
    ## 
    ## 
    ## $remove
    ## [1] 3 6

    ltfrDesign[, -comboInfo$remove]

    ##      [,1] [,2] [,3] [,4]
    ## [1,]    1    1    1    0
    ## [2,]    1    1    0    1
    ## [3,]    1    1    0    0
    ## [4,]    1    0    1    0
    ## [5,]    1    0    0    1
    ## [6,]    1    0    0    0

### The `preProcess` function

The `preProcess` class can be used for many operations on predictors,
including centering and scaling. The function preProcess estimates the
required parameters for each operation and `predict.preProcess` is used
to apply them to specific data sets. This function can also be
interfaces when calling the train function.

Several types of techniques are described in the next few sections and
then another example is used to demonstrate how multiple methods can be
used. Note that, in all cases, the `preProcess` function estimates
whatever it requires from a specific data set (e.g. the training set)
and then applies these transformations to any data set without
recomputing the values.

#### Centering and scaling

    preProcValues <- preProcess(training, method = c("center", "scale"))

    trainTransformed <- predict(preProcValues, training)
    testTransformed <- predict(preProcValues, test)

The `preProcess` option `"range"`scales the data to the interval between
zero and one.

#### Imputation using `knnImpute`

The `preProcess` can be used to impute data sets based only on
information in the training set. One method of doing this is with
K-nearest neighbors. For an arbitrary sample, the K closest neighbors
are found in the training set and the value for the predictor is imputed
using these values (e.g. using the mean). Using this approach will
automatically trigger `preProcess` to center and scale the data,
regardless of what is in the method argument. Alternatively, bagged
trees can also be used to impute. For each predictor in the data, a
bagged tree is created using all of the other predictors in the training
set. When a new sample has a missing predictor value, the bagged model
is used to predict the value. While, in theory, this is a more powerful
method of imputing, the computational costs are much higher than the
nearest neighbor technique.

    preproc_opts <- preProcess(data, method = c("knnImpute"), k = 5)
    imputed_data <- predict(preproc_opts, newdata = data)

#### Transforming Predictors using `pca`

In some cases, there is a need to use principal component analysis (PCA)
to transform the data to a smaller sub–space where the new variable are
uncorrelated with one another. The preProcess class can apply this
transformation by including “pca” in the method argument. Doing this
will also force scaling of the predictors. Note that when PCA is
requested, predict.preProcess changes the column names to PC1, PC2 and
so on.

    preproc_opts <- preProcess(data, method = c("center", "scale", "pca"), pcaComp = 10)
    pca_transformed_data <- predict(preproc_opts, newdata = data)

Another option, “BoxCox” will estimate a Box–Cox transformation on the
predictors if the data are greater than zero.

    preProcValues2 <- preProcess(training, method = "BoxCox")
    trainBC <- predict(preProcValues2, training)
    testBC <- predict(preProcValues2, test)

This transformation requires the data to be greater than zero. Two
similar transformations, the `Yeo-Johnson` and exponential
transformation

### Putting it all together

    pp_no_nzv <- preProcess(schedulingData[, -8], 
                            method = c("center", "scale", "YeoJohnson", "nzv"))
                            
    predict(pp_no_nzv, newdata = schedulingData[1:6, -8])                      

## Data splitting

The function `createDataPartition` can be used to create balanced splits
of the data. If the y argument to this function is a factor, the random
sampling occurs within each class and should preserve the overall class
distribution of the data. For example, to create a single 80/20% split
of the iris data:

    trainIndex <- createDataPartition(iris$Species, p = .8, 
                                      list = FALSE, 
                                      times = 1)

    irisTrain <- iris[ trainIndex,]
    irisTest  <- iris[-trainIndex,]

The `list = FALSE` avoids returning the data as a list. This function
also has an argument, `times`, that can create multiple splits at once;
the data indices are returned in a list of integer vectors. Similarly,
`createResample` can be used to make simple bootstrap samples and
`createFolds` can be used to generate balanced cross–validation
groupings from a set of data.

## Model Training and Parameter Tuning

The `train` function can be used to

- evaluate, using resampling, the effect of model tuning parameters on
  performance

- choose the “optimal” model across these parameters

- estimate model performance from a training set

First, a specific model must be chosen. Currently, 238 are available
using caret

Once the model and tuning parameter values have been defined, the type
of resampling should be also be specified. Currently, k-fold
cross-validation (once or repeated), leave-one-out cross-validation and
bootstrap (simple estimation or the 632 rule) resampling methods can be
used by train. After resampling, the process produces a profile of
performance measures is available to guide the user as to which tuning
parameter values should be chosen. By default, the function
automatically chooses the tuning parameters associated with the best
value, although different algorithms can be used

    library(caret)
    set.seed(998)
    inTraining <- createDataPartition(Sonar$Class, p = .75, list = FALSE)
    training <- Sonar[ inTraining,]
    testing  <- Sonar[-inTraining,]

## Basic Parameter Tuning

By default, simple bootstrap resampling is used for line 3 in the
algorithm above. Others are available, such as repeated K-fold
cross-validation, leave-one-out etc. The function trainControl can be
used to specifiy the type of resampling:

    fitControl <- trainControl(## 10-fold CV
                               method = "repeatedcv",
                               number = 10,
                               ## repeated ten times
                               repeats = 10)

The first two arguments to train are the predictor and outcome data
objects, respectively. The third argument, method, specifies the type of
model To illustrate, we will fit a boosted tree model via the gbm
package. The basic syntax for fitting this model using repeated
cross-validation is shown below:

    set.seed(825)
    gbmFit1 <- train(Class ~ ., data = training, 
                     method = "gbm", 
                     trControl = fitControl,
                     ## This last option is actually one
                     ## for gbm() that passes through
                     verbose = FALSE)
    gbmFit1

    ## Stochastic Gradient Boosting 
    ## 
    ## 157 samples
    ##  60 predictor
    ##   2 classes: 'M', 'R' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold, repeated 10 times) 
    ## Summary of sample sizes: 141, 142, 141, 142, 141, 142, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  Accuracy   Kappa    
    ##   1                   50      0.7935784  0.5797839
    ##   1                  100      0.8171078  0.6290208
    ##   1                  150      0.8219608  0.6386184
    ##   2                   50      0.8041912  0.6027771
    ##   2                  100      0.8302059  0.6556940
    ##   2                  150      0.8283627  0.6520181
    ##   3                   50      0.8110343  0.6170317
    ##   3                  100      0.8301275  0.6551379
    ##   3                  150      0.8310343  0.6577252
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## 
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final values used for the model were n.trees = 150,
    ##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.

For these models, `train` can automatically create a grid of tuning
parameters. By default, if p is the number of tuning parameters, the
grid size is 3^p. 

### Reproducibility
