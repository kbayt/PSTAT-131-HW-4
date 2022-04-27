library(ggplot2)
library(tidyverse)
library(tidymodels)
library(corrplot)
library(ggthemes)
tidymodels_prefer()
library(ISLR)
library(yardstick)
library(corrr)
library(discrim)
library(poissonreg)
library(klaR)
library(tune)
set.seed(4857)
titanic <- read.csv("C:\\titanic.csv")
view(titanic)

titanic$survived <- factor(titanic$survived, levels = c("Yes", "No"))
titanic$pclass <- factor(titanic$pclass)

## QUESTION 1
# split the data stratifying on survived
titanic_split <- initial_split(titanic, prop = 0.7,
                               strata = survived)
titanic_train <- training(titanic_split)
titanic_test <- testing(titanic_split)
# check num of observations in each set
dim(titanic_train)
dim(titanic_test)

# create recipe (same as hw3
titanic_recipe <- recipe(survived ~ pclass + sex +
                           age + sib_sp + parch + fare, 
                         data = titanic_train) %>%
  step_impute_linear(age) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_interact(terms =~starts_with("sex"):fare) %>%
  step_interact(terms=~age:fare)
summary(titanic_recipe)


## QUESTION 2
# fold the training data
# use k-fold cross-validation w k=10
titanic_folds <- vfold_cv(titanic_train, v=10)
titanic_folds

## QUESTION 4
# set up 3 workflows
# 1. logistic regression with glm engine
log_reg <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")
log_wkflow <- workflow() %>%
  add_model(log_reg) %>%
  add_recipe(titanic_recipe)
# 2. linear discriminant analysis with MASS engine
lda_mod <- discrim_linear() %>%
  set_mode("classification") %>%
  set_engine("MASS")
lda_wkflow <- workflow() %>%
  add_model(lda_mod) %>%
  add_recipe(titanic_recipe)
# 3. quadratic disciminant analysis with MASS engine
qda_mod <- discrim_quad() %>%
  set_mode("classification") %>%
  set_engine("MASS")
qda_wkflow <- workflow() %>%
  add_model(qda_mod) %>%
  add_recipe(titanic_recipe)

## QUESTION 5
# fit each of the models
# 1. logistic regression
log_fit <- 
  log_wkflow %>% 
  fit_resamples(titanic_folds)
log_fit

# 2. linear disriminant
lda_fit <- 
  lda_wkflow %>% 
  fit_resamples(titanic_folds)
lda_fit
# 3.quadratic discriminant 
qda_fit <- 
  qda_wkflow %>% 
  fit_resamples(titanic_folds)
qda_fit

## QUESTION 6
# print mean and standard error across all folds 
collect_metrics(log_fit, metrics = "accuracy")
collect_metrics(lda_fit)
collect_metrics(qda_fit)

## QUESTION 7
qda_fit_train <- fit(qda_wkflow, titanic_train)
qda_fit_train

## QUESTION 8
qda_test <- fit(qda_wkflow, titanic_test)
predict(qda_test, new_data = titanic_test, type = "class") %>% 
  bind_cols(titanic_test %>% select(survived)) %>% 
  accuracy(truth = survived, estimate = .pred_class)



