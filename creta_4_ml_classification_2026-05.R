###############################################################################################
###                          Creta 2026   - Dr. Diac & Co                                  ###
###############################################################################################
###          4. Classification Models - Building, Tuning & Test Set Evaluation             ###
###############################################################################################
# last update: 2026-05-24

# Tidymodels ecosystem reference:  https://www.tidymodels.org/start/
# Tidy Modeling with R (book):     https://www.tmwr.org

options(scipen = 999)

library(tidyverse)
library(tidymodels)
#install.packages('discrim')
library(discrim)
library(future)        # parallel processing
plan("multisession")

library(glmnet)
library(MASS)
library(discrim)
#install.packages('mda')
library(mda)
#install.packages('sparsediscrim')
library(sparsediscrim)

library(ranger)        # Random Forest engine
library(xgboost)       # XGBoost engine
library(kernlab)       # SVM (kernlab) engine
library(nnet)          # MLP engine

library(scales)
library(patchwork)
library(viridis)
library(ggsci)
library(svglite)
library(vip)
library(tinytable)


###############################################################################################
###                                     Set paths                                          ###
###############################################################################################

base_dir <- '/Users/marinfotache/Library/CloudStorage/Dropbox/proiecte_r_in_lucru/2026-04_DrDiac_Creta__R'

setwd(paste(base_dir, 'datasets', sep = '/'))


###############################################################################################
###                                  Load & prepare data                                   ###
###############################################################################################

load(file = 'ml_classification_2026-04-24.RData')

glimpse(df_classif)

# Encode sex as a binary factor: 0 = Male, 1 = Female
df <- df_classif |>
  mutate(sex = case_when(
    sex == 'Female' ~ 1,
    sex == 'Male'   ~ 0,
    .default = NA_real_
  )) |>
  mutate(sex = as.factor(sex))

glimpse(df)
table(df$sex, useNA = 'ifany')


###############################################################################################
###                               Train / Test split                                       ###
###############################################################################################

set.seed(1234)
splits    <- initial_split(df, prop = 0.75, strata = sex)
train_tbl <- training(splits)
test_tbl  <- testing(splits)

cat('\nTraining set size:', nrow(train_tbl),
    '\nTest set size:    ', nrow(test_tbl), '\n')


###############################################################################################
###                       Cross-validation folds (5 x 5)                                  ###
###############################################################################################

set.seed(1234)
cv_train <- vfold_cv(train_tbl, v = 5, repeats = 5, strata = sex)
cv_train


###############################################################################################
###                               Pre-processing recipe                                    ###
###############################################################################################
# - Convert nominal predictors to dummy variables
# - Impute missing values with 3-nearest-neighbours
# - Remove zero-variance predictors
# - Normalise all numeric predictors (centres & scales; important for SVM & MLP)

recipe_cl <- recipe(sex ~ ., data = train_tbl) |>
  step_dummy(all_nominal_predictors()) |>
  step_impute_knn(all_predictors(), neighbors = 3) |>
  step_zv(all_predictors()) |>
  step_normalize(all_numeric_predictors())

# Quick sanity check
any(is.na(recipe_cl |> prep() |> bake(new_data = NULL)))


###############################################################################################
###                              Model specifications                                      ###
###   Full parsnip model list: https://www.tidymodels.org/find/parsnip/                   ###
###############################################################################################

## 1. Logistic Regression (elastic-net regularisation via glmnet)
logreg_spec <- logistic_reg(
    penalty = tune(),
    mixture = tune()
  ) |>
  set_engine("glmnet")


## 2. Linear Discriminant Analysis via MASS
lda_mass_spec <- discrim_linear(
  mode = "classification",
  engine = "MASS"
)


## 3. Linear discriminant analysis via flexible discriminant analysis
lda_fda_spec <- discrim_linear(penalty = tune()) |>
  set_engine("mda") 


## 4. Linear discriminant analysis via regularization
lda_r_diagonal_spec <- discrim_linear(regularization_method = "diagonal") |>
  set_engine("sparsediscrim") 
# lda_r_min_distance_spec <- discrim_linear(regularization_method = "min_distance") |>
#   set_engine("sparsediscrim") 
lda_r_shrink_mean_spec <- discrim_linear(regularization_method = "shrink_mean") |>
  set_engine("sparsediscrim") 
lda_r_shrink_cov_spec <- discrim_linear(regularization_method = "shrink_cov") |>
  set_engine("sparsediscrim") 


## 5. Random Forest (via ranger)
rf_spec <- rand_forest(
    mtry  = tune(),
    trees = 700,
    min_n = tune()
  ) |>
  set_engine("ranger", importance = "permutation") |>
  set_mode("classification")


## 6. Extreme Gradient Boosting (via xgboost)
xgb_spec <- boost_tree(
    trees          = 1000,
    tree_depth     = tune(),
    min_n          = tune(),
    loss_reduction = tune(),     # model complexity (gamma)
    sample_size    = tune(),     # row sub-sampling
    mtry           = tune(),     # column sub-sampling
    learn_rate     = tune()      # eta
  ) |>
  set_engine("xgboost") |>
  set_mode("classification")


## 7. Radial Basis Function SVM (via kernlab)
svm_rbf_spec <- svm_rbf(
    cost      = tune(),
    rbf_sigma = tune(),
    margin    = tune()
  ) |>
  set_mode("classification") |>
  set_engine("kernlab", scaled = FALSE)   # recipe already normalises


## 8. Multi-Layer Perceptron (via nnet)
mlp_spec <- mlp(
    hidden_units = tune(),
    penalty      = tune(),
    epochs       = tune()
  ) |>
  set_engine("nnet") |>
  set_mode("classification")



###############################################################################################
###                                  Assemble workflows                                    ###
###############################################################################################

wf_logreg <- workflow() |> add_model(logreg_spec) |> add_recipe(recipe_cl)
wf_lda_mass <- workflow() |> add_model(lda_mass_spec) |> add_recipe(recipe_cl)
wf_lda_fda <- workflow() |> add_model(lda_fda_spec) |> add_recipe(recipe_cl)
wf_lda_r_diagonal <- workflow() |> add_model(lda_r_diagonal_spec) |> add_recipe(recipe_cl)
# wf_lda_r_min_distance <- workflow() |> add_model(lda_r_min_distance_spec) |> add_recipe(recipe_cl)
wf_lda_r_shrink_mean <- workflow() |> add_model(lda_r_shrink_mean_spec) |> add_recipe(recipe_cl)
wf_lda_r_shrink_cov <- workflow() |> add_model(lda_r_shrink_cov_spec) |> add_recipe(recipe_cl)
wf_rf     <- workflow() |> add_model(rf_spec)     |> add_recipe(recipe_cl)
wf_xgb    <- workflow() |> add_model(xgb_spec)    |> add_recipe(recipe_cl)
wf_svm    <- workflow() |> add_model(svm_rbf_spec) |> add_recipe(recipe_cl)
wf_mlp    <- workflow() |> add_model(mlp_spec)    |> add_recipe(recipe_cl)



###############################################################################################
###                       Hyper-parameter tuning grids (random search)                    ###
###############################################################################################

set.seed(1234)
logreg_grid <- grid_random(
  extract_parameter_set_dials(logreg_spec),
  size = 200
)


set.seed(1234)
lda_fda_grid <- grid_random(
  extract_parameter_set_dials(lda_fda_spec),
  size = 200
)


set.seed(1234)
rf_grid <- grid_random(
  extract_parameter_set_dials(rf_spec) |>
    finalize(train_tbl[, 2:8]), 
  size = 200
)

glimpse(train_tbl)


set.seed(1234)
xgb_grid <- grid_random(
  extract_parameter_set_dials(xgb_spec) |>
    finalize(train_tbl[, 2:8]), 
  size = 600
)


set.seed(1234)
svm_grid <- grid_random(
  extract_parameter_set_dials(svm_rbf_spec),
  size = 200
)


set.seed(1234)
mlp_grid <- grid_random(
  extract_parameter_set_dials(mlp_spec),
  size = 200
)



###############################################################################################
###              Fit models across all CV folds × hyper-parameter grid rows               ###
###############################################################################################

set.seed(1234)
logreg_resamples <- wf_logreg |>
  tune_grid(resamples = cv_train, grid = logreg_grid,
            metrics = metric_set(roc_auc, accuracy))


set.seed(1234)
lda_mass_resamples <- wf_lda_mass |>
  fit_resamples(resamples = cv_train, 
                control = control_resamples(save_pred = TRUE))


set.seed(1234)
lda_fda_resamples <- wf_lda_fda |>
  tune_grid(resamples = cv_train, grid = lda_fda_grid,
            metrics = metric_set(roc_auc, accuracy))


set.seed(1234)
lda_r_diagonal_resamples <- wf_lda_r_diagonal |>
  fit_resamples(resamples = cv_train, 
                control = control_resamples(save_pred = TRUE))

set.seed(1234)
lda_r_shrink_mean_resamples <- wf_lda_r_shrink_mean |>
  fit_resamples(resamples = cv_train, 
                control = control_resamples(save_pred = TRUE))

set.seed(1234)
lda_r_shrink_cov_resamples <- wf_lda_r_shrink_cov |>
  fit_resamples(resamples = cv_train, 
                control = control_resamples(save_pred = TRUE))


set.seed(1234)
rf_resamples <- wf_rf |>
  tune_grid(resamples = cv_train, grid = rf_grid,
            metrics = metric_set(roc_auc, accuracy))


set.seed(1234)
xgb_resamples <- wf_xgb |>
  tune_grid(resamples = cv_train, grid = xgb_grid,
            metrics = metric_set(roc_auc, accuracy))


set.seed(1234)
svm_resamples <- wf_svm |>
  tune_grid(resamples = cv_train, grid = svm_grid,
            metrics = metric_set(roc_auc, accuracy))


set.seed(1234)
mlp_resamples <- wf_mlp |>
  tune_grid(resamples = cv_train, grid = mlp_grid,
            metrics = metric_set(roc_auc, accuracy))


###############################################################################################
###                    Inspect CV results & select best hyperparameters                    ###
###############################################################################################

## --- Visualise tuning results ---
autoplot(logreg_resamples) + ggtitle("Logistic Regression tuning")
autoplot(lda_fda_resamples) + ggtitle("Linear Discriminant Analysis (via Flexible Discriminant Analysis tuning)")
autoplot(rf_resamples)     + ggtitle("Random Forest tuning")
autoplot(xgb_resamples)    + ggtitle("XGBoost tuning")
autoplot(svm_resamples)    + ggtitle("SVM-RBF tuning")
autoplot(mlp_resamples)    + ggtitle("MLP tuning")


logreg_resamples |> collect_metrics(summarize = FALSE)
logreg_resamples |> collect_metrics(summarize = TRUE)
logreg_resamples |> collect_metrics(summarize = FALSE) |>
  group_by(`.metric`) |> summarise(mean_roc_auc_train = mean(`.estimate`, na.rm = TRUE))

lda_mass_resamples |> collect_metrics(summarize = FALSE)
lda_mass_resamples |> collect_metrics(summarize = TRUE)
lda_mass_resamples |> collect_metrics(summarize = FALSE) |>
  group_by(`.metric`) |> summarise(mean_roc_auc_train = mean(`.estimate`, na.rm = TRUE))

lda_fda_resamples |> collect_metrics(summarize = FALSE)
lda_fda_resamples |> collect_metrics(summarize = TRUE)
lda_fda_resamples |> collect_metrics(summarize = FALSE) |>
  group_by(`.metric`) |> summarise(mean_roc_auc_train = mean(`.estimate`, na.rm = TRUE))

rf_resamples |> collect_metrics(summarize = FALSE)
rf_resamples |> collect_metrics(summarize = TRUE)
rf_resamples |> collect_metrics(summarize = FALSE) |>
  group_by(`.metric`) |> summarise(mean_roc_auc_train = mean(`.estimate`, na.rm = TRUE))

xgb_resamples |> collect_metrics(summarize = FALSE)
xgb_resamples |> collect_metrics(summarize = TRUE)
xgb_resamples |> collect_metrics(summarize = FALSE) |>
  group_by(`.metric`) |> summarise(mean_roc_auc_train = mean(`.estimate`, na.rm = TRUE))

svm_resamples |> collect_metrics(summarize = FALSE)
svm_resamples |> collect_metrics(summarize = TRUE)
svm_resamples |> collect_metrics(summarize = FALSE) |>
  group_by(`.metric`) |> summarise(mean_roc_auc_train = mean(`.estimate`, na.rm = TRUE))

mlp_resamples |> collect_metrics(summarize = FALSE)
mlp_resamples |> collect_metrics(summarize = TRUE)
mlp_resamples |> collect_metrics(summarize = FALSE) |>
  group_by(`.metric`) |> summarise(mean_roc_auc_train = mean(`.estimate`, na.rm = TRUE))




## --- Select best hyperparameters by ROC AUC ---
best_logreg <- logreg_resamples |> select_best(metric = "roc_auc")
best_lda_fda <- lda_fda_resamples |> select_best(metric = "roc_auc")
best_rf     <- rf_resamples     |> select_best(metric = "roc_auc")
best_xgb    <- xgb_resamples    |> select_best(metric = "roc_auc")
best_svm    <- svm_resamples    |> select_best(metric = "roc_auc")
best_mlp    <- mlp_resamples    |> select_best(metric = "roc_auc")


## --- CV performance of the best configuration for each model ---
cv_summary <- bind_rows(
  logreg_resamples |> show_best(metric = "roc_auc", n = 1) |>
    mutate(algorithm = "Logistic Regression (glmnet)"),
  lda_fda_resamples |> show_best(metric = "roc_auc", n = 1) |>
    mutate(algorithm = "Linear Discriminant Analysis (via FDA)"),
  rf_resamples     |> show_best(metric = "roc_auc", n = 1) |>
    mutate(algorithm = "Random Forest (ranger)"),
  xgb_resamples    |> show_best(metric = "roc_auc", n = 1) |>
    mutate(algorithm = "XGBoost"),
  svm_resamples    |> show_best(metric = "roc_auc", n = 1) |>
    mutate(algorithm = "SVM - RBF (kernlab)"),
  mlp_resamples    |> show_best(metric = "roc_auc", n = 1) |>
    mutate(algorithm = "MLP (nnet)")
    ) 
#  select(algorithm, mean, std_err, n)

glimpse(cv_summary)

cat("\n--- Best ROC AUC per model (cross-validation) ---\n")
print(cv_summary)

## --- Identify the overall best model ---
best_cv_model <- cv_summary |> slice_max(mean, n = 1)
cat("\nBest model on CV:", best_cv_model$algorithm,
    "  ROC AUC =", round(best_cv_model$mean, 4), "\n")


###############################################################################################
###                    Finalize workflows with the best hyperparameters                    ###
###############################################################################################

set.seed(1234)
final_wf_logreg <- wf_logreg |> finalize_workflow(best_logreg)
final_wf_lda_mass <- wf_lda_mass |> fit(data = train_tbl) 
final_wf_lda_fda <- wf_lda_fda |> finalize_workflow(best_lda_fda)
final_wf_lda_r_diagonal <- wf_lda_r_diagonal |> fit(data = train_tbl) 
final_wf_lda_r_shrink_mean <- wf_lda_r_shrink_mean |> fit(data = train_tbl) 
final_wf_lda_r_shrink_cov <- wf_lda_r_shrink_cov |> fit(data = train_tbl) 
final_wf_rf     <- wf_rf     |> finalize_workflow(best_rf)
final_wf_xgb    <- wf_xgb    |> finalize_workflow(best_xgb)
final_wf_svm    <- wf_svm    |> finalize_workflow(best_svm)
final_wf_mlp    <- wf_mlp    |> finalize_workflow(best_mlp)


average_performance_train <- bind_rows(
  logreg_resamples |> 
    collect_metrics(summarize = FALSE) |>
    inner_join(best_logreg) |>
    rename(metric = .metric, value = .estimate) |>
    group_by(algorithm = 'logreg', metric) |>
    summarise(average_across_cv_folds = mean(value, na.rm = TRUE)),
  lda_mass_resamples |> 
    collect_metrics(summarize = FALSE) |>
    rename(metric = .metric, value = .estimate) |>
    group_by(algorithm = 'lda_mass', metric) |>
    summarise(average_across_cv_folds = mean(value, na.rm = TRUE)) |>
    filter (metric != 'brier_class'),
  lda_fda_resamples |> 
    collect_metrics(summarize = FALSE) |>
    inner_join(best_lda_fda) |>
    rename(metric = .metric, value = .estimate) |>
    group_by(algorithm = 'lda_fda', metric) |>
    summarise(average_across_cv_folds = mean(value, na.rm = TRUE)),
  lda_r_diagonal_resamples |> 
    collect_metrics(summarize = FALSE) |>
    rename(metric = .metric, value = .estimate) |>
    group_by(algorithm = 'lda_r_diagonal', metric) |>
    summarise(average_across_cv_folds = mean(value, na.rm = TRUE)) |>
    filter (metric != 'brier_class'),
  lda_r_shrink_mean_resamples |> 
    collect_metrics(summarize = FALSE) |>
    rename(metric = .metric, value = .estimate) |>
    group_by(algorithm = 'lda_r_shrink_mean', metric) |>
    summarise(average_across_cv_folds = mean(value, na.rm = TRUE)) |>
    filter (metric != 'brier_class'),
  lda_r_shrink_cov_resamples |> 
    collect_metrics(summarize = FALSE) |>
    rename(metric = .metric, value = .estimate) |>
    group_by(algorithm = 'lda_r_shrink_cov', metric) |>
    summarise(average_across_cv_folds = mean(value, na.rm = TRUE)) |>
    filter (metric != 'brier_class'),
  rf_resamples |> 
    collect_metrics(summarize = FALSE) |>
    inner_join(best_rf) |>
    rename(metric = .metric, value = .estimate) |>
    group_by(algorithm = 'RF', metric) |>
    summarise(average_across_cv_folds = mean(value, na.rm = TRUE)),
  xgb_resamples |> 
    collect_metrics(summarize = FALSE) |>
    inner_join(best_xgb) |>
    rename(metric = .metric, value = .estimate) |>
    group_by(algorithm = 'XGB', metric) |>
    summarise(average_across_cv_folds = mean(value, na.rm = TRUE)),
  svm_resamples |> 
    collect_metrics(summarize = FALSE) |>
    inner_join(best_svm) |>
    rename(metric = .metric, value = .estimate) |>
    group_by(algorithm = 'SVM', metric) |>
    summarise(average_across_cv_folds = mean(value, na.rm = TRUE)),
  mlp_resamples |> 
    collect_metrics(summarize = FALSE) |>
    inner_join(best_mlp) |>
    rename(metric = .metric, value = .estimate) |>
    group_by(algorithm = 'MLP', metric) |>
    summarise(average_across_cv_folds = mean(value, na.rm = TRUE))
) |>
  pivot_wider(names_from = metric, values_from = average_across_cv_folds)




###############################################################################################
###   last_fit(): re-fit on full training set, evaluate once on held-out test set          ###
###############################################################################################

set.seed(1234)
test__logreg <- final_wf_logreg |> last_fit(splits)

set.seed(1234)
test__lda_mass <- workflow() |>
  add_recipe(recipe_cl) |>
  add_model(lda_mass_spec) |>
  last_fit(splits) 

set.seed(1234)
test__lda_fda <- final_wf_lda_fda |> last_fit(splits)


set.seed(1234)
test__lda_r_diagonal <- workflow() |>
  add_recipe(recipe_cl) |>
  add_model(lda_r_diagonal_spec) |>
  last_fit(splits) 

set.seed(1234)
test__lda_r_shrink_mean <- workflow() |>
  add_recipe(recipe_cl) |>
  add_model(lda_r_shrink_mean_spec) |>
  last_fit(splits) 

set.seed(1234)
test__lda_r_shrink_cov <- workflow() |>
  add_recipe(recipe_cl) |>
  add_model(lda_r_shrink_cov_spec) |>
  last_fit(splits) 


set.seed(1234)
test__rf  <- final_wf_rf  |> last_fit(splits)

set.seed(1234)
test__xgb <- final_wf_xgb |> last_fit(splits)

set.seed(1234)
test__svm <- final_wf_svm |> last_fit(splits)

set.seed(1234)
test__mlp <- final_wf_mlp |> last_fit(splits)



###############################################################################################
###                       Test-set performance: ROC AUC + Accuracy                        ###
###############################################################################################

summary_test_init <- bind_rows(
  test__logreg |> collect_metrics() |> mutate(algorithm = "Logistic Regression (glmnet)"),
  test__lda_mass |> collect_metrics() |> mutate(algorithm = "Linear Discriminant Analysis (MASS)"),
  test__lda_fda |> collect_metrics() |> mutate(algorithm = "Linear Discriminant Analysis (via FDA)"),
  
  test__lda_r_diagonal |> collect_metrics() |> mutate(algorithm = "Regularized LDA (diagonal)"),
  test__lda_r_shrink_mean |> collect_metrics() |> mutate(algorithm = "Regularized LDA (shrink_mean)"),
  test__lda_r_shrink_cov |> collect_metrics() |> mutate(algorithm = "Regularized LDA (shrink_cov)"),

  test__rf     |> collect_metrics() |> mutate(algorithm = "Random Forest (ranger)"),
  test__xgb    |> collect_metrics() |> mutate(algorithm = "XGBoost"),
  test__svm    |> collect_metrics() |> mutate(algorithm = "SVM - RBF (kernlab)"),
  test__mlp    |> collect_metrics() |> mutate(algorithm = "MLP (nnet)")
) 

summary_test <- summary_test_init |>
  transmute(algorithm, metric = `.metric`, estimate = `.estimate`) |>
  pivot_wider(names_from = metric, values_from = estimate) |>
  arrange(desc(roc_auc))

cat("\n--- Test-set performance (ROC AUC & Accuracy) ---\n")
print(summary_test)

## --- Formatted table (tinytable) ---
setwd(paste0(base_dir, '/figures'))
getwd()
tt_test <- tinytable::tt(summary_test,
    caption = "Test-set performance of tuned classification models") |>
  format_tt(j = 2:3, digits = 3, num_zero = TRUE, num_fmt = "decimal") |>
  style_tt(j = 2:3, align = "r")

tt_test
tt_test |> tinytable::save_tt("ml_classif_test_performance.docx", overwrite = TRUE)
tt_test |> tinytable::save_tt("ml_classif_test_performance.png",  overwrite = TRUE)


###############################################################################################
###                              ROC curves on the test set                                ###
###############################################################################################

make_roc_plot <- function(last_fit_obj, title_str) {
  last_fit_obj |>
    collect_predictions() |>
    roc_curve(sex, .pred_0) |>
    autoplot() +
    ggtitle(title_str) +
    theme_bw(base_size = 11) +
    theme(
      plot.title  = element_text(size = 11, hjust = 0.5),
      legend.text = element_text(size = 10)
    )
}

g1 <- make_roc_plot(test__logreg, "Logistic Regression\n(glmnet)")
g2 <- make_roc_plot(test__rf,     "Random Forest\n(ranger)")
g3 <- make_roc_plot(test__xgb,    "XGBoost")
g4 <- make_roc_plot(test__svm,    "SVM - RBF\n(kernlab)")
g5 <- make_roc_plot(test__mlp,    "Multi-Layer Perceptron\n(nnet)")

roc_panel <- g1 + g2 + g3 + g4 + g5 + plot_layout(nrow = 2)
roc_panel

ggsave("ml_classif_roc_curves_test_set.pdf",
       plot   = roc_panel,
       device = "pdf",
       width  = 25, height = 14, units = "cm")

ggsave("ml_classif_roc_curves_test_set.png",
       plot   = roc_panel,
       device = "png", dpi = 300,
       width  = 25, height = 14, units = "cm")


getwd()
setwd(paste(base_dir, 'datasets', sep = '/'))
#save.image(file = "2026-05-20_ml_classif.RData")

###############################################################################################
###                              Variable Importance                                       ###
###############################################################################################

## Logistic Regression
set.seed(1234)
g_vi1 <- workflow() |>
  add_recipe(recipe_cl) |>
  add_model(logreg_spec |> finalize_model(best_logreg)) |>
  fit(train_tbl) |>
  extract_fit_parsnip() |>
  vip(num_features = 20L) +
  ggtitle("Logistic Regression") +
  theme_bw(base_size = 11) +
  theme(plot.title = element_text(hjust = 0.5))


## MLP
set.seed(1234)
g_vi2 <- workflow() |>
  add_recipe(recipe_cl) |>
  add_model(mlp_spec |> finalize_model(best_mlp)) |>
  fit(train_tbl) |>
  extract_fit_parsnip() |>
  vip(num_features = 20L) +
  ggtitle("Multi-Layer Perceptron") +
  theme_bw(base_size = 11) +
  theme(plot.title = element_text(hjust = 0.5))


## Random Forest
set.seed(1234)
g_vi3 <- workflow() |>
  add_recipe(recipe_cl) |>
  add_model(rf_spec |> finalize_model(best_rf) |>
              set_engine("ranger", importance = "permutation")) |>
  fit(train_tbl) |>
  extract_fit_parsnip() |>
  vip(num_features = 20L) +
  ggtitle("Random Forest") +
  theme_bw(base_size = 11) +
  theme(plot.title = element_text(hjust = 0.5))


## SVM
# set.seed(1234)
# g_vi4 <- workflow() |>
#   add_recipe(recipe_cl) |>
#   add_model(svm_rbf_spec |> finalize_model(best_svm) |>
#             set_engine("kernlab", scaled = FALSE, importance = "permutation") ) |>
#   fit(train_tbl) |>
#   extract_fit_parsnip() |>
#   vip(num_features = 20L) +
#   ggtitle("SVM - RBF") +
#   theme_bw(base_size = 11) +
#   theme(plot.title = element_text(hjust = 0.5))
# 
vi_panel <- g_vi1 + g_vi2 + g_vi3 + plot_layout(nrow = 1)
vi_panel

ggsave("ml_classif_variable_importance.pdf",
       plot = vi_panel, device = "pdf",
       width = 24, height = 12, units = "cm")


###############################################################################################
###                           Save workspace for downstream use                            ###
###   (used by creta_5_iml and creta_6_dalex scripts)                                     ###
###############################################################################################

setwd(paste(base_dir, 'datasets', sep = '/'))
save.image(file = 'ml_classification_2026-04-04.RData')
cat("\nWorkspace saved to ml_classification_2026-04-04.RData\n")
