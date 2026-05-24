###############################################################################################
###        Extract trained models from .RData into individual .rds files for Shiny app     ###
###############################################################################################
# Run this script once after training models to prepare them for the Shiny app.
# It loads the .RData workspace, extracts each fitted workflow, and saves them
# along with metadata about the predictors.

library(tidymodels)
library(discrim)

# --- Configuration ---
rdata_file <- "ml_classification_2026-04-24.RData"
output_dir <- "models/creta"

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# --- Load workspace ---
load(rdata_file)

# --- Extract the training data structure for metadata ---
# The recipe tells us what predictors exist and their roles
prepped_recipe <- recipe_cl |> prep()
var_info <- prepped_recipe$var_info

predictors <- var_info |>

  filter(role == "predictor") |>
  select(variable, type, source)

# Get predictor summary from the original training data
predictor_meta <- list()
for (v in predictors$variable) {
  col <- train_tbl[[v]]
  if (is.factor(col) || is.character(col)) {
    predictor_meta[[v]] <- list(
      type = "factor",
      levels = levels(as.factor(col)),
      label = v
    )
  } else {
    predictor_meta[[v]] <- list(
      type = "numeric",
      min = min(col, na.rm = TRUE),
      max = max(col, na.rm = TRUE),
      mean = mean(col, na.rm = TRUE),
      label = v
    )
  }
}

# --- Extract fitted workflows from last_fit objects ---
# last_fit objects contain the fitted workflow in .workflow[[1]]
models_to_export <- list(
  list(name = "wf_logreg",          object = test__logreg,          display = "Logistic Regression (glmnet)"),
  list(name = "wf_lda_mass",        object = test__lda_mass,        display = "LDA (MASS)"),
  list(name = "wf_lda_fda",         object = test__lda_fda,         display = "LDA (FDA)"),
  list(name = "wf_lda_r_diagonal",  object = test__lda_r_diagonal,  display = "Regularized LDA (diagonal)"),
  list(name = "wf_lda_r_shrink_mean", object = test__lda_r_shrink_mean, display = "Regularized LDA (shrink mean)"),
  list(name = "wf_lda_r_shrink_cov", object = test__lda_r_shrink_cov, display = "Regularized LDA (shrink cov)"),
  list(name = "wf_rf",              object = test__rf,              display = "Random Forest (ranger)"),
  list(name = "wf_xgb",             object = test__xgb,             display = "XGBoost"),
  list(name = "wf_svm",             object = test__svm,             display = "SVM - RBF (kernlab)"),
  list(name = "wf_mlp",             object = test__mlp,             display = "MLP (nnet)")
)

model_list <- list()

for (m in models_to_export) {
  # Extract the fitted workflow from the last_fit object
  fitted_wf <- m$object$.workflow[[1]]

  rds_file <- paste0(m$name, ".rds")
  saveRDS(fitted_wf, file = file.path(output_dir, rds_file))

  model_list[[m$name]] <- list(
    file = rds_file,
    display_name = m$display
  )

  cat("Saved:", rds_file, "\n")
}

# --- Extract test-set metrics ---
metrics_test <- summary_test |>
  rename(Model = algorithm, Accuracy = accuracy, `ROC AUC` = roc_auc, `Brier Score` = brier_class)

metrics_train <- average_performance_train |>
  ungroup() |>
  rename(Model = algorithm, `CV Accuracy` = accuracy, `CV ROC AUC` = roc_auc)

# --- Extract ROC curve data for plots ---
roc_data <- bind_rows(
  test__logreg |> collect_predictions() |> roc_curve(sex, .pred_0) |> mutate(model = "Logistic Regression (glmnet)"),
  test__lda_mass |> collect_predictions() |> roc_curve(sex, .pred_0) |> mutate(model = "LDA (MASS)"),
  test__lda_fda |> collect_predictions() |> roc_curve(sex, .pred_0) |> mutate(model = "LDA (FDA)"),
  test__lda_r_diagonal |> collect_predictions() |> roc_curve(sex, .pred_0) |> mutate(model = "Regularized LDA (diagonal)"),
  test__lda_r_shrink_mean |> collect_predictions() |> roc_curve(sex, .pred_0) |> mutate(model = "Regularized LDA (shrink mean)"),
  test__lda_r_shrink_cov |> collect_predictions() |> roc_curve(sex, .pred_0) |> mutate(model = "Regularized LDA (shrink cov)"),
  test__rf |> collect_predictions() |> roc_curve(sex, .pred_0) |> mutate(model = "Random Forest (ranger)"),
  test__xgb |> collect_predictions() |> roc_curve(sex, .pred_0) |> mutate(model = "XGBoost"),
  test__svm |> collect_predictions() |> roc_curve(sex, .pred_0) |> mutate(model = "SVM - RBF (kernlab)"),
  test__mlp |> collect_predictions() |> roc_curve(sex, .pred_0) |> mutate(model = "MLP (nnet)")
)

# --- Save metadata ---
meta <- list(
  dataset_name = "Creta 2026 - Skeletal Sex Classification",
  target = list(
    variable = "sex",
    levels = c("0" = "Male", "1" = "Female"),
    positive_class = "1"
  ),
  predictors = predictor_meta,
  models = model_list,
  metrics_test = metrics_test,
  metrics_train = metrics_train,
  roc_data = roc_data
)

saveRDS(meta, file = file.path(output_dir, "meta.rds"))
cat("\nMetadata saved to:", file.path(output_dir, "meta.rds"), "\n")
cat("Done! All models exported to", output_dir, "\n")
