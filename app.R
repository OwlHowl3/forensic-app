###############################################################################################
###                    Forensic Classification - Shiny Prediction App                      ###
###############################################################################################

library(shiny)
library(bslib)
library(bsicons)
library(tidymodels)
library(discrim)
library(DT)
library(ggplot2)

# Engine packages (must be installed for predict() to work)
library(glmnet)
library(MASS)
library(mda)
library(sparsediscrim)
library(ranger)
library(xgboost)
library(kernlab)
library(nnet)

# --- Discover available datasets ---
get_available_datasets <- function(models_dir = "models") {
  dirs <- list.dirs(models_dir, full.names = TRUE, recursive = FALSE)
  datasets <- list()
  for (d in dirs) {
    meta_path <- file.path(d, "meta.rds")
    if (file.exists(meta_path)) {
      meta <- readRDS(meta_path)
      datasets[[basename(d)]] <- meta
    }
  }
  datasets
}

# --- Theme ---
forensic_theme <- bs_theme(
  version = 5,
  bg = "#f5f6f8",
  fg = "#1f2933",
  primary = "#1f3a5f",
  secondary = "#52606d",
  base_font = font_google("Inter"),
  heading_font = font_google("Inter"),
  font_scale = 0.95
)

# --- UI ---
ui <- page_fillable(
  title = "Forensic ML Classification",
  theme = forensic_theme,
  padding = 0,
  gap = 0,
  fillable = FALSE,

  tags$head(tags$style(HTML("
    :root {
      --fc-bg: #e9edf3;
      --fc-surface: #ffffff;
      --fc-surface-2: #f6f8fb;
      --fc-ink: #0f172a;
      --fc-ink-2: #334155;
      --fc-muted: #64748b;
      --fc-border: #d8dee7;
      --fc-border-strong: #c3cad6;
      --fc-primary: #1f3a5f;
      --fc-primary-tint: #e8eef7;
      --fc-accent: #2563eb;
      --fc-shadow-sm: 0 1px 2px rgba(15,23,42,0.04);
      --fc-shadow:    0 1px 3px rgba(15,23,42,0.08), 0 1px 2px rgba(15,23,42,0.04);
    }
    body { background: var(--fc-bg); color: var(--fc-ink); }

    /* ───── Header ───── */
    .app-header {
      background: #ffffff;
      color: var(--fc-ink);
      padding: 0.35rem 1.25rem;
      border-bottom: 1px solid var(--fc-border);
      box-shadow: 0 1px 0 rgba(15,23,42,0.04);
    }
    .app-header-row {
      display: flex; align-items: center; justify-content: space-between;
      gap: 1rem;
    }
    .header-left, .header-right {
      display: flex; align-items: center;
    }
    .app-header .brand {
      display: inline-flex; align-items: center; gap: 0.45rem;
      margin: 0; line-height: 1;
    }
    .app-header .brand-mark {
      width: 20px; height: 20px; border-radius: 4px;
      background: linear-gradient(135deg, #1f3a5f, #2563eb);
      color: #fff; display: inline-flex; align-items: center; justify-content: center;
      font-size: 0.7rem; box-shadow: inset 0 -1px 0 rgba(0,0,0,0.15);
    }
    .app-header .brand-mark svg { width: 0.7rem; height: 0.7rem; }
    .app-header .brand-text {
      font-size: 0.82rem; font-weight: 700; letter-spacing: 0.03em;
      color: var(--fc-ink);
    }
    .app-header .brand-sub {
      font-size: 0.62rem; color: var(--fc-muted); margin-left: 0.35rem;
      text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;
    }
    .app-header .selector-wrap {
      display: inline-flex; align-items: center; gap: 0.4rem;
      background: var(--fc-bg); padding: 0.1rem 0.1rem 0.1rem 0.55rem;
      border-radius: 5px; border: 1px solid var(--fc-border);
    }
    .app-header .selector-wrap .lab {
      font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.1em;
      color: var(--fc-muted); font-weight: 600;
    }
    .app-header .form-group, .app-header .shiny-input-container { margin-bottom: 0; }
    /* Selectize wrapper — leave room on the right for the caret */
    .app-header .selectize-control { margin: 0; min-height: 0; }
    .app-header .selectize-input {
      min-width: 380px;
      padding: 3px 26px 3px 8px;
      min-height: 0; height: auto;
      font-size: 0.78rem; line-height: 1.35;
      border: 1px solid var(--fc-border); background: #fff;
      box-shadow: none;
    }
    .app-header .selectize-input input { font-size: 0.78rem; line-height: 1.35; height: auto; }
    .app-header .selectize-input::after {
      top: 50% !important; margin-top: -3px !important;
      right: 10px !important;
      border-width: 5px 4px 0 4px !important;
      border-color: var(--fc-muted) transparent transparent transparent !important;
    }
    /* Dropdown options menu */
    .app-header .selectize-dropdown,
    .app-header .selectize-dropdown .option {
      font-size: 0.8rem; line-height: 1.3;
    }
    .app-header .selectize-dropdown .option { padding: 4px 8px; }

    /* ───── Sections ───── */
    .app-section { padding: 0.9rem 1.25rem 0; }
    .app-section:last-of-type { padding-bottom: 1rem; }

    .section-eyebrow {
      display: flex; align-items: center; gap: 0.5rem;
      font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em;
      color: var(--fc-muted); font-weight: 700;
      margin: 0 0 0.55rem 0;
      padding-bottom: 0.4rem;
      border-bottom: 1px solid var(--fc-border);
    }
    .section-eyebrow::before {
      content: ''; display: inline-block; width: 4px; height: 4px;
      background: var(--fc-primary); border-radius: 50%; flex: 0 0 4px;
    }

    /* navset_underline inside cards */
    .card-body .nav.nav-underline {
      border-bottom: 1px solid var(--fc-border);
      margin-bottom: 0.85rem;
      gap: 0.25rem;
    }
    .card-body .nav.nav-underline .nav-link {
      font-size: 0.78rem; font-weight: 600;
      text-transform: uppercase; letter-spacing: 0.06em;
      color: var(--fc-muted);
      padding: 0.45rem 0.75rem;
      border-bottom-width: 2px;
    }
    .card-body .nav.nav-underline .nav-link:hover { color: var(--fc-ink-2); }
    .card-body .nav.nav-underline .nav-link.active {
      color: var(--fc-primary);
      border-bottom-color: var(--fc-primary);
    }

    /* ───── Stats bar ───── */
    .meta-bar {
      display: flex; flex-wrap: wrap; align-items: stretch;
      background: var(--fc-surface);
      border: 1px solid var(--fc-border);
      border-radius: 10px;
      padding: 0;
      box-shadow: var(--fc-shadow);
      overflow: hidden;
    }
    .stat-item {
      display: inline-flex; align-items: center; gap: 0.7rem;
      padding: 0.7rem 1.1rem;
      flex: 1 1 auto; min-width: 0;
      border-right: 1px solid var(--fc-border);
    }
    .stat-item:last-child { border-right: none; }
    .stat-icon {
      width: 32px; height: 32px; flex: 0 0 32px;
      display: inline-flex; align-items: center; justify-content: center;
      background: var(--fc-primary-tint);
      color: var(--fc-primary);
      border-radius: 8px;
      font-size: 1rem;
    }
    .stat-icon svg { width: 1rem; height: 1rem; }
    .stat-text { display: inline-flex; flex-direction: column; min-width: 0; line-height: 1.15; }
    .stat-label {
      color: var(--fc-muted); text-transform: uppercase; letter-spacing: 0.08em;
      font-size: 0.65rem; font-weight: 700;
    }
    .stat-value {
      color: var(--fc-ink); font-weight: 600; font-size: 0.92rem;
      margin-top: 0.1rem;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }

    /* ───── Variables grid ───── */
    .var-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 0.5rem;
    }
    .var-cell {
      display: flex; gap: 0.55rem; align-items: flex-start;
      background: var(--fc-surface-2);
      border: 1px solid transparent;
      border-radius: 6px;
      padding: 0.5rem 0.65rem;
      transition: background .12s, border-color .12s;
    }
    .var-cell:hover {
      background: #eef2f8;
      border-color: var(--fc-border);
    }
    .var-cell .badge-code {
      flex: 0 0 auto;
      background: var(--fc-primary-tint);
      color: var(--fc-primary);
      font-family: 'SF Mono', Menlo, Consolas, monospace;
      font-size: 0.72rem; font-weight: 700;
      padding: 0.15rem 0.45rem;
      border-radius: 4px;
      letter-spacing: 0.02em;
      line-height: 1.3;
    }
    .var-cell .var-body { min-width: 0; }
    .var-cell .name {
      font-weight: 600; font-size: 0.82rem; color: var(--fc-ink);
      line-height: 1.25;
    }
    .var-cell .desc {
      font-size: 0.74rem; color: var(--fc-muted);
      line-height: 1.35; margin-top: 0.1rem;
    }

    /* ───── Cards ───── */
    .card {
      border: 1px solid var(--fc-border);
      border-radius: 10px;
      box-shadow: var(--fc-shadow);
      overflow: hidden;
      background: var(--fc-surface);
    }
    .card-header {
      background: linear-gradient(180deg, #fbfcfe 0%, #f1f4f9 100%);
      border-bottom: 1px solid var(--fc-border);
      font-weight: 700; font-size: 0.78rem; color: var(--fc-ink);
      text-transform: uppercase; letter-spacing: 0.08em;
      padding: 0.55rem 1rem;
      display: flex; align-items: center; gap: 0.5rem;
      position: relative;
    }
    .card-header::before {
      content: ''; display: inline-block;
      width: 3px; height: 14px;
      background: var(--fc-primary); border-radius: 1px;
    }
    .card-body { padding: 1rem; background: var(--fc-surface); }

    /* ───── Inputs ───── */
    .input-form .form-group,
    .input-form .shiny-input-container { margin-bottom: 0 !important; }
    .input-form label {
      font-weight: 600; font-size: 0.8rem; color: var(--fc-ink-2);
      margin-bottom: 0.15rem;
    }
    .input-form .form-control, .input-form .form-select {
      border-color: var(--fc-border); font-size: 0.85rem;
      padding: 0.3rem 0.55rem; min-height: 0; height: auto;
    }
    .input-form .form-control:focus, .input-form .form-select:focus {
      border-color: var(--fc-primary);
      box-shadow: 0 0 0 3px rgba(31,58,95,0.12);
    }
    .input-cell { display: block; }
    /* Tighten layout_column_wrap row gap */
    .input-form .html-fill-container { gap: 0.55rem 0.75rem !important; }

    /* ───── Predict action ───── */
    .predict-action {
      display: flex; justify-content: flex-end;
      margin-top: 0.75rem; padding-top: 0.6rem;
      border-top: 1px solid var(--fc-border);
    }
    .predict-action .btn-primary {
      background: var(--fc-primary); border-color: var(--fc-primary);
      padding: 0.5rem 1.6rem; font-weight: 600; letter-spacing: 0.04em;
      text-transform: uppercase; font-size: 0.82rem;
      border-radius: 6px;
      box-shadow: 0 1px 2px rgba(31,58,95,0.2);
    }
    .predict-action .btn-primary:hover {
      background: #16294a; border-color: #16294a;
    }

    /* ───── Result strip ───── */
    .result-strip {
      margin-top: 1rem;
      background: linear-gradient(180deg, #f8fafc 0%, #eef2f8 100%);
      border: 1px solid var(--fc-border-strong);
      border-radius: 8px;
      padding: 0.85rem 1rem;
      position: relative;
    }
    .result-strip::before {
      content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
      background: var(--fc-primary); border-radius: 8px 0 0 8px;
    }
    .result-strip .strip-label {
      display: inline-flex; align-items: center; gap: 0.4rem;
      font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;
      color: var(--fc-primary); font-weight: 700; margin-bottom: 0.6rem;
    }
    .result-strip .strip-label::before {
      content: ''; width: 6px; height: 6px; background: var(--fc-primary);
      border-radius: 50%;
    }

    /* ───── Footer ───── */
    .app-footer {
      padding: 1rem 1.5rem;
      margin-top: 0.5rem;
      border-top: 1px solid var(--fc-border);
      color: var(--fc-muted); font-size: 0.76rem;
      background: var(--fc-surface);
      text-align: center;
      letter-spacing: 0.02em;
    }
  "))),

  # === Section 1: Header bar ===
  div(
    class = "app-header",
    div(
      class = "app-header-row",
      div(
        class = "header-left",
        h1(
          class = "brand",
          span(class = "brand-mark", bs_icon("activity")),
          span(class = "brand-text", "Forensic ML Classification"),
          span(class = "brand-sub", "Skeletal Sex Estimation")
        )
      ),
      div(
        class = "header-right",
        uiOutput("dataset_selector")
      )
    )
  ),

  # === Section 2: Dataset overview ===
  div(
    class = "app-section",
    uiOutput("dataset_overview")
  ),

  # === Section 3: Prediction input ===
  div(
    class = "app-section",
    card(
      card_header("Enter measurements"),
      card_body(
        class = "input-form",
        uiOutput("input_form"),
        div(
          class = "predict-action",
          actionButton("predict_btn", "Run prediction", class = "btn-primary")
        ),
        uiOutput("prediction_result")
      )
    )
  ),

  # === Section 4: Model performance ===
  div(
    class = "app-section",
    card(
      card_header("Model reference (test set)"),
      card_body(
        navset_underline(
          nav_panel(
            "Metrics",
            DTOutput("metrics_test_table"),
            br(),
            div(class = "section-eyebrow", "Cross-validation (training)"),
            DTOutput("metrics_train_table")
          ),
          nav_panel(
            "Plots",
            div(class = "section-eyebrow", "ROC curves"),
            plotOutput("roc_plot", height = "450px"),
            br(),
            div(class = "section-eyebrow", "Model comparison"),
            plotOutput("metrics_plot", height = "400px")
          )
        )
      )
    )
  ),

  # === Footer ===
  div(
    class = "app-footer",
    "Forensic ML Classification · trained on CRETA skeletal measurements · Built with R, tidymodels & Shiny"
  )
)

# --- Server ---
server <- function(input, output, session) {

  # Load all available datasets
  datasets <- reactiveVal(get_available_datasets())

  # Selected dataset metadata
  selected_meta <- reactive({
    req(input$dataset_choice)
    datasets()[[input$dataset_choice]]
  })

  # Dataset selector (inline in header) ----------------------------------------
  output$dataset_selector <- renderUI({
    ds <- datasets()
    if (length(ds) == 0) {
      return(span(style = "color:var(--fc-muted);",
                  "No datasets found in models/ — run extract_models.R"))
    }
    choices <- setNames(names(ds), sapply(ds, function(x) x$dataset_name))
    div(
      class = "selector-wrap",
      span(class = "lab", "Dataset"),
      selectInput(
        "dataset_choice",
        label = NULL,
        choices = choices,
        width = "400px"
      )
    )
  })

  # Dataset overview: stats strip + variables grid -----------------------------
  output$dataset_overview <- renderUI({
    meta <- selected_meta()
    req(meta)

    target_levels_str <- paste(
      paste(names(meta$target$levels), meta$target$levels, sep = " = "),
      collapse = ", "
    )

    stat_item <- function(icon, label, value) {
      div(
        class = "stat-item",
        span(class = "stat-icon", bs_icon(icon)),
        div(
          class = "stat-text",
          span(class = "stat-label", label),
          span(class = "stat-value", value)
        )
      )
    }

    stats_bar <- div(
      class = "meta-bar",
      stat_item("database",  "Dataset",    meta$dataset_name),
      stat_item("bullseye",  "Target",     paste0(meta$target$variable, " (", target_levels_str, ")")),
      stat_item("rulers",    "Predictors", length(meta$predictors)),
      stat_item("diagram-3", "Models",     length(meta$models))
    )

    var_cells <- lapply(names(meta$predictors), function(var_name) {
      info <- meta$predictors[[var_name]]
      friendly <- if (!is.null(info$label) && nzchar(info$label) && info$label != var_name) {
        info$label
      } else {
        var_name
      }
      description <- if (!is.null(info$description) && nzchar(info$description)) {
        info$description
      } else {
        NULL
      }
      div(
        class = "var-cell",
        span(class = "badge-code", var_name),
        div(
          class = "var-body",
          div(class = "name", friendly),
          if (!is.null(description)) div(class = "desc", description)
        )
      )
    })

    tagList(
      stats_bar,
      div(
        style = "margin-top: 1rem;",
        card(
          card_header("Variables"),
          card_body(
            div(class = "var-grid", var_cells)
          )
        )
      )
    )
  })

  # Dynamic input form ---------------------------------------------------------
  output$input_form <- renderUI({
    meta <- selected_meta()
    req(meta)

    inputs <- lapply(names(meta$predictors), function(var_name) {
      info <- meta$predictors[[var_name]]

      input_tag <- if (info$type == "factor") {
        selectInput(
          inputId = paste0("pred_", var_name),
          label = info$label,
          choices = c("Select..." = "", info$levels),
          selected = "",
          width = "100%"
        )
      } else {
        numericInput(
          inputId = paste0("pred_", var_name),
          label = info$label,
          value = NA,
          step = 0.01,
          width = "100%"
        )
      }

      div(class = "input-cell", input_tag)
    })

    layout_column_wrap(width = 1/3, gap = "0.55rem", !!!inputs)
  })

  # Run predictions ------------------------------------------------------------
  prediction_results <- eventReactive(input$predict_btn, {
    meta <- selected_meta()
    req(meta)

    # Build input tibble from form values
    new_data <- tibble::tibble(.rows = 1)
    for (var_name in names(meta$predictors)) {
      info <- meta$predictors[[var_name]]
      val <- input[[paste0("pred_", var_name)]]

      if (is.null(val)) next
      if (info$type == "numeric" && (is.na(val) || val == "")) next
      if (info$type == "factor" && val == "") next

      if (info$type == "factor") {
        new_data[[var_name]] <- factor(val, levels = info$levels)
      } else {
        new_data[[var_name]] <- as.numeric(val)
      }
    }

    if (ncol(new_data) == 0) {
      showNotification("Please enter at least one value.", type = "warning")
      return(NULL)
    }

    # Load models and predict
    results <- list()
    dataset_dir <- file.path("models", input$dataset_choice)

    for (model_id in names(meta$models)) {
      model_info <- meta$models[[model_id]]
      model_path <- file.path(dataset_dir, model_info$file)

      tryCatch({
        wf <- readRDS(model_path)
        pred_class <- predict(wf, new_data)
        pred_prob <- predict(wf, new_data, type = "prob")

        pred_label <- meta$target$levels[as.character(pred_class$.pred_class)]

        results[[model_id]] <- tibble(
          Model = model_info$display_name,
          Prediction = pred_label,
          `P(Male)` = pred_prob$.pred_0,
          `P(Female)` = pred_prob$.pred_1
        )
      }, error = function(e) {
        results[[model_id]] <<- tibble(
          Model = model_info$display_name,
          Prediction = paste("Error:", e$message),
          `P(Male)` = NA_real_,
          `P(Female)` = NA_real_
        )
      })
    }

    bind_rows(results) |> arrange(desc(`P(Female)`))
  })

  # Prediction result strip (inside the prediction card) -----------------------
  output$prediction_result <- renderUI({
    df <- prediction_results()
    req(df)
    div(
      class = "result-strip",
      div(class = "strip-label", "Predictions"),
      DTOutput("results_table")
    )
  })

  output$results_table <- renderDT({
    df <- prediction_results()
    req(df)
    datatable(
      df,
      options = list(dom = "t", pageLength = 20, ordering = TRUE),
      rownames = FALSE
    ) |>
      formatPercentage(c("P(Male)", "P(Female)"), digits = 1)
  })

  # Model performance: metrics tables ------------------------------------------
  output$metrics_test_table <- renderDT({
    meta <- selected_meta()
    req(meta, meta$metrics_test)

    datatable(
      meta$metrics_test,
      options = list(dom = "t", pageLength = 20, ordering = TRUE),
      rownames = FALSE
    ) |>
      formatRound(2:ncol(meta$metrics_test), digits = 3)
  })

  output$metrics_train_table <- renderDT({
    meta <- selected_meta()
    req(meta, meta$metrics_train)

    datatable(
      meta$metrics_train,
      options = list(dom = "t", pageLength = 20, ordering = TRUE),
      rownames = FALSE
    ) |>
      formatRound(2:ncol(meta$metrics_train), digits = 3)
  })

  # Model performance: plots ---------------------------------------------------
  output$roc_plot <- renderPlot({
    meta <- selected_meta()
    req(meta, meta$roc_data)

    ggplot(meta$roc_data, aes(x = 1 - specificity, y = sensitivity, color = model)) +
      geom_line(linewidth = 0.8) +
      geom_abline(lty = 2, color = "grey50") +
      labs(
        x = "1 - Specificity (False Positive Rate)",
        y = "Sensitivity (True Positive Rate)",
        color = "Model"
      ) +
      theme_bw(base_size = 13) +
      theme(legend.position = "bottom", legend.direction = "horizontal")
  })

  output$metrics_plot <- renderPlot({
    meta <- selected_meta()
    req(meta, meta$metrics_test)

    df_plot <- meta$metrics_test |>
      tidyr::pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value")

    ggplot(df_plot, aes(x = reorder(Model, Value), y = Value, fill = Metric)) +
      geom_col(position = "dodge") +
      coord_flip() +
      labs(x = NULL, y = "Score") +
      theme_bw(base_size = 13) +
      theme(legend.position = "top")
  })
}

shinyApp(ui = ui, server = server)
