###############################################################################################
###                    Forensic Classification - Shiny Prediction App                      ###
###############################################################################################

library(shiny)
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

# --- UI ---
ui <- fluidPage(
  titlePanel("Forensic ML Classification"),

  sidebarLayout(
    sidebarPanel(
      width = 3,

      # Dataset selector
      uiOutput("dataset_selector"),
      hr(),

      # Dataset metadata
      uiOutput("dataset_info"),
      hr(),

      # Column descriptions
      uiOutput("column_descriptions")
    ),

    mainPanel(
      width = 9,
      tabsetPanel(
        id = "main_tabs",

        # --- Tab 1: Predictions ---
        tabPanel(
          "Predictions",
          br(),
          h4("Enter measurements"),
          uiOutput("input_form"),
          br(),
          actionButton("predict_btn", "Predict", class = "btn-primary btn-lg"),
          br(), br(),
          DTOutput("results_table")
        ),

        # --- Tab 2: Model Metrics ---
        tabPanel(
          "Model Metrics",
          br(),
          h4("Test Set Performance"),
          DTOutput("metrics_test_table"),
          br(),
          h4("Cross-Validation Performance (Training)"),
          DTOutput("metrics_train_table")
        ),

        # --- Tab 3: Plots ---
        tabPanel(
          "Plots",
          br(),
          h4("ROC Curves (Test Set)"),
          plotOutput("roc_plot", height = "500px"),
          br(),
          h4("Model Comparison"),
          plotOutput("metrics_plot", height = "400px")
        )
      )
    )
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

  # Dataset selector dropdown
  output$dataset_selector <- renderUI({
    ds <- datasets()
    if (length(ds) == 0) {
      return(p("No datasets found in models/ directory. Run extract_models.R first."))
    }
    choices <- setNames(names(ds), sapply(ds, function(x) x$dataset_name))
    selectInput("dataset_choice", "Dataset:", choices = choices)
  })

  # Dataset metadata display
  output$dataset_info <- renderUI({
    meta <- selected_meta()
    req(meta)
    tagList(
      h5("Dataset Info"),
      p(strong("Name:"), meta$dataset_name),
      p(strong("Target:"), meta$target$variable,
        sprintf("(%s)", paste(paste(names(meta$target$levels), meta$target$levels, sep = " = "), collapse = ", "))),
      p(strong("Predictors:"), length(meta$predictors)),
      p(strong("Models:"), length(meta$models))
    )
  })

  # Column descriptions
  output$column_descriptions <- renderUI({
    meta <- selected_meta()
    req(meta)

    desc_items <- lapply(names(meta$predictors), function(var_name) {
      info <- meta$predictors[[var_name]]
      description <- if (!is.null(info$description)) info$description else "No description available"
      tags$li(strong(var_name), " \u2014 ", description)
    })

    tagList(
      h5("Variables"),
      tags$ul(desc_items)
    )
  })

  # Dynamic input form based on metadata
  output$input_form <- renderUI({
    meta <- selected_meta()
    req(meta)

    inputs <- lapply(names(meta$predictors), function(var_name) {
      info <- meta$predictors[[var_name]]

      if (info$type == "factor") {
        selectInput(
          inputId = paste0("pred_", var_name),
          label = info$label,
          choices = c("Select..." = "", info$levels),
          selected = ""
        )
      } else {
        numericInput(
          inputId = paste0("pred_", var_name),
          label = info$label,
          value = NA,
          step = 0.01
        )
      }
    })

    # Arrange inputs in rows of 3
    rows <- split(inputs, ceiling(seq_along(inputs) / 3))
    row_tags <- lapply(rows, function(row_inputs) {
      cols <- lapply(row_inputs, function(inp) column(4, inp))
      do.call(fluidRow, cols)
    })

    tagList(row_tags)
  })

  # --- Tab 1: Run predictions ---
  observeEvent(input$predict_btn, {
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
      return()
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

    results_df <- bind_rows(results) |>
      arrange(desc(`P(Female)`))

    output$results_table <- renderDT({
      datatable(
        results_df,
        options = list(dom = "t", pageLength = 20, ordering = TRUE),
        rownames = FALSE
      ) |>
        formatPercentage(c("P(Male)", "P(Female)"), digits = 1)
    })
  })

  # --- Tab 2: Model Metrics ---
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

  # --- Tab 3: Plots ---
  output$roc_plot <- renderPlot({
    meta <- selected_meta()
    req(meta, meta$roc_data)

    ggplot(meta$roc_data, aes(x = 1 - specificity, y = sensitivity, color = model)) +
      geom_line(linewidth = 0.8) +
      geom_abline(lty = 2, color = "grey50") +
      labs(
        title = "ROC Curves - Test Set",
        x = "1 - Specificity (False Positive Rate)",
        y = "Sensitivity (True Positive Rate)",
        color = "Model"
      ) +
      theme_bw(base_size = 13) +
      theme(legend.position = "bottom", legend.direction = "vertical")
  })

  output$metrics_plot <- renderPlot({
    meta <- selected_meta()
    req(meta, meta$metrics_test)

    df_plot <- meta$metrics_test |>
      tidyr::pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value")

    ggplot(df_plot, aes(x = reorder(Model, Value), y = Value, fill = Metric)) +
      geom_col(position = "dodge") +
      coord_flip() +
      labs(title = "Model Performance Comparison", x = NULL, y = "Score") +
      theme_bw(base_size = 13) +
      theme(legend.position = "top")
  })
}

shinyApp(ui = ui, server = server)
