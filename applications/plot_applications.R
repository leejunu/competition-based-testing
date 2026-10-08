#!/usr/bin/env Rscript
# Render bundled application metrics, or fresh analysis outputs selected with --input.
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[[1]])
root <- dirname(dirname(normalizePath(script)))
source(file.path(root, "plotting", "io.R"))
args <- plot_options(list(application = "utopia", input = "", output = file.path(root, "plots")))
if (!args$application %in% c("utopia", "crispr")) stop("--application must be utopia or crispr")
if (!nzchar(args$input)) {
  args$input <- file.path(root, "results", "applications", args$application)
}
source(file.path(root, "plotting", "style.R"))
source(file.path(root, "plotting", paste0(args$application, ".R")))

check_curves <- function(data, methods, facets, facet_column, alphas) {
  require_columns(data, c("method", "alpha", facet_column))
  if (!setequal(data$method, methods) || !setequal(data[[facet_column]], facets) ||
      anyNA(data[c("method", "alpha", facet_column)]) ||
      anyDuplicated(data[c("method", "alpha", facet_column)])) stop("Invalid application curve keys")
  for (facet in facets) for (method in methods) {
    selected <- data[data[[facet_column]] == facet & data$method == method, ]
    if (!isTRUE(all.equal(sort(selected$alpha), alphas, tolerance = 1e-12))) {
      stop("Incomplete alpha grid for ", facet, ": ", method)
    }
  }
}

if (args$application == "utopia") {
  data <- do.call(rbind, lapply(c(1, 2, 4), function(scale) {
    read.csv(file.path(args$input, paste0("scale", scale, "_curves.csv")))
  }))
  require_columns(data, c("scale", "method", "alpha", "discoveries", "true", "fdp"))
  check_curves(data, utopia_method_levels, c(1, 2, 4), "scale", seq(0.005, 0.5, 0.005))
  data$num_rejections <- data$discoveries
  data$true_rejections <- data$true
  plot <- build_utopia_plot(make_utopia_long(data))
  save_official_plot(plot, args$output, "utopia", width = 6, height = 3.5)
} else {
  for (task in c("unrestricted", "global_null")) {
    data <- read.csv(file.path(args$input, paste0(task, "_curves.csv")))
    require_columns(data, c("task", "method", "alpha", "num_rejections",
                            "known_true_rejections", "kFDR", "global_null_fdp"))
    data$method[data$method == "Dempster (global)"] <- dempster_global_label
    data$method[data$method == "Dempster (within-cell)"] <- dempster_within_label
    check_curves(data, crispr_method_levels, task, "task", seq(0.005, 0.2, 0.005))
    crispr_metric_labels[["Unrestricted: kFDR"]] <- if (task == "global_null") "FDP" else "Labeled FDP"
    plot <- build_crispr_plot(make_crispr_long(data), legend_nrow = 1,
                              legend_height = 0.55, legend_box = "vertical")
    save_official_plot(plot, args$output, paste0("crispr_", task), width = 6, height = 2.2)
  }
}
message("Wrote application figures to ", args$output)
