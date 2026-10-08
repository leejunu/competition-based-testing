# Small shared CLI and input checks; plotting never invokes a simulator.
plot_options <- function(defaults, args = commandArgs(trailingOnly = TRUE)) {
  if (identical(args, "--help")) {
    cat("Options (each takes a value):\n")
    for (key in names(defaults)) cat("  --", key, "  ", defaults[[key]], "\n", sep = "")
    quit(status = 0)
  }
  if (length(args) %% 2L) stop("Options must be --name value pairs; use --help.")
  if (length(args)) for (i in seq(1L, length(args), by = 2L)) {
    key <- sub("^--", "", args[[i]])
    if (!startsWith(args[[i]], "--") || !key %in% names(defaults)) {
      stop("Unknown option: ", args[[i]])
    }
    defaults[[key]] <- args[[i + 1L]]
  }
  defaults
}

require_columns <- function(data, required) {
  missing <- setdiff(required, names(data))
  if (length(missing)) stop("Missing columns: ", paste(missing, collapse = ", "))
}

read_setting <- function(directory, setting, methods) {
  paths <- file.path(directory, paste0(setting, c(".csv.gz", ".csv")))
  paths <- paths[file.exists(paths)]
  if (length(paths) != 1L) stop("Expected one results file for ", setting, " in ", directory)
  data <- read.csv(paths, stringsAsFactors = FALSE)
  keys <- c("setting", "seed", "eta", "method", "procedure")
  metrics <- c("fdp", "power", "fwer", "discoveries")
  require_columns(data, c(keys, metrics))
  if (!nrow(data) || anyNA(data[c(keys, metrics)]) || anyDuplicated(data[keys])) {
    stop("Empty, missing, or duplicate results in ", paths)
  }
  if (!all(data$setting == setting) || !setequal(data$method, methods)) {
    stop("Unexpected setting or methods in ", paths)
  }
  if (any(!is.finite(as.matrix(data[metrics]))) ||
      any(as.matrix(data[c("fdp", "power", "fwer")]) < 0) ||
      any(as.matrix(data[c("fdp", "power", "fwer")]) > 1)) {
    stop("Invalid metric values in ", paths)
  }
  expected <- expand.grid(seed = unique(data$seed), eta = unique(data$eta))
  for (method in methods) {
    procedures <- if (method == "CBT e-values") "eBH" else if (
      startsWith(setting, "shared_") || startsWith(setting, "q_")) "BH" else c("BH", "Bonferroni")
    selected <- data[data$method == method, ]
    if (!setequal(selected$procedure, procedures)) stop("Unexpected procedures for ", method)
    for (procedure in procedures) {
      curve <- selected[selected$procedure == procedure, ]
      if (nrow(curve) != nrow(expected)) stop("Incomplete seed/effect grid for ", method)
    }
  }
  data
}

summarize_results <- function(data) {
  keys <- c("setting", "eta", "method", "procedure")
  groups <- split(data, interaction(data[keys], drop = TRUE))
  rows <- lapply(groups, function(group) {
    out <- group[1, keys]
    out$replicates <- nrow(group)
    for (metric in c("fdp", "power", "fwer", "discoveries")) {
      values <- group[[metric]]
      out[[metric]] <- mean(values)
      out[[paste0(metric, "_se")]] <- if (length(values) > 1) sd(values) / sqrt(length(values)) else 0
    }
    out
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out[do.call(order, unname(as.list(out[keys]))), ]
}

# Preserve the manuscript's 1.96-SE bands; q-finalist panels use two SEs.
manuscript_summary <- function(summary) {
  out <- summary
  mapping <- c(fdr = "fdp", fdp = "fdp", tpr = "power", fwer = "fwer")
  for (metric in names(mapping)) {
    mean <- summary[[mapping[[metric]]]]
    se <- summary[[paste0(mapping[[metric]], "_se")]]
    out[[paste0(metric, "_mean")]] <- mean
    out[[paste0(metric, "_ci95_low")]] <- pmax(0, mean - 1.96 * se)
    out[[paste0(metric, "_ci95_high")]] <- pmin(1, mean + 1.96 * se)
    out[[paste0(metric, "_two_se_low")]] <- pmax(0, mean - 2 * se)
    out[[paste0(metric, "_two_se_high")]] <- pmin(1, mean + 2 * se)
  }
  out$procedure <- ifelse(out$procedure == "Bonferroni", "bonferroni", "bh")
  out
}
