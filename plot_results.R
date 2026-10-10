#!/usr/bin/env Rscript
# Render saved results with the manuscript's R figure builders.
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[[1]])
root <- dirname(normalizePath(script))
source(file.path(root, "plotting", "io.R"))
args <- plot_options(list(methods = "supplementary", baseline = file.path(root, "results/published"),
                          supplementary = file.path(root, "results/supplementary"), output = ""))
if (!args$methods %in% c("main", "supplementary")) stop("--methods must be main or supplementary")
supplementary <- args$methods == "supplementary"
if (!nzchar(args$output)) args$output <- file.path(root, "plots")
for (file in c("style", "nonshared", "shared", "q_finalist")) {
  source(file.path(root, "plotting", paste0(file, ".R")))
}

groups <- list(
  homogeneous = c("homogeneous_uniform", "homogeneous_weighted"),
  binary = c("binary_increasing", "binary_decreasing"),
  smooth = c("smooth_increasing", "smooth_decreasing"),
  shared = c("shared_2000", "shared_4000"),
  q_finalist = c("q_n50_m500", "q_n500_m50")
)
figure_stems <- c(
  homogeneous = "nonshared-homogeneous-m500-n25-uniform-weighted-w",
  binary = "nonshared-binary-10-50",
  smooth = "nonshared-smooth-10-50",
  shared = "shared-block-bh",
  q_finalist = "q-finalist-bh-uniform-w"
)
if (supplementary) groups <- groups[1:3]
summaries <- list()
for (group in names(groups)) {
  methods <- switch(group,
    shared = c("CBT (Oracle)", "CBT (Estimated)", "Pooled empirical",
               "Block empirical (Oracle)", "Block empirical (Estimated)"),
    q_finalist = paste0("q=", c(1, 2, 5, 10, 25)),
    c("CBT", "Pooled empirical", "Individual empirical", "Compound empirical"))
  parts <- lapply(groups[[group]], function(setting) {
    baseline <- read_setting(args$baseline, setting, methods)
    if (!supplementary) return(baseline)
    added <- read_setting(args$supplementary, setting,
                          c("Equal-weight CBT", "Subsampled CBT", "CBT e-values"))
    grid_key <- function(data) unique(paste(data$seed, data$eta))
    if (!setequal(grid_key(baseline), grid_key(added))) {
      stop("Baseline and supplementary seed/effect grids differ for ", setting)
    }
    rbind(baseline, added)
  })
  summaries[[group]] <- summarize_results(do.call(rbind, parts))
  data <- manuscript_summary(summaries[[group]])
  if (group == "shared") {
    data$method_plot <- sub(" \\((Oracle|Estimated)\\)$", "", data$method)
    data$block_plot <- ifelse(grepl("Oracle", data$method), "Oracle Blocks",
                             ifelse(grepl("Estimated", data$method), "Estimated Blocks", NA))
    data$design_facet <- ifelse(data$setting == "shared_2000", "N[total]==2000", "N[total]==4000")
    plot <- build_shared_plot(make_shared_long(data))
    height <- 2
  } else if (group == "q_finalist") {
    data$q <- as.integer(sub("q=", "", data$method))
    data$setting <- factor(ifelse(data$setting == "q_n50_m500", "m500_n50", "m50_n500"),
                           levels = c("m500_n50", "m50_n500"))
    plot <- build_q_comparison_overall(data)
    height <- 2.15
  } else {
    # These two methods coincide with CBT for homogeneous control sizes.
    if (group == "homogeneous") {
      data <- data[!data$method %in% c("Equal-weight CBT", "Subsampled CBT"), ]
    }
    data$method_plot <- data$method
    data$design_facet <- factor(data$setting, levels = groups[[group]])
    data <- data[order(data$design_facet, data$eta), ]
    labels <- if (group == "homogeneous") c("plain('Uniform')~italic(K)", "plain('Weighted')~italic(K)") else c("Case~1", "Case~2")
    plot <- build_heterogeneous_overall(data, column_labels = labels,
      method_legend_nrow = if (supplementary && group == "homogeneous") 3L else if (supplementary) 2L else 1L,
      bh_label = if (supplementary) "BH / e-BH" else "BH",
      include_bonferroni = !(supplementary && group == "homogeneous"))
    height <- if (group == "binary") 3.7 else 3.4
  }
  stem <- figure_stems[[group]]
  if (supplementary) stem <- paste0(stem, "-supplementary-methods")
  width <- 6
  if (supplementary && group == "homogeneous") {
    # Preserve one half of the original grid; only the wrapped legend grows.
    width <- (6 - 0.12) / 2
    height <- height + (0.75 - 0.58)
  }
  save_official_plot(plot, args$output, stem, width = width, height = height)
  message("Wrote ", file.path(args$output, paste0(stem, ".{pdf,png}")))
}
write.csv(do.call(rbind, summaries), file.path(args$output, paste0("summary-", args$methods, ".csv")), row.names = FALSE)
