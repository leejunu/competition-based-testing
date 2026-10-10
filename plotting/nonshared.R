# Paired BH/Bonferroni grids used for homogeneous, binary, and smooth sizes.
nonshared_method_levels <- c(
  "CBT", "Equal-weight CBT", "Subsampled CBT", "CBT e-values",
  "Pooled empirical", "Individual empirical", "Compound empirical"
)
nonshared_shapes <- c(
  "CBT" = 16, "Equal-weight CBT" = 15, "Subsampled CBT" = 15,
  "CBT e-values" = 8, "Pooled empirical" = 16,
  "Individual empirical" = 16, "Compound empirical" = 16
)
# Keep marker sizes uniform; the asterisk distinguishes e-BH from CBT circles.
nonshared_point_sizes <- setNames(rep(homogeneous_point_size, length(nonshared_method_levels)),
                                 nonshared_method_levels)

make_nonshared_long <- function(data, procedure, error_metric, alpha) {
  selected <- data[data$procedure == procedure, ]
  if (nrow(selected) == 0) {
    stop(sprintf("No rows found for procedure %s", procedure))
  }

  error_facet <- if (error_metric == "fdr") "FDR" else "FWER"

  tpr <- data.frame(
    eta = selected$eta,
    method_plot = selected$method_plot,
    design_facet = selected$design_facet,
    metric_facet = "TPR",
    estimate = selected$tpr_mean,
    ci_low = selected$tpr_ci95_low,
    ci_high = selected$tpr_ci95_high,
    stringsAsFactors = FALSE
  )

  error <- data.frame(
    eta = selected$eta,
    method_plot = selected$method_plot,
    design_facet = selected$design_facet,
    metric_facet = error_facet,
    estimate = selected[[paste0(error_metric, "_mean")]],
    ci_low = selected[[paste0(error_metric, "_ci95_low")]],
    ci_high = selected[[paste0(error_metric, "_ci95_high")]],
    stringsAsFactors = FALSE
  )

  out <- rbind(tpr, error)
  method_levels <- unname(
    nonshared_method_levels[
      nonshared_method_levels %in% unique(out$method_plot)
    ]
  )
  out$method_plot <- factor(out$method_plot, levels = method_levels)
  out$metric_facet <- factor(out$metric_facet, levels = c(error_facet, "TPR"))
  out$design_facet <- factor(out$design_facet, levels = unique(data$design_facet))
  attr(out, "error_facet") <- error_facet
  out
}

extract_bottom_legend <- function(plot) {
  grob <- ggplotGrob(plot)
  candidates <- grep("^guide-box", grob$layout$name)
  for (index in candidates) {
    candidate <- grob$grobs[[index]]
    if (!inherits(candidate, "zeroGrob")) {
      return(candidate)
    }
  }
  stop("Could not extract the homogeneous-plot legend")
}

make_higher_level_facet_strip <- function(label) {
  grid::grobTree(
    grid::rectGrob(
      gp = grid::gpar(
        fill = official_strip_fill,
        col = official_strip_color,
        lwd = official_strip_grid_lwd
      )
    ),
    grid::textGrob(
      label,
      gp = grid::gpar(
        fontfamily = "Times New Roman",
        fontsize = 9
      )
    )
  )
}

add_higher_level_facet_strip <- function(grob, label) {
  panel_indices <- grep("^panel", grob$layout$name)
  strip_indices <- grep("^strip-t", grob$layout$name)
  if (length(panel_indices) == 0L || length(strip_indices) == 0L) {
    stop("Could not locate the panel and top-strip cells")
  }

  panel_left <- min(grob$layout$l[panel_indices])
  panel_right <- max(grob$layout$r[panel_indices])
  strip_top <- min(grob$layout$t[strip_indices])
  grob <- gtable::gtable_add_rows(
    grob,
    heights = grid::unit(0.18, "in"),
    pos = strip_top - 1L
  )
  gtable::gtable_add_grob(
    grob,
    make_higher_level_facet_strip(label),
    t = strip_top,
    l = panel_left,
    r = panel_right,
    clip = "on",
    name = paste0("higher-level-strip-", tolower(label))
  )
}

build_homogeneous_method_legend <- function(data, nrow = 2L) {
  selected <- data[data$metric_facet == "TPR", ]
  method_levels <- nonshared_method_levels
  method_levels <- method_levels[
    method_levels %in% unique(as.character(selected$method_plot))
  ]
  # Place e-BH immediately to the right of ordinary CBT.
  if ("CBT e-values" %in% method_levels) {
    method_levels <- append(setdiff(method_levels, "CBT e-values"), "CBT e-values",
                            after = match("CBT", setdiff(method_levels, "CBT e-values")))
  }
  compact_single_row <- nrow == 1L && length(method_levels) >= 4L
  legend_text_size <- 9
  legend_key_width <- if (compact_single_row) 0.6 else 1.05
  legend_spacing_x <- if (compact_single_row) 0 else 0.15

  plot <- ggplot(
    selected,
    aes(
      x = eta,
      y = estimate,
      colour = method_plot,
      group = method_plot
    )
  ) +
    geom_line(linewidth = homogeneous_linewidth) +
    geom_point(aes(shape = method_plot, size = method_plot)) +
    scale_shape_manual(values = nonshared_shapes, guide = "none") +
    scale_size_manual(values = nonshared_point_sizes, guide = "none") +
    scale_colour_manual(
      values = official_colors[method_levels],
      breaks = method_levels,
      labels = ifelse(method_levels == "CBT e-values", "CBT (e-BH)", method_levels),
      guide = guide_legend(
        title = NULL,
        nrow = nrow,
        byrow = TRUE,
        override.aes = list(
          linetype = rep("solid", length(method_levels)),
          shape = unname(nonshared_shapes[method_levels]),
          size = unname(nonshared_point_sizes[method_levels])
        )
      )
    ) +
    official_theme() +
    theme(
      legend.position = "bottom",
      legend.text = element_text(size = legend_text_size),
      legend.key.width = grid::unit(legend_key_width, "lines"),
      legend.spacing.x = grid::unit(legend_spacing_x, "lines")
    )

  extract_bottom_legend(plot)
}

build_homogeneous_nominal_legend <- function() {
  nominal <- data.frame(
    x = c(0, 1),
    y = c(0, 0),
    nominal_key = official_nominal_key
  )
  plot <- ggplot(
    nominal,
    aes(
      x = x,
      y = y,
      linewidth = nominal_key,
      group = nominal_key
    )
  ) +
    geom_line(
      colour = official_nominal_color,
      linetype = "dashed"
    ) +
    scale_linewidth_manual(
      values = c("Nominal level" = 0.4),
      breaks = official_nominal_key,
      labels = "Nominal level",
      guide = guide_legend(
        title = NULL,
        nrow = 1,
        override.aes = list(
          colour = official_nominal_color,
          linetype = "dashed"
        )
      )
    ) +
    official_theme() +
    theme(legend.position = "bottom")

  extract_bottom_legend(plot)
}

build_homogeneous_legend <- function(data, method_nrow = 2L) {
  method_legend <- build_homogeneous_method_legend(data, nrow = method_nrow)
  nominal_legend <- build_homogeneous_nominal_legend()
  gtable::gtable_matrix(
    name = "homogeneous-legends",
    grobs = matrix(list(method_legend, nominal_legend), ncol = 1),
    widths = grid::unit(1, "null"),
    heights = grid::unit(c(1, 1), "null")
  )
}

build_heterogeneous_group_plot <- function(
    data,
    alpha,
    column_labels = c("Case~1", "Case~2")) {
  error_facet <- attr(data, "error_facet")
  method_levels <- levels(data$method_plot)
  design_levels <- levels(data$design_facet)
  if (length(design_levels) != 2L) {
    stop("The paired facet grid requires exactly two column designs")
  }
  if (length(column_labels) != 2L) {
    stop("column_labels must contain exactly two parsed labels")
  }

  data$case_facet <- factor(
    as.character(data$design_facet),
    levels = design_levels,
    labels = column_labels
  )
  cbt_data <- data[data$method_plot == "CBT", ]
  error_upper <- max(
    alpha,
    data$estimate[data$metric_facet == error_facet],
    na.rm = TRUE
  ) + 0.015

  limits <- rbind(
    expand.grid(
      metric_facet = "TPR",
      case_facet = levels(data$case_facet),
      y = c(0, 1),
      stringsAsFactors = FALSE
    ),
    expand.grid(
      metric_facet = error_facet,
      case_facet = levels(data$case_facet),
      y = c(0, error_upper),
      stringsAsFactors = FALSE
    )
  )
  limits$metric_facet <- factor(
    limits$metric_facet,
    levels = levels(data$metric_facet)
  )
  limits$case_facet <- factor(
    limits$case_facet,
    levels = levels(data$case_facet)
  )

  nominal <- unique(
    data[
      data$metric_facet == error_facet,
      c("metric_facet", "case_facet")
    ]
  )
  nominal$threshold <- alpha

  ggplot(
    data,
    aes(
      x = eta,
      y = estimate,
      colour = method_plot,
      group = method_plot
    )
  ) +
    geom_blank(
      data = limits,
      aes(x = 2, y = y),
      inherit.aes = FALSE
    ) +
    geom_hline(
      data = nominal,
      aes(yintercept = threshold),
      inherit.aes = FALSE,
      colour = official_nominal_color,
      linetype = "dashed",
      linewidth = 0.4
    ) +
    geom_ribbon(
      aes(ymin = ci_low, ymax = ci_high, fill = method_plot),
      colour = NA,
      alpha = official_ribbon_alpha,
      show.legend = FALSE
    ) +
    geom_line(linewidth = homogeneous_linewidth) +
    geom_point(aes(shape = method_plot, size = method_plot)) +
    scale_shape_manual(values = nonshared_shapes, guide = "none") +
    scale_size_manual(values = nonshared_point_sizes, guide = "none") +
    geom_line(
      data = cbt_data,
      linewidth = homogeneous_linewidth,
      show.legend = FALSE
    ) +
    geom_point(
      data = cbt_data,
      size = homogeneous_point_size,
      show.legend = FALSE
    ) +
    facet_grid(
      metric_facet ~ case_facet,
      scales = "free_y",
      switch = "y",
      labeller = labeller(
        metric_facet = c(
          "TPR" = "Power",
          "FDR" = "FDR",
          "FWER" = "FWER"
        ),
        case_facet = label_parsed
      )
    ) +
    scale_x_continuous(
      breaks = 2:6,
      expand = expansion(mult = c(0.075, 0.075))
    ) +
    scale_y_continuous(expand = expansion(mult = c(0.035, 0))) +
    scale_colour_manual(
      values = official_colors[method_levels],
      breaks = method_levels,
      drop = FALSE
    ) +
    scale_fill_manual(
      values = official_colors[method_levels],
      breaks = method_levels,
      drop = FALSE,
      guide = "none"
    ) +
    labs(
      x = expression(eta),
      y = NULL
    ) +
    official_theme() +
    theme(
      legend.position = "none",
      strip.background.y = element_blank(),
      strip.text.y.left = element_text(
        angle = 90,
        margin = margin(r = 2)
      ),
      panel.spacing.x = grid::unit(0, "pt"),
      panel.spacing.y = grid::unit(0.12, "in"),
      plot.margin = margin(5, 5, 5, 5)
    )
}

build_heterogeneous_overall <- function(
    data,
    bh_alpha = 0.1,
    bonferroni_alpha = 0.2,
    column_labels = c("Case~1", "Case~2"),
    method_legend_nrow = 2L,
    bh_label = "BH",
    include_bonferroni = TRUE) {
  measurement_file <- tempfile(fileext = ".pdf")
  grDevices::cairo_pdf(measurement_file, width = 6, height = 3.1)
  on.exit({
    grDevices::dev.off()
    unlink(measurement_file)
  }, add = TRUE)

  bh_data <- make_nonshared_long(
    data,
    procedure = "bh",
    error_metric = "fdr",
    alpha = bh_alpha
  )
  bonferroni_data <- make_nonshared_long(
    data,
    procedure = "bonferroni",
    error_metric = "fwer",
    alpha = bonferroni_alpha
  )
  bh_grob <- add_higher_level_facet_strip(
    ggplotGrob(
      build_heterogeneous_group_plot(
        bh_data,
        alpha = bh_alpha,
        column_labels = column_labels
      )
    ),
    bh_label
  )
  bonferroni_grob <- add_higher_level_facet_strip(
    ggplotGrob(
      build_heterogeneous_group_plot(
        bonferroni_data,
        alpha = bonferroni_alpha,
        column_labels = column_labels
      )
    ),
    "Bonferroni"
  )
  aligned_heights <- grid::unit.pmax(
    bh_grob$heights,
    bonferroni_grob$heights
  )
  aligned_widths <- grid::unit.pmax(
    bh_grob$widths,
    bonferroni_grob$widths
  )
  bh_grob$heights <- aligned_heights
  bonferroni_grob$heights <- aligned_heights
  bh_grob$widths <- aligned_widths
  bonferroni_grob$widths <- aligned_widths
  if (!include_bonferroni) {
    return(gtable::gtable_matrix(
      name = "bh-only",
      grobs = matrix(list(
        bh_grob,
        build_homogeneous_legend(bh_data, method_nrow = method_legend_nrow)
      ), ncol = 1),
      widths = grid::unit(1, "null"),
      heights = grid::unit.c(grid::unit(1, "null"), grid::unit(0.75, "in"))
    ))
  }
  legend <- build_homogeneous_legend(
    bh_data,
    method_nrow = method_legend_nrow
  )

  result <- gtable::gtable(
    widths = grid::unit.c(
      grid::unit(1, "null"),
      official_group_separator_gap,
      grid::unit(1, "null")
    ),
    heights = grid::unit.c(
      grid::unit(1, "null"),
      grid::unit(0.58, "in")
    )
  )
  result <- gtable::gtable_add_grob(result, bh_grob, t = 1, l = 1)
  result <- gtable::gtable_add_grob(
    result,
    bonferroni_grob,
    t = 1,
    l = 3
  )
  gtable::gtable_add_grob(result, legend, t = 2, l = 1, r = 3)
}
