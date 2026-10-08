# Shared-control curves: color encodes method, shape encodes block estimation.
shared_curve_point_size <- 1.30
shared_legend_point_size <- 1.30
shared_curve_linewidth <- homogeneous_linewidth
shared_block_shapes <- c(
  "Oracle Blocks" = 16,
  "Estimated Blocks" = 17
)

make_shared_long <- function(data) {
  tpr <- data.frame(
    eta = data$eta,
    method_plot = data$method_plot,
    block_plot = data$block_plot,
    design_facet = data$design_facet,
    metric_facet = "TPR",
    estimate = data$tpr_mean,
    ci_low = data$tpr_ci95_low,
    ci_high = data$tpr_ci95_high,
    stringsAsFactors = FALSE
  )

  fdr_facet <- "FDR"
  fdr <- data.frame(
    eta = data$eta,
    method_plot = data$method_plot,
    block_plot = data$block_plot,
    design_facet = data$design_facet,
    metric_facet = fdr_facet,
    estimate = data$fdp_mean,
    ci_low = data$fdp_ci95_low,
    ci_high = data$fdp_ci95_high,
    stringsAsFactors = FALSE
  )

  out <- rbind(tpr, fdr)
  out$method_plot <- factor(
    out$method_plot,
    levels = c("CBT", "Pooled empirical", "Block empirical")
  )
  out$block_plot <- factor(
    out$block_plot,
    levels = c("Oracle Blocks", "Estimated Blocks")
  )
  out$metric_facet <- factor(out$metric_facet, levels = c(fdr_facet, "TPR"))
  out$design_facet <- factor(
    out$design_facet,
    levels = c("N[total]==2000", "N[total]==4000")
  )
  attr(out, "error_facet") <- fdr_facet
  out
}

extract_shared_bottom_legend <- function(plot) {
  grob <- ggplotGrob(plot)
  candidates <- grep("^guide-box", grob$layout$name)
  for (index in candidates) {
    candidate <- grob$grobs[[index]]
    if (!inherits(candidate, "zeroGrob")) {
      return(candidate)
    }
  }
  stop("Could not extract the shared-plot legend")
}

make_shared_total_subtitle <- function(n_total) {
  grid::textGrob(
    bquote(N[plain(total)] == .(n_total)),
    gp = grid::gpar(
      fontfamily = "Times New Roman",
      fontsize = 8.5
    )
  )
}

build_shared_method_legend <- function() {
  method_levels <- c("CBT", "Pooled empirical", "Block empirical")
  dummy <- data.frame(
    x = rep(c(0, 1), times = length(method_levels)),
    y = 0,
    method_plot = factor(
      rep(method_levels, each = 2),
      levels = method_levels
    )
  )
  plot <- ggplot(
    dummy,
    aes(
      x = x,
      y = y,
      colour = method_plot,
      group = method_plot
    )
  ) +
    geom_line(linewidth = shared_curve_linewidth) +
    scale_colour_manual(
      values = official_colors[method_levels],
      breaks = method_levels,
      drop = FALSE,
      guide = guide_legend(
        title = NULL,
        nrow = 1,
        override.aes = list(
          linetype = rep("solid", length(method_levels))
        )
      )
    ) +
    official_theme() +
    theme(legend.position = "bottom")

  extract_shared_bottom_legend(plot)
}

build_shared_block_legend <- function() {
  block_levels <- c("Oracle Blocks", "Estimated Blocks")
  dummy <- data.frame(
    x = seq_along(block_levels),
    y = 0,
    block_plot = factor(
      block_levels,
      levels = block_levels
    )
  )
  plot <- ggplot(
    dummy,
    aes(
      x = x,
      y = y,
      shape = block_plot
    )
  ) +
    geom_point(
      colour = official_nominal_color,
      size = shared_legend_point_size
    ) +
    scale_shape_manual(
      name = NULL,
      values = shared_block_shapes,
      breaks = block_levels,
      drop = FALSE
    ) +
    guides(
      shape = guide_legend(title = NULL, nrow = 1)
    ) +
    official_theme() +
    theme(legend.position = "bottom")

  extract_shared_bottom_legend(plot)
}

build_shared_nominal_legend <- function() {
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

  extract_shared_bottom_legend(plot)
}

build_shared_legend <- function() {
  method_legend <- build_shared_method_legend()
  block_legend <- build_shared_block_legend()
  nominal_legend <- build_shared_nominal_legend()
  method_and_block_legend <- gtable::gtable_matrix(
    name = "shared-method-and-block-legends",
    grobs = matrix(
      list(method_legend, block_legend),
      nrow = 1
    ),
    widths = grid::unit(c(1.25, 1), "null"),
    heights = grid::unit(1, "null")
  )
  gtable::gtable_matrix(
    name = "shared-legends",
    grobs = matrix(
      list(method_and_block_legend, nominal_legend),
      ncol = 1
    ),
    widths = grid::unit(1, "null"),
    heights = grid::unit(c(1, 0.8), "null")
  )
}

build_shared_metric_panel <- function(
    data,
    metric_facet,
    alpha,
    error_upper,
    show_x_axis = FALSE) {
  selected <- data[data$metric_facet == metric_facet, ]
  pooled <- selected[is.na(selected$block_plot), ]
  blocked <- selected[!is.na(selected$block_plot), ]
  block_rand <- blocked[blocked$method_plot == "Block empirical", ]
  cbt <- blocked[blocked$method_plot == "CBT", ]
  method_levels <- levels(data$method_plot)
  y_upper <- if (metric_facet == "TPR") 1 else error_upper

  plot <- ggplot() +
    geom_blank(
      data = data.frame(eta = 2, estimate = c(0, y_upper)),
      aes(x = eta, y = estimate),
      inherit.aes = FALSE
    )

  if (metric_facet != "TPR") {
    plot <- plot +
      geom_hline(
        yintercept = alpha,
        colour = official_nominal_color,
        linetype = "dashed",
        linewidth = 0.4
      )
  }

  plot <- plot +
    geom_ribbon(
      data = pooled,
      aes(
        x = eta,
        ymin = ci_low,
        ymax = ci_high,
        fill = method_plot,
        group = method_plot
      ),
      colour = NA,
      alpha = official_ribbon_alpha,
      show.legend = FALSE
    ) +
    geom_ribbon(
      data = blocked,
      aes(
        x = eta,
        ymin = ci_low,
        ymax = ci_high,
        fill = method_plot,
        group = interaction(method_plot, block_plot)
      ),
      colour = NA,
      alpha = official_ribbon_alpha,
      show.legend = FALSE
    ) +
    geom_line(
      data = pooled,
      aes(
        x = eta,
        y = estimate,
        colour = method_plot,
        group = method_plot
      ),
      linewidth = shared_curve_linewidth
    ) +
    geom_line(
      data = block_rand,
      aes(
        x = eta,
        y = estimate,
        colour = method_plot,
        linetype = block_plot,
        group = interaction(method_plot, block_plot)
      ),
      linewidth = shared_curve_linewidth
    ) +
    geom_point(
      data = block_rand,
      aes(
        x = eta,
        y = estimate,
        colour = method_plot,
        shape = block_plot,
        group = interaction(method_plot, block_plot)
      ),
      size = shared_curve_point_size
    ) +
    geom_line(
      data = cbt,
      aes(
        x = eta,
        y = estimate,
        colour = method_plot,
        linetype = block_plot,
        group = interaction(method_plot, block_plot)
      ),
      linewidth = shared_curve_linewidth
    ) +
    geom_point(
      data = cbt,
      aes(
        x = eta,
        y = estimate,
        colour = method_plot,
        shape = block_plot,
        group = interaction(method_plot, block_plot)
      ),
      size = shared_curve_point_size
    ) +
    scale_x_continuous(
      breaks = 2:6,
      expand = expansion(mult = c(0.025, 0.025))
    ) +
    scale_y_continuous(
      limits = c(0, y_upper),
      expand = expansion(mult = c(0.035, 0))
    ) +
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
    scale_shape_manual(
      values = shared_block_shapes,
      breaks = c("Oracle Blocks", "Estimated Blocks"),
      drop = FALSE
    ) +
    scale_linetype_manual(
      values = c(
        "Oracle Blocks" = "solid",
        "Estimated Blocks" = "dashed"
      ),
      breaks = c("Oracle Blocks", "Estimated Blocks"),
      drop = FALSE
    ) +
    labs(
      x = if (show_x_axis) expression(eta) else NULL,
      y = if (metric_facet == "TPR") "Power" else metric_facet
    ) +
    official_theme() +
    theme(
      legend.position = "none",
      axis.text.x = if (show_x_axis) {
        element_text()
      } else {
        element_blank()
      },
      axis.ticks.x = if (show_x_axis) {
        element_line()
      } else {
        element_blank()
      },
      plot.margin = margin(1, 1, 1, 1)
    )

  plot
}

build_shared_plot <- function(data) {
  measurement_file <- tempfile(fileext = ".pdf")
  grDevices::cairo_pdf(measurement_file, width = 6, height = 2)
  on.exit({
    grDevices::dev.off()
    unlink(measurement_file)
  }, add = TRUE)

  alpha <- 0.1
  error_upper <- max(
    alpha,
    data$estimate[data$metric_facet == "FDR"],
    na.rm = TRUE
  ) + 0.025
  calibration_levels <- levels(data$design_facet)
  if (length(calibration_levels) != 2L) {
    stop("The shared figure requires exactly two calibration sizes")
  }

  panel_specs <- list(
    list(
      calibration = calibration_levels[[1]],
      metric = "FDR",
      show_x_axis = TRUE
    ),
    list(
      calibration = calibration_levels[[1]],
      metric = "TPR",
      show_x_axis = TRUE
    ),
    list(
      calibration = calibration_levels[[2]],
      metric = "FDR",
      show_x_axis = TRUE
    ),
    list(
      calibration = calibration_levels[[2]],
      metric = "TPR",
      show_x_axis = TRUE
    )
  )
  panel_grobs <- lapply(panel_specs, function(spec) {
    panel_data <- data[data$design_facet == spec$calibration, ]
    ggplotGrob(
      build_shared_metric_panel(
        panel_data,
        metric_facet = spec$metric,
        alpha = alpha,
        error_upper = error_upper,
        show_x_axis = spec$show_x_axis
      ) +
        theme(plot.margin = margin(1, 1, 1, 1))
    )
  })
  legend <- build_shared_legend()

  result <- gtable::gtable(
    widths = grid::unit.c(
      grid::unit(1, "null"),
      grid::unit(1, "null"),
      official_group_separator_gap,
      grid::unit(1, "null"),
      grid::unit(1, "null")
    ),
    heights = grid::unit.c(
      grid::unit(0.14, "in"),
      grid::unit(0.05, "in"),
      grid::unit(1, "null"),
      grid::unit(0.48, "in")
    )
  )
  result <- gtable::gtable_add_grob(
    result,
    make_shared_total_subtitle(2000),
    t = 1,
    l = 1,
    r = 2
  )
  result <- gtable::gtable_add_grob(
    result,
    make_shared_total_subtitle(4000),
    t = 1,
    l = 4,
    r = 5
  )
  panel_columns <- c(1, 2, 4, 5)
  for (index in seq_along(panel_grobs)) {
    result <- gtable::gtable_add_grob(
      result,
      panel_grobs[[index]],
      t = 3,
      l = panel_columns[[index]]
    )
  }
  gtable::gtable_add_grob(result, legend, t = 4, l = 1, r = 5)
}

