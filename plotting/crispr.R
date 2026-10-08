# Global and within-cell BH comparisons in the manuscript layout.
dempster_within_label <- "Dempster et al. p-values (cell line BH)"
dempster_global_label <- "Dempster et al. p-values (global BH)"

# Normalize any legacy metric-table keys to the paper terminology.
# Shared paper convention: GG first in blue/solid, ARC second in green/dashed.
crispr_method_levels <- c(
  "CBT (GG)",
  "CBT (ARC)",
  dempster_within_label,
  dempster_global_label
)

crispr_method_labels <- c(
  "CBT (GG)" = "CBT (GG)",
  "CBT (ARC)" = "CBT (ARC)",
  "Dempster et al. p-values (cell line BH)" =
    "Dempster et al. p-values\n(cell line BH)",
  "Dempster et al. p-values (global BH)" =
    "Dempster et al. p-values\n(global BH)"
)

crispr_colors <- c(
  "CBT (GG)" = unname(official_colors["CBT"]),
  "CBT (ARC)" = unname(official_colors["Block empirical"]),
  "Dempster et al. p-values (cell line BH)" =
    unname(official_colors["Pooled empirical"]),
  "Dempster et al. p-values (global BH)" = "#FF7F00"
)

crispr_linetypes <- c(
  "CBT (GG)" = "solid",
  "CBT (ARC)" = "dashed",
  "Dempster et al. p-values (cell line BH)" = "dotdash",
  "Dempster et al. p-values (global BH)" = "twodash"
)

make_crispr_long <- function(data) {
  unrestricted <- data

  out <- rbind(
    data.frame(
      alpha = unrestricted$alpha,
      method = unrestricted$method,
      metric = "Unrestricted: rejections",
      estimate = unrestricted$num_rejections
    ),
    data.frame(
      alpha = unrestricted$alpha,
      method = unrestricted$method,
      metric = "Unrestricted: kFDR",
      estimate = if (all(data$task == "global_null")) unrestricted$global_null_fdp else unrestricted$kFDR
    ),
    data.frame(
      alpha = unrestricted$alpha,
      method = unrestricted$method,
      metric = "Unrestricted: known rejections",
      estimate = unrestricted$known_true_rejections
    )
  )

  out$method <- factor(out$method, levels = crispr_method_levels)
  out$metric <- factor(
    out$metric,
    levels = c(
      "Unrestricted: rejections",
      "Unrestricted: known rejections",
      "Unrestricted: kFDR"
    )
  )
  out
}

crispr_metric_labels <- c(
  "Unrestricted: rejections" = "Rejections",
  "Unrestricted: known rejections" = "Labeled true positives",
  "Unrestricted: kFDR" = "Labeled FDP"
)

build_crispr_metric_panel <- function(
    data,
    metric_name,
    legend_position = "none",
    legend_nrow = 1,
    legend_box = "horizontal") {
  selected <- droplevels(data[data$metric == metric_name, ])
  plot_mapping <- aes(x = alpha, y = estimate, colour = method,
                      linetype = method, group = method)
  plot <- ggplot(selected, plot_mapping)

  if (metric_name == "Unrestricted: kFDR") {
    plot <- plot +
      geom_abline(
        data = data.frame(
          metric = factor(
            "Unrestricted: kFDR",
            levels = levels(data$metric)
          ),
          slope = 1,
          intercept = 0,
          nominal_key = official_nominal_key
        ),
        aes(
          slope = slope,
          intercept = intercept,
          linewidth = nominal_key
        ),
        inherit.aes = FALSE,
        colour = "black",
        linetype = "dotted",
        key_glyph = ggplot2::draw_key_path
      )
  }

  plot <- plot +
    geom_line(linewidth = official_linewidth)
  plot <- plot +
    facet_wrap(
      vars(metric),
      nrow = 1,
      labeller = as_labeller(crispr_metric_labels)
    ) +
    scale_x_continuous(
      breaks = seq(0, 0.20, by = 0.05),
      limits = c(0, 0.20),
      expand = expansion(mult = c(0.02, 0.01))
    ) +
    scale_y_continuous(
      expand = expansion(mult = c(0.035, 0.06))
    ) +
    scale_colour_manual(
      values = crispr_colors,
      breaks = crispr_method_levels,
      labels = unname(crispr_method_labels[crispr_method_levels]),
      drop = FALSE,
      guide = guide_legend(title = NULL, nrow = legend_nrow, order = 1)
    ) +
    scale_linetype_manual(
      values = crispr_linetypes,
      breaks = crispr_method_levels,
      labels = unname(crispr_method_labels[crispr_method_levels]),
      drop = FALSE,
      guide = guide_legend(title = NULL, nrow = legend_nrow, order = 1)
    ) +
    labs(
      x = expression(alpha),
      y = NULL
    ) +
    official_theme() +
    theme(
      legend.position = legend_position,
      legend.box = legend_box,
      legend.box.just = "center",
      legend.text = element_text(size = 9),
      strip.text = element_text(size = 9),
      axis.text.y = if (
        metric_name %in% c(
          "Unrestricted: rejections",
          "Unrestricted: known rejections"
        )
      ) {
        element_text(size = 7.5)
      } else {
        element_text(size = 9)
      },
      plot.margin = margin(1, 5, 1, 2)
    )

  if (metric_name == "Unrestricted: kFDR") {
    plot <- plot +
      scale_linewidth_manual(
        values = c("Nominal level" = 0.45),
        breaks = official_nominal_key,
        guide = guide_legend(
          title = NULL,
          nrow = 1,
          order = 2,
          override.aes = list(
            colour = "black",
            linetype = "dotted",
            shape = NA
          )
        )
      )
  }

  plot
}

extract_crispr_legend <- function(plot) {
  grob <- ggplotGrob(plot)
  candidates <- grep("^guide-box", grob$layout$name)
  for (index in candidates) {
    candidate <- grob$grobs[[index]]
    if (!inherits(candidate, "zeroGrob")) {
      return(candidate)
    }
  }
  stop("Could not extract the CRISPR legend")
}

build_crispr_plot <- function(
    data,
    legend_nrow = 1,
    legend_height = 0.34,
    legend_box = "horizontal") {
  measurement_file <- tempfile(fileext = ".pdf")
  grDevices::cairo_pdf(measurement_file, width = 6, height = 2)
  on.exit({
    grDevices::dev.off()
    unlink(measurement_file)
  }, add = TRUE)

  metrics <- levels(data$metric)
  panel_grobs <- lapply(metrics, function(metric_name) {
    ggplotGrob(
      build_crispr_metric_panel(
        data,
        metric_name = metric_name,
        legend_position = "none",
        legend_nrow = legend_nrow,
        legend_box = legend_box
      )
    )
  })
  aligned_widths <- Reduce(
    grid::unit.pmax,
    lapply(panel_grobs, function(grob) grob$widths)
  )
  panel_grobs <- lapply(panel_grobs, function(grob) {
    grob$widths <- aligned_widths
    grob
  })
  legend <- extract_crispr_legend(
    build_crispr_metric_panel(
      data,
      metric_name = "Unrestricted: kFDR",
      legend_position = "bottom",
      legend_nrow = legend_nrow,
      legend_box = legend_box
    )
  )

  result <- gtable::gtable(
    widths = grid::unit.c(
      grid::unit(0.04, "in"),
      grid::unit(1, "null"),
      grid::unit(0.04, "in"),
      grid::unit(1, "null"),
      grid::unit(0.04, "in"),
      grid::unit(1, "null"),
      grid::unit(0.04, "in")
    ),
    heights = grid::unit.c(
      grid::unit(1, "null"),
      grid::unit(legend_height, "in")
    )
  )
  panel_columns <- c(2, 4, 6)
  for (index in seq_along(panel_grobs)) {
    result <- gtable::gtable_add_grob(
      result,
      panel_grobs[[index]],
      t = 1,
      l = panel_columns[[index]]
    )
  }
  gtable::gtable_add_grob(result, legend, t = 2, l = 2, r = 6)
}
