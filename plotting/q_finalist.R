# BH comparison for the two manuscript (m, n) settings.
selected_q_colours <- c(
  "1" = "#A6D96A",
  "2" = "#1B7837",
  "5" = "#2C7FB8",
  "10" = "#762A83",
  "20" = "#D73027",
  "25" = "#D73027"
)

q_colour_scale <- function(q_values, guide = "colourbar") {
  scale_colour_manual(values = selected_q_colours[as.character(q_values)],
    breaks = as.character(q_values), name = expression(q), guide = guide, drop = FALSE)
}

q_fill_scale <- function(q_values) {
  scale_fill_manual(values = selected_q_colours[as.character(q_values)],
    breaks = as.character(q_values), guide = "none", drop = FALSE)
}

extract_q_legend <- function(plot) {
  grob <- ggplotGrob(plot)
  candidates <- grep("^guide-box", grob$layout$name)
  for (index in candidates) {
    candidate <- grob$grobs[[index]]
    if (!inherits(candidate, "zeroGrob")) {
      return(candidate)
    }
  }
  stop("Could not extract q-finalist legend.")
}

build_q_nominal_legend <- function() {
  nominal <- data.frame(
    x = c(0, 1),
    y = c(0, 0),
    label = "Nominal level"
  )
  plot <- ggplot(
    nominal,
    aes(x = x, y = y, linetype = label, group = label)
  ) +
    geom_line(colour = official_nominal_color, linewidth = 0.4) +
    scale_linetype_manual(
      values = c("Nominal level" = "dashed"),
      breaks = "Nominal level",
      guide = guide_legend(title = NULL)
    ) +
    official_theme() +
    theme(legend.position = "bottom")
  extract_q_legend(plot)
}

comparison_setting_ids <- c(
  original = "m500_n50",
  new = "m50_n500"
)

comparison_setting_values <- list(
  "m500_n50" = c(m = 500L, n = 50L),
  "m50_n500" = c(m = 50L, n = 500L)
)

make_q_comparison_metric <- function(data, metric) {
  bh <- data[data$procedure == "bh", ]
  if (metric == "Power") {
    bh$estimate <- bh$tpr_mean
    bh$lower <- bh$tpr_two_se_low
    bh$upper <- bh$tpr_two_se_high
  } else {
    bh$estimate <- bh$fdr_mean
    bh$lower <- bh$fdr_two_se_low
    bh$upper <- bh$fdr_two_se_high
  }
  bh$metric <- metric
  bh
}

make_q_comparison_limits <- function(data, metric, alpha = 0.10) {
  upper <- if (metric == "Power") {
    1
  } else {
    max(alpha, data$estimate, na.rm = TRUE) + 0.025
  }
  data.frame(
    eta = min(data$eta),
    y = c(0, upper)
  )
}

build_q_comparison_metric_plot <- function(data, metric, alpha = 0.10) {
  metric_data <- make_q_comparison_metric(data, metric)
  q_values <- sort(unique(metric_data$q))
  metric_data$q_plot <- factor(metric_data$q, levels = q_values)
  limits <- make_q_comparison_limits(metric_data, metric, alpha)
  nominal <- data.frame(threshold = alpha)

  plot <- ggplot(
    metric_data,
    aes(
      x = eta,
      y = estimate,
      colour = q_plot,
      fill = q_plot,
      group = q
    )
  ) +
    geom_blank(
      data = limits,
      aes(x = eta, y = y),
      inherit.aes = FALSE
    )
  if (metric == "FDR") {
    plot <- plot +
      geom_hline(
        data = nominal,
        aes(yintercept = threshold),
        inherit.aes = FALSE,
        colour = official_nominal_color,
        linetype = "dashed",
        linewidth = 0.4
      )
  }
  plot +
    geom_ribbon(
      aes(ymin = lower, ymax = upper),
      colour = NA,
      alpha = 0.035,
      show.legend = FALSE
    ) +
    geom_line(linewidth = 0.34, show.legend = FALSE) +
    geom_point(size = 0.66, show.legend = FALSE) +
    scale_x_continuous(
      breaks = function(limits) pretty(limits, n = 5),
      expand = expansion(mult = c(0.075, 0.075))
    ) +
    scale_y_continuous(expand = expansion(mult = c(0.035, 0))) +
    q_colour_scale(q_values, guide = "none") +
    q_fill_scale(q_values) +
    labs(x = expression(eta), y = metric) +
    official_theme() +
    theme(
      legend.position = "none",
      plot.margin = margin(1, 1, 1, 1)
    )
}

make_q_comparison_setting_subtitle <- function(setting) {
  values <- comparison_setting_values[[setting]]
  if (is.null(values)) {
    stop(sprintf("Unknown q-comparison setting: %s", setting))
  }
  m_value <- unname(values[["m"]])
  n_value <- unname(values[["n"]])
  grid::textGrob(
    bquote(
      group("(", list(m, n), ")") ==
        group("(", list(.(m_value), .(n_value)), ")")
    ),
    gp = grid::gpar(
      fontfamily = "Times New Roman",
      fontsize = 8.5
    )
  )
}

build_q_comparison_colour_legend <- function(data) {
  q_values <- sort(unique(data$q))
  q_count <- length(q_values)
  legend_data <- data.frame(
    eta = rep(c(2, 6), each = q_count),
    estimate = rep(seq_len(q_count), times = 2) / q_count,
    q = rep(q_values, times = 2)
  )
  legend_data$q_plot <- factor(legend_data$q, levels = q_values)
  plot <- ggplot(
    legend_data,
    aes(x = eta, y = estimate, colour = q_plot, group = q)
  ) +
    geom_line(linewidth = 0.34) +
    geom_point(size = 0.66) +
    q_colour_scale(
      q_values,
      guide = guide_legend(
        title = expression(q),
        title.position = "left",
        title.hjust = 0.5,
        direction = "horizontal",
        nrow = 1,
        byrow = TRUE
      )
    ) +
    official_theme() +
    theme(legend.position = "bottom")
  extract_q_legend(plot)
}

build_q_comparison_legend <- function(data) {
  gtable::gtable_matrix(
    name = "q-comparison-legends",
    grobs = matrix(
      list(
        build_q_comparison_colour_legend(data),
        build_q_nominal_legend()
      ),
      ncol = 1
    ),
    widths = grid::unit(1, "null"),
    heights = grid::unit(c(1, 0.72), "null")
  )
}

build_q_comparison_overall <- function(data) {
  measurement_file <- tempfile(fileext = ".pdf")
  grDevices::cairo_pdf(measurement_file, width = 6, height = 2.15)
  on.exit({
    grDevices::dev.off()
    unlink(measurement_file)
  }, add = TRUE)

  setting_levels <- levels(data$setting)
  if (length(setting_levels) != 2L) {
    stop("The q-comparison figure requires exactly two (m, n) settings")
  }
  panel_specs <- do.call(
    c,
    lapply(
      setting_levels,
      function(setting) {
        list(
          list(setting = setting, metric = "FDR"),
          list(setting = setting, metric = "Power")
        )
      }
    )
  )
  panel_grobs <- lapply(panel_specs, function(spec) {
    setting_data <- data[data$setting == spec$setting, ]
    ggplotGrob(
      build_q_comparison_metric_plot(setting_data, spec$metric)
    )
  })
  aligned_heights <- do.call(
    grid::unit.pmax,
    lapply(panel_grobs, function(grob) grob$heights)
  )
  aligned_widths <- do.call(
    grid::unit.pmax,
    lapply(panel_grobs, function(grob) grob$widths)
  )
  panel_grobs <- lapply(panel_grobs, function(grob) {
    grob$heights <- aligned_heights
    grob$widths <- aligned_widths
    grob
  })
  legend <- build_q_comparison_legend(data)

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
      grid::unit(0.56, "in")
    )
  )
  result <- gtable::gtable_add_grob(
    result,
    make_q_comparison_setting_subtitle(setting_levels[[1]]),
    t = 1,
    l = 1,
    r = 2
  )
  result <- gtable::gtable_add_grob(
    result,
    make_q_comparison_setting_subtitle(setting_levels[[2]]),
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

