# Three scales with FDP, rejection counts, and true positives.
utopia_method_levels <- c(
  "CBT (GG)",
  "CBT (ARC)",
  "CBT (one control)",
  "UTOPIA (within-cluster)",
  "UTOPIA (global)"
)

utopia_method_labels <- expression(
  "CBT (GG)" * phantom("")[phantom(j)],
  "CBT (ARC)" * phantom("")[phantom(j)],
  "CBT (" * n[j] == 1 * ")",
  "UTOPIA (cluster BH)" * phantom("")[phantom(j)],
  "UTOPIA (global BH)" * phantom("")[phantom(j)]
)
names(utopia_method_labels) <- utopia_method_levels

utopia_colors <- c(
  "CBT (GG)" = unname(official_colors["CBT"]),
  "CBT (ARC)" = unname(official_colors["Block empirical"]),
  "CBT (one control)" = unname(official_colors["BarSam"]),
  "UTOPIA (within-cluster)" = unname(official_colors["Pooled empirical"]),
  "UTOPIA (global)" = "#FF7F00"
)

utopia_linetypes <- c(
  "CBT (GG)" = "solid",
  "CBT (ARC)" = "dashed",
  "CBT (one control)" = "longdash",
  "UTOPIA (within-cluster)" = "dotdash",
  "UTOPIA (global)" = "twodash"
)

utopia_metric_levels <- c("FDP", "Rejections", "True positives")
utopia_inflation_factor <- pi^2 / 6
utopia_inflation_key <- "pi2_over_6_nominal"
utopia_reference_levels <- c(official_nominal_key, utopia_inflation_key)
utopia_reference_linewidth <- 0.65
utopia_reference_labels <- expression(
  "Nominal level",
  paste("(", pi^2, "/6)") %*% "nominal"
)

utopia_metric_labels <- c(
  "FDP" = "FDP",
  "Rejections" = "Rejections",
  "True positives" = "True positives"
)

make_utopia_long <- function(data) {
  out <- rbind(
    data.frame(
      scale = data$scale,
      alpha = data$alpha,
      method = data$method,
      metric = "FDP",
      estimate = data$fdp
    ),
    data.frame(
      scale = data$scale,
      alpha = data$alpha,
      method = data$method,
      metric = "Rejections",
      estimate = data$num_rejections
    ),
    data.frame(
      scale = data$scale,
      alpha = data$alpha,
      method = data$method,
      metric = "True positives",
      estimate = data$true_rejections
    )
  )
  out$method <- factor(out$method, levels = utopia_method_levels)
  out$metric <- factor(out$metric, levels = utopia_metric_levels)
  out
}

utopia_y_axis_labels <- function(values) {
  labels <- if (max(abs(values), na.rm = TRUE) <= 1) {
    scales::label_number(accuracy = 0.1)(values)
  } else {
    scales::label_comma(accuracy = 1)(values)
  }
  labels[which.min(abs(values))] <- ""
  labels
}

build_utopia_scale_column <- function(
    data,
    scale_value,
    show_metric_labels = FALSE,
    show_x_title = FALSE,
    legend_position = "none") {
  selected <- droplevels(
    data[data$scale == scale_value, ]
  )
  limit_rows <- do.call(
    rbind,
    lapply(utopia_metric_levels, function(metric_name) {
      range_data <- if (metric_name == "FDP") data else selected
      observed_max <- max(
        range_data$estimate[range_data$metric == metric_name],
        na.rm = TRUE
      )
      upper <- if (metric_name == "FDP") {
        min(1, max(0.525, 1.06 * observed_max))
      } else {
        max(1, 1.06 * observed_max)
      }
      data.frame(
        alpha = 0,
        metric = factor(metric_name, levels = utopia_metric_levels),
        estimate = c(0, upper)
      )
    })
  )
  plot <- ggplot(
    selected,
    aes(
      x = alpha,
      y = estimate,
      colour = method,
      linetype = method,
      group = method
    )
  )

  plot <- plot +
    geom_blank(
      data = limit_rows,
      aes(x = alpha, y = estimate),
      inherit.aes = FALSE
    ) +
    geom_abline(
      data = data.frame(
        metric = factor(
          rep("FDP", length(utopia_reference_levels)),
          levels = utopia_metric_levels
        ),
        slope = c(1, utopia_inflation_factor),
        intercept = 0,
        reference_key = factor(
          utopia_reference_levels,
          levels = utopia_reference_levels
        )
      ),
      aes(
        slope = slope,
        intercept = intercept,
        linewidth = reference_key
      ),
      inherit.aes = FALSE,
      colour = c("black", "#CC0000"),
      linetype = c("dotted", "dotted"),
      alpha = c(1, 1),
      key_glyph = ggplot2::draw_key_path
    ) +
    geom_line(linewidth = official_linewidth) +
    facet_grid(
      rows = vars(metric),
      scales = "free_y",
      switch = "y"
    ) +
    scale_x_continuous(
      breaks = seq(0, 0.5, by = 0.1),
      limits = c(0, 0.5),
      expand = expansion(mult = c(0.02, 0.01))
    ) +
    scale_y_continuous(
      labels = utopia_y_axis_labels,
      expand = expansion(mult = c(0, 0.12))
    ) +
    scale_colour_manual(
      values = utopia_colors,
      breaks = utopia_method_levels,
      labels = unname(utopia_method_labels[utopia_method_levels]),
      drop = FALSE,
      guide = guide_legend(title = NULL, nrow = 1, byrow = TRUE, order = 1)
    ) +
    scale_linetype_manual(
      values = utopia_linetypes,
      breaks = utopia_method_levels,
      labels = unname(utopia_method_labels[utopia_method_levels]),
      drop = FALSE,
      guide = guide_legend(title = NULL, nrow = 1, byrow = TRUE, order = 1)
    ) +
    labs(
      x = if (show_x_title) expression(alpha) else NULL,
      y = NULL
    ) +
    official_theme() +
    theme(
      legend.position = legend_position,
      legend.box = "vertical",
      legend.box.just = "center",
      axis.title.x = if (show_x_title) element_text(size = 9) else element_blank(),
      strip.text.y.left = if (show_metric_labels) {
        element_text(size = 9, angle = 90)
      } else {
        element_blank()
      },
      strip.background.y = element_blank(),
      panel.spacing.y = grid::unit(0.07, "in"),
      plot.margin = margin(0, 0, 0, 0)
    )

  plot +
    scale_linewidth_manual(
      values = setNames(
        rep(utopia_reference_linewidth, length(utopia_reference_levels)),
        utopia_reference_levels
      ),
      breaks = utopia_reference_levels,
      labels = utopia_reference_labels,
      guide = guide_legend(
        title = NULL,
        nrow = 1,
        order = 2,
        override.aes = list(
          colour = c("black", "#CC0000"),
          linetype = c("dotted", "dotted"),
          alpha = c(1, 1)
        )
      )
    )
}

scale_strip_grob <- function(label) {
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

shrink_count_axis_text <- function(grob, fontsize = 7.5) {
  axis_indices <- which(grob$layout$name %in% c("axis-l-2", "axis-l-3"))
  for (axis_index in axis_indices) {
    axis_table <- grob$grobs[[axis_index]]$children$axis
    label_indices <- which(vapply(
      axis_table$grobs,
      function(axis_grob) inherits(axis_grob, "titleGrob"),
      logical(1)
    ))
    for (label_index in label_indices) {
      label_grob <- axis_table$grobs[[label_index]]
      for (child_name in names(label_grob$children)) {
        label_grob$children[[child_name]]$gp$fontsize <- fontsize
      }
      axis_table$grobs[[label_index]] <- label_grob
    }
    grob$grobs[[axis_index]]$children$axis <- axis_table
  }
  grob
}

add_scale_strip <- function(grob, label) {
  panel_rows <- unique(grob$layout$t[grepl("^panel", grob$layout$name)])
  panel_columns <- unique(grob$layout$l[grepl("^panel", grob$layout$name)])
  if (length(panel_columns) != 1) {
    stop("Expected one panel column in each scale-specific plot.")
  }
  insert_after <- min(panel_rows) - 1
  grob <- gtable::gtable_add_rows(
    grob,
    heights = grid::unit(0.18, "in"),
    pos = insert_after
  )
  gtable::gtable_add_grob(
    grob,
    scale_strip_grob(label),
    t = insert_after + 1,
    l = panel_columns[[1]],
    name = paste0("scale-strip-", label)
  )
}

extract_utopia_legend <- function(plot) {
  grob <- ggplotGrob(plot)
  candidates <- grep("^guide-box", grob$layout$name)
  for (index in candidates) {
    candidate <- grob$grobs[[index]]
    if (!inherits(candidate, "zeroGrob")) {
      return(candidate)
    }
  }
  stop("Could not extract the UTOPIA legend")
}

build_utopia_plot <- function(data) {
  measurement_file <- tempfile(fileext = ".pdf")
  grDevices::cairo_pdf(measurement_file, width = 6, height = 3.5)
  on.exit({
    grDevices::dev.off()
    unlink(measurement_file)
  }, add = TRUE)

  scale_values <- c(1, 2, 4)
  column_grobs <- lapply(seq_along(scale_values), function(column_index) {
    scale_value <- scale_values[[column_index]]
    column_grob <- shrink_count_axis_text(
      ggplotGrob(
        build_utopia_scale_column(
          data,
          scale_value = scale_value,
          show_metric_labels = column_index == 1,
          show_x_title = TRUE,
          legend_position = "none"
        )
      )
    )
    add_scale_strip(
      column_grob,
      sprintf("Scale = %d", scale_value)
    )
  })

  aligned_heights <- Reduce(
    grid::unit.pmax,
    lapply(column_grobs, function(grob) grob$heights)
  )
  for (column_index in seq_along(column_grobs)) {
    column_grobs[[column_index]]$heights <- aligned_heights
  }

  legend <- extract_utopia_legend(
    build_utopia_scale_column(
        data,
      scale_value = 1,
      show_metric_labels = TRUE,
      show_x_title = TRUE,
      legend_position = "bottom"
    )
  )

  result <- do.call(
    cbind,
    c(column_grobs, list(size = "first"))
  )
  result <- gtable::gtable_add_rows(
    result,
    heights = grid::unit(0.50, "in")
  )
  result <- gtable::gtable_add_grob(
    result,
    legend,
    t = nrow(result),
    l = 1,
    r = ncol(result)
  )
  result <- gtable::gtable_add_cols(
    result,
    widths = grid::unit(0.04, "in"),
    pos = 0
  )
  gtable::gtable_add_cols(
    result,
    widths = grid::unit(0.10, "in")
  )
}

