# Shared manuscript typography, palette, and PDF/PNG export.
if (!nzchar(Sys.getenv("XDG_CACHE_HOME", unset = ""))) {
  Sys.setenv(XDG_CACHE_HOME = tempdir())
}

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  stop(
    paste(
      "The ggplot2 package is required.",
      "Install it with install.packages('ggplot2') and rerun the script."
    )
  )
}

suppressPackageStartupMessages(library(ggplot2))

official_colors <- c(
  "CBT" = "#0072B2",
  "Equal-weight CBT" = "#CC79A7",
  "Subsampled CBT" = "#D55E00",
  "Pooled empirical" = "#E69F00",
  "Block empirical" = "#009E73",
  "Individual empirical" = "#009E73",
  "BarSam" = "#7B3294",
  "Compound empirical" = "#7B3294",
  "CBT e-values" = "#333333"
)

official_linewidth <- 0.58
official_point_size <- 1.5
homogeneous_linewidth <- 0.40
homogeneous_point_size <- 1.05
official_ribbon_alpha <- 0.12
official_nominal_color <- "#4D4D4D"
official_nominal_key <- "Nominal level"
official_strip_fill <- "grey88"
official_strip_color <- "grey35"
official_strip_linewidth <- 0.35
official_strip_grid_lwd <- official_strip_linewidth * 72.27 / 25.4
official_group_separator_gap <- grid::unit(0.12, "in")

official_theme <- function() {
  theme_bw(base_size = 9, base_family = "Times New Roman") +
    theme(
      text = element_text(family = "Times New Roman", size = 9),
      axis.title = element_text(size = 9),
      axis.text = element_text(size = 9),
      strip.text = element_text(size = 9),
      strip.background = element_rect(
        fill = official_strip_fill,
        colour = official_strip_color,
        linewidth = official_strip_linewidth
      ),
      strip.placement = "outside",
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = "grey90", linewidth = 0.25),
      panel.spacing = grid::unit(0.35, "lines"),
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = 9),
      legend.background = element_blank(),
      legend.margin = margin(0, 0, 0, 0),
      legend.box.margin = margin(0, 0, 0, 0),
      legend.box.spacing = grid::unit(0.05, "lines"),
      legend.spacing.x = grid::unit(0.15, "lines"),
      legend.key.height = grid::unit(0.7, "lines"),
      legend.key.width = grid::unit(1.05, "lines"),
      plot.margin = margin(4, 4, 2, 4)
    )
}

save_official_plot <- function(plot, output_dir, stem, width, height) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  pdf_path <- file.path(output_dir, paste0(stem, ".pdf"))
  png_path <- file.path(output_dir, paste0(stem, ".png"))

  ggsave(
    filename = pdf_path,
    plot = plot,
    width = width,
    height = height,
    device = cairo_pdf,
    bg = "white"
  )
  ggsave(
    filename = png_path,
    plot = plot,
    width = width,
    height = height,
    dpi = 300,
    bg = "white"
  )

  invisible(c(pdf = pdf_path, png = png_path))
}

