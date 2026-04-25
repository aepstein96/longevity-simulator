#!/usr/bin/env Rscript
# Pull NHANES chronic-condition data across cycles I (2015-16), J (2017-18),
# and L (2021-Aug 2023), compute survey-weighted prevalence of multimorbidity
# by 5-year age band x sex, and write tidy CSVs to data/NHANES/.
#
# Run inside the `nhanes` conda env:
#   mamba run -n nhanes Rscript preprocessing/process_nhanes.R

suppressPackageStartupMessages({
  library(nhanesA)
  library(dplyr)
  library(tidyr)
  library(survey)
})

CYCLES <- c("I", "J", "L")  # 2015-16, 2017-18, 2021-Aug 2023
N_CYCLES <- length(CYCLES)

OUT_DIR <- "data/NHANES"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

CONDITIONS <- c("asthma","arthritis","chf","chd","angina","heart_attack",
                "stroke","liver","cancer","diabetes","high_bp","high_chol",
                "kidney","depression")

# --- pull + recode one cycle ---------------------------------------------

yesno <- function(x) {
  # NHANES: 1 = Yes, 2 = No, 7 = Refused, 9 = Don't know
  r <- rep(NA_integer_, length(x))
  r[x == 1] <- 1L
  r[x == 2] <- 0L
  r
}

pull_cycle <- function(suffix) {
  cat("  cycle", suffix, "... ")
  # translated = FALSE gives raw numeric codes (easier to recode)
  demo <- nhanesA::nhanes(paste0("DEMO_", suffix), translated = FALSE)
  mcq  <- nhanesA::nhanes(paste0("MCQ_",  suffix), translated = FALSE)
  diq  <- nhanesA::nhanes(paste0("DIQ_",  suffix), translated = FALSE)
  bpq  <- nhanesA::nhanes(paste0("BPQ_",  suffix), translated = FALSE)
  kiq  <- nhanesA::nhanes(paste0("KIQ_U_",suffix), translated = FALSE)
  dpq  <- nhanesA::nhanes(paste0("DPQ_",  suffix), translated = FALSE)

  demo <- demo[, intersect(c("SEQN","RIDAGEYR","RIAGENDR","WTMEC2YR",
                              "SDMVPSU","SDMVSTRA"), colnames(demo))]
  mcq_cols <- c("MCQ010","MCQ160A","MCQ160B","MCQ160C","MCQ160D",
                "MCQ160E","MCQ160F","MCQ160L","MCQ220",
                # older cycles use lowercase a/b/c/... in some variables
                "MCQ160a","MCQ160b","MCQ160c","MCQ160d","MCQ160e",
                "MCQ160f","MCQ160l")
  mcq <- mcq[, c("SEQN", intersect(mcq_cols, colnames(mcq)))]
  diq <- diq[, c("SEQN", intersect("DIQ010", colnames(diq)))]
  bpq <- bpq[, c("SEQN", intersect(c("BPQ020","BPQ080"), colnames(bpq)))]
  kiq <- kiq[, c("SEQN", intersect("KIQ022", colnames(kiq)))]
  dpq_items <- intersect(c("DPQ010","DPQ020","DPQ030","DPQ040","DPQ050",
                           "DPQ060","DPQ070","DPQ080","DPQ090"), colnames(dpq))
  dpq <- dpq[, c("SEQN", dpq_items)]

  df <- demo %>%
    left_join(mcq, by = "SEQN") %>%
    left_join(diq, by = "SEQN") %>%
    left_join(bpq, by = "SEQN") %>%
    left_join(kiq, by = "SEQN") %>%
    left_join(dpq, by = "SEQN")

  # normalise column casing: some cycles uppercase, some lowercase
  canon <- function(name) {
    matches <- intersect(c(name, toupper(name), tolower(name)), colnames(df))
    if (length(matches) == 0) return(rep(NA_integer_, nrow(df)))
    df[[matches[1]]]
  }

  df$cycle <- suffix
  df$asthma       <- yesno(canon("MCQ010"))
  df$arthritis    <- yesno(canon("MCQ160A"))
  df$chf          <- yesno(canon("MCQ160B"))
  df$chd          <- yesno(canon("MCQ160C"))
  df$angina       <- yesno(canon("MCQ160D"))
  df$heart_attack <- yesno(canon("MCQ160E"))
  df$stroke       <- yesno(canon("MCQ160F"))
  df$liver        <- yesno(canon("MCQ160L"))
  df$cancer       <- yesno(canon("MCQ220"))

  # Diabetes: 1=yes, 2=no, 3=borderline, 7/9=missing.
  # Follow King 2018 and treat borderline as yes.
  diq_raw <- canon("DIQ010")
  df$diabetes <- NA_integer_
  df$diabetes[diq_raw %in% c(1, 3)] <- 1L
  df$diabetes[diq_raw == 2]         <- 0L

  df$high_bp   <- yesno(canon("BPQ020"))
  df$high_chol <- yesno(canon("BPQ080"))
  df$kidney    <- yesno(canon("KIQ022"))

  # PHQ-9 depression: sum of DPQ010-DPQ090, >= 10 positive; NA if any item missing
  phq_cols <- intersect(c("DPQ010","DPQ020","DPQ030","DPQ040","DPQ050",
                          "DPQ060","DPQ070","DPQ080","DPQ090"), colnames(df))
  if (length(phq_cols) == 9) {
    M <- as.matrix(df[, phq_cols])
    M[M %in% c(7, 9)] <- NA
    phq_total <- rowSums(M)
    df$depression <- as.integer(phq_total >= 10)
  } else {
    df$depression <- NA_integer_
  }

  keep_cols <- c("SEQN","cycle","RIDAGEYR","RIAGENDR","WTMEC2YR",
                 "SDMVPSU","SDMVSTRA", CONDITIONS)
  df <- df[, intersect(keep_cols, colnames(df))]
  cat(nrow(df), "rows\n")
  df
}

# --- pull all cycles -----------------------------------------------------

cat("Pulling NHANES cycles:", paste(CYCLES, collapse=", "), "\n")
frames <- lapply(CYCLES, pull_cycle)
all_data <- bind_rows(frames)
cat("Combined:", nrow(all_data), "rows across", N_CYCLES, "cycles\n")

# --- derive multimorbidity flags ----------------------------------------

cond_mat <- as.matrix(all_data[, CONDITIONS])
all_data$n_conditions <- rowSums(cond_mat, na.rm = TRUE)
all_data$n_missing    <- rowSums(is.na(cond_mat))
all_data$any_cond     <- as.integer(all_data$n_conditions >= 1)
all_data$multi_cond   <- as.integer(all_data$n_conditions >= 2)

# Combined-cycle weights: divide MEC weight by number of cycles
all_data$wt <- all_data$WTMEC2YR / N_CYCLES

# Filter: adults only, valid weight
all_data <- all_data %>%
  filter(!is.na(RIDAGEYR), RIDAGEYR >= 18,
         !is.na(WTMEC2YR), WTMEC2YR > 0)

# 5-year bands (RIDAGEYR top-coded at 80 so 80+ is an aggregate)
age_breaks <- c(18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, Inf)
age_labels <- c("18-24","25-29","30-34","35-39","40-44","45-49",
                "50-54","55-59","60-64","65-69","70-74","75-79","80+")
all_data$age_band <- cut(all_data$RIDAGEYR, breaks = age_breaks,
                         right = FALSE, labels = age_labels)
all_data$sex <- ifelse(all_data$RIAGENDR == 1, "Male", "Female")

# --- survey design ------------------------------------------------------

options(survey.lonely.psu = "adjust")
design <- svydesign(ids = ~SDMVPSU, strata = ~SDMVSTRA, weights = ~wt,
                    nest = TRUE, data = all_data)

# helper: run svyby for both sex-stratified and sex-combined, return tidy df
# svyby's CI columns are always literally "ci_l" and "ci_u" regardless of var.
svyby_tidy <- function(varname, design, value_col = "value") {
  f <- as.formula(paste0("~", varname))
  by_sex  <- as.data.frame(svyby(f, ~age_band + sex, design, svymean,
                                 na.rm = TRUE, vartype = "ci"))
  by_both <- as.data.frame(svyby(f, ~age_band,       design, svymean,
                                 na.rm = TRUE, vartype = "ci"))
  by_both$sex <- "Both"
  tidy <- function(d) {
    data.frame(
      age_band = as.character(d$age_band),
      sex      = as.character(d$sex),
      value    = d[[varname]],
      ci_lower = d$ci_l,
      ci_upper = d$ci_u,
      stringsAsFactors = FALSE
    )
  }
  out <- rbind(tidy(by_sex), tidy(by_both))
  names(out)[names(out) == "value"] <- value_col
  out
}

# --- write outputs ------------------------------------------------------

any_df   <- svyby_tidy("any_cond",     design, "prevalence")
multi_df <- svyby_tidy("multi_cond",   design, "prevalence")
mean_df  <- svyby_tidy("n_conditions", design, "mean")

# per-condition prevalence (long format) - subset to rows where the condition is non-NA
per_cond_rows <- list()
for (c in CONDITIONS) {
  if (all(is.na(all_data[[c]]))) {
    cat("  condition", c, "is all NA, skipping\n")
    next
  }
  sub_design <- subset(design, !is.na(all_data[[c]]))
  tryCatch({
    tidy <- svyby_tidy(c, sub_design, "prevalence")
    tidy$condition <- c
    per_cond_rows[[c]] <- tidy[, c("condition","age_band","sex",
                                    "prevalence","ci_lower","ci_upper")]
  }, error = function(e) {
    cat("  condition", c, "skipped:", conditionMessage(e), "\n")
  })
}
per_cond_df <- bind_rows(per_cond_rows)

# Per-person output for downstream simulation.
# good_health == 1 iff the respondent has zero of the tracked conditions
# (NA items are treated as 0 — standard NHANES convention, matches how
#  n_conditions was computed above).
participants <- all_data %>%
  transmute(
    SEQN        = SEQN,
    cycle       = cycle,
    age         = RIDAGEYR,
    sex         = sex,
    wt          = wt,
    SDMVPSU     = SDMVPSU,
    SDMVSTRA    = SDMVSTRA,
    n_conditions = n_conditions,
    n_missing    = n_missing,
    good_health  = as.integer(n_conditions == 0),
    !!!rlang::syms(CONDITIONS)
  )

write.csv(any_df,      file.path(OUT_DIR, "prevalence_any_chronic_by_age_sex.csv"),
          row.names = FALSE)
write.csv(multi_df,    file.path(OUT_DIR, "prevalence_2plus_chronic_by_age_sex.csv"),
          row.names = FALSE)
write.csv(mean_df,     file.path(OUT_DIR, "mean_conditions_by_age_sex.csv"),
          row.names = FALSE)
write.csv(per_cond_df, file.path(OUT_DIR, "prevalence_per_condition_by_age_sex.csv"),
          row.names = FALSE)
write.csv(participants, file.path(OUT_DIR, "participants.csv"),
          row.names = FALSE)

cat("\nWrote:\n")
for (f in list.files(OUT_DIR, full.names = TRUE)) {
  cat("  ", f, "  (", file.size(f), " bytes)\n", sep = "")
}
cat("\nSample (any chronic condition, Both sexes):\n")
print(subset(any_df, sex == "Both"))
