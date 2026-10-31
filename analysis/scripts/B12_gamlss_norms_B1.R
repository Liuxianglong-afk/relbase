# 课题 B1：各数据长度的 GAMLSS 常模与交叉拟合 z 值（protocol_B1 v0.2 第 6 节）
#
# 模型（每个特征一个）：y ~ family(mu = pb(age) + sex [+ meanfd], sigma = pb(age) + sex)
#   功能特征（strength、netfc）μ 中加 meanfd（全长保留帧的平均 FD）；厚度不加。
#   分布族：NO / SHASHo 按 BIC 在全长 TC 上选一次，其余数据长度沿用。
# z 值不来自含本人的模型：
#   TC：按站点分层的 5 折交叉拟合（seed 42，所有数据长度共用同一折分配）
#   ASD：全部 TC 拟合的模型
#   run1/run2：用 half1 的模型评估（TC 用其所在折的 half1 模型，ASD 用全部 TC 的 half1 模型）
# 站点重新居中不在本脚本中做（B13）。
# 输出（derivatives/norms_B1/）：B12_families.csv、B12_z_<set>_<cond>.csv.gz（长表：cohort, sub_n, feature, z）、B12_folds.csv
# 用法: Rscript scripts/B12_gamlss_norms_B1.R [核数] [每个特征集最多特征数，0=全部] [变体，默认 main]
#   变体（敏感性分析）: fd30 / oddeven / gordon / combat → 读 features_B/<变体>/，写 norms_B1/sens_<变体>/，只做 strength；
#   fd30 用 pass_fd30 与 meanfd_retained_fd30。main 的行为与此前完全相同。
suppressPackageStartupMessages({
  .libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
  library(gamlss); library(data.table); library(parallel)
})
args <- commandArgs(trailingOnly = TRUE)
ncores <- ifelse(length(args) >= 1, as.integer(args[1]), 12)
maxfeat <- ifelse(length(args) >= 2, as.integer(args[2]), 0)
variant <- ifelse(length(args) >= 3, args[3], "main")
root <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))
fdir <- file.path(root, "derivatives/features_B"); outd <- file.path(root, "derivatives/norms_B1")
if (variant != "main") { fdir <- file.path(fdir, variant); outd <- file.path(outd, paste0("sens_", variant)) }
pcol <- if (variant == "fd30") "pass_fd30" else "pass_fd20"; fdcol <- if (variant == "fd30") "meanfd_retained_fd30" else "meanfd_retained_fd20"
dir.create(outd, recursive = TRUE, showWarnings = FALSE)
FAMS <- c("NO", "SHASHo")

qc <- fread(tail(sort(Sys.glob(file.path(root, "data/interim/abide_rest_qc_*.csv"))), 1))
cov <- qc[get(pcol) == TRUE, .(cohort, sub_n, site, dx = dx_group, age, sex = factor(ifelse(sex == 2, "F", "M"), levels = c("F", "M")),
                               meanfd = get(fdcol))]
set.seed(42)
tcs <- cov[dx == 2]
tcs[, fold := sample(rep(1:5, length.out = .N)), by = site]
fwrite(tcs[, .(cohort, sub_n, site, fold)], file.path(outd, "B12_folds.csv"))
cov <- merge(cov, tcs[, .(cohort, sub_n, fold)], by = c("cohort", "sub_n"), all.x = TRUE)
cat("TC:", nrow(tcs), " ASD:", cov[dx == 1, .N], "\n")

load_set <- function(set, cond) {
  f <- file.path(fdir, sprintf("B11_%s_%s.csv", set, cond))
  if (!file.exists(f)) return(NULL)
  merge(cov, fread(f, check.names = FALSE), by = c("cohort", "sub_n"))
}

fit_one <- function(dat, y, fam, use_fd) {
  dd <- data.frame(y = y, age = dat$age, sex = dat$sex, meanfd = dat$meanfd)
  fml <- if (use_fd) y ~ pb(age) + sex + meanfd else y ~ pb(age) + sex
  m <- tryCatch(do.call(gamlss, list(formula = fml, sigma.formula = ~ pb(age) + sex, family = fam, data = dd,
                                     control = gamlss.control(n.cyc = 200, trace = FALSE))), error = function(e) NULL)
  if (!is.null(m)) attr(m, "train") <- dd
  m
}

zscore <- function(m, fam, dat, y) {
  nd <- data.frame(age = dat$age, sex = dat$sex, meanfd = dat$meanfd)
  pa <- predictAll(m, newdata = nd, data = attr(m, "train"), type = "response")
  argl <- list(q = y, mu = pa$mu, sigma = pa$sigma)
  if (!is.null(pa$nu)) argl$nu <- pa$nu
  if (!is.null(pa$tau)) argl$tau <- pa$tau
  qnorm(pmin(pmax(do.call(get(paste0("p", fam)), argl), 1e-7), 1 - 1e-7))
}

cl <- makeCluster(ncores)
clusterEvalQ(cl, { .libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths())); suppressPackageStartupMessages({ library(gamlss); library(data.table) }) })
clusterExport(cl, c("fit_one", "zscore", "FAMS"))

sets <- if (variant == "main") list(strength = TRUE, netfc = TRUE, thickness = FALSE) else list(strength = TRUE)
fam_all <- list()
for (set in names(sets)) {
  use_fd <- sets[[set]]
  cond_full <- if (set == "thickness") "" else "full"
  full <- if (set == "thickness") merge(cov, fread(file.path(fdir, "B11_thickness.csv"), check.names = FALSE), by = c("cohort", "sub_n")) else load_set(set, "full")
  feats <- setdiff(names(full), c(names(cov)))
  if (maxfeat > 0) feats <- head(feats, maxfeat)
  clusterExport(cl, c("use_fd"), envir = environment())

  # 1. 分布族（全长 TC）
  tcfull <- full[dx == 2]
  clusterExport(cl, "tcfull", envir = environment())
  fam_tab <- rbindlist(parLapply(cl, feats, function(f) {
    b <- sapply(FAMS, function(fa) { m <- fit_one(tcfull, tcfull[[f]], fa, use_fd); if (is.null(m)) NA_real_ else BIC(m) })
    data.table(feature = f, family = if (all(is.na(b))) NA_character_ else FAMS[which.min(b)], bic_NO = b[1], bic_SHASHo = b[2])
  }))
  fam_tab[, set := set]
  fam_all[[set]] <- fam_tab
  fam_of <- setNames(fam_tab$family, fam_tab$feature)
  clusterExport(cl, "fam_of", envir = environment())
  cat(set, "families:", paste(names(table(fam_tab$family)), table(fam_tab$family), collapse = " "), " failed:", sum(is.na(fam_tab$family)), "\n")

  conds <- if (set == "thickness") "full" else c("full", "half1", "half2")
  for (cond in conds) {
    dat <- if (set == "thickness") full else load_set(set, cond)
    runs <- if (set != "thickness" && cond == "half1") list(run1 = load_set(set, "run1"), run2 = load_set(set, "run2")) else list()
    runs <- runs[!sapply(runs, function(x) is.null(x) || nrow(x) == 0)]
    clusterExport(cl, c("dat", "runs"), envir = environment())
    res <- parLapply(cl, feats, function(f) {
      fam <- fam_of[[f]]
      if (is.na(fam)) return(NULL)
      out <- list(); tc <- dat[dx == 2]
      for (k in 1:5) {
        m <- fit_one(tc[fold != k], tc[fold != k][[f]], fam, use_fd)
        if (is.null(m)) next
        te <- tc[fold == k]
        out[[length(out) + 1]] <- data.table(cond = "main", cohort = te$cohort, sub_n = te$sub_n, feature = f, z = zscore(m, fam, te, te[[f]]))
        for (rn in names(runs)) {
          rt <- runs[[rn]][dx == 2 & fold == k]
          if (nrow(rt)) out[[length(out) + 1]] <- data.table(cond = rn, cohort = rt$cohort, sub_n = rt$sub_n, feature = f, z = zscore(m, fam, rt, rt[[f]]))
        }
      }
      m <- fit_one(tc, tc[[f]], fam, use_fd)
      if (!is.null(m)) {
        as_ <- dat[dx == 1]
        out[[length(out) + 1]] <- data.table(cond = "main", cohort = as_$cohort, sub_n = as_$sub_n, feature = f, z = zscore(m, fam, as_, as_[[f]]))
        for (rn in names(runs)) {
          ra <- runs[[rn]][dx == 1]
          if (nrow(ra)) out[[length(out) + 1]] <- data.table(cond = rn, cohort = ra$cohort, sub_n = ra$sub_n, feature = f, z = zscore(m, fam, ra, ra[[f]]))
        }
      }
      rbindlist(out)
    })
    res <- rbindlist(res)
    fwrite(res[cond == "main", .(cohort, sub_n, feature, z)], file.path(outd, sprintf("B12_z_%s_%s.csv.gz", set, cond)))
    for (rn in names(runs)) fwrite(res[cond == rn, .(cohort, sub_n, feature, z)], file.path(outd, sprintf("B12_z_%s_%s.csv.gz", set, rn)))
    cat(set, cond, "rows:", res[cond == "main", .N], " runs:", paste(names(runs), sapply(names(runs), function(rn) res[cond == rn, .N]), collapse = " "), "\n")
  }
}
fwrite(rbindlist(fam_all), file.path(outd, "B12_families.csv"))
stopCluster(cl)
cat("DONE\n")
