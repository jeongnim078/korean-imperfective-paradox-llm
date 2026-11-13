suppressPackageStartupMessages(library(lme4))
d <- read.csv("glmm_input.csv", encoding = "UTF-8")
q <- subset(d, g == "C" & model == "qwen")
q$strategy <- relevel(factor(q$strategy), ref = "zero-shot")
q$dap <- as.integer(q$strategy == "dap"); q$said_false <- as.integer(q$predicted_label == "False")
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
show <- function(name, m, term) {
  b <- coef(summary(m))[term, ]; ci <- exp(b[1] + c(-1.96, 1.96) * b[2])
  vc <- unlist(lapply(VarCorr(m), function(v) diag(v)))
  cat(sprintf("%-34s OR=%.3g [%.3g, %.3g] p=%.2g | 분산 %s | singular=%s | 수렴경고=%s\n", name, exp(b[1]), ci[1], ci[2], b[4],
      paste(sprintf("%.2f", vc), collapse = "/"), isSingular(m), length(m@optinfo$conv$lme4$messages) > 0))
}
cat("=== 전략 4개 모형 (동사당 관측 4개) ===\n")
for (y in c("correct", "said_false")) {
  m1 <- glmer(as.formula(paste(y, "~ strategy + (1|verb)")), data = q, family = binomial, control = ctrl)
  m2 <- glmer(as.formula(paste(y, "~ strategy + (1|verb) + (0 + dap|verb)")), data = q, family = binomial, control = ctrl)
  show(paste(y, "랜덤절편"), m1, "strategydap"); show(paste(y, "절편+DAP기울기(무상관)"), m2, "strategydap")
  a <- anova(m1, m2); cat(sprintf("   LRT 기울기 추가: chi2=%.2f, p=%.3g\n", a$Chisq[2], a$`Pr(>Chisq)`[2]))
}
cat("\n=== 조건부 로지스틱 = 정확 McNemar (Zero-shot vs DAP 짝 비교) ===\n")
w <- reshape(q[q$strategy %in% c("zero-shot", "dap"), c("verb", "strategy", "correct", "said_false")], idvar = "verb", timevar = "strategy", direction = "wide")
for (y in c("correct", "said_false")) {
  zs <- w[[paste0(y, ".zero-shot")]]; dp <- w[[paste0(y, ".dap")]]
  b <- sum(zs == 0 & dp == 1); cc <- sum(zs == 1 & dp == 0)
  bt <- binom.test(b, b + cc)
  # 조건부 OR = b/c, 정확 CI는 이항 비율 CI에서 변환
  ci <- bt$conf.int; orci <- ci / (1 - ci)
  cat(sprintf("%-11s DAP에서만 1: %d, Zero-shot에서만 1: %d → 조건부 OR=%.3g [%.3g, %.3g], 정확 p=%.2g\n", y, b, cc, b / max(cc, 0.5), orci[1], orci[2], bt$p.value))
}
