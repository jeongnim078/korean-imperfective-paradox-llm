# glmm_slope_check.R — 동사 랜덤효과 구조 비교 (Qwen30B 본실험, 그룹 C, Zero-shot vs DAP)
suppressPackageStartupMessages(library(lme4))
d <- read.csv("glmm_input.csv", encoding = "UTF-8")
c2 <- subset(d, g == "C" & model == "qwen" & strategy %in% c("zero-shot", "dap"))
c2$dap <- as.integer(c2$strategy == "dap")
c2$said_false <- as.integer(c2$predicted_label == "False")
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
cat("관측", nrow(c2), "| 동사", length(unique(c2$verb)), "| 동사당 관측", nrow(c2)/length(unique(c2$verb)), "\n")

fit_all <- function(y) {
  f <- as.formula(paste(y, "~ dap"))
  m0 <- glm(f, data = c2, family = binomial)
  m1 <- glmer(as.formula(paste(y, "~ dap + (1|verb)")), data = c2, family = binomial, control = ctrl)
  m2 <- tryCatch(glmer(as.formula(paste(y, "~ dap + (1 + dap|verb)")), data = c2, family = binomial, control = ctrl),
                 error = function(e) { cat("  랜덤기울기 모형 오류:", conditionMessage(e), "\n"); NULL })
  m2u <- tryCatch(glmer(as.formula(paste(y, "~ dap + (1|verb) + (0 + dap|verb)")), data = c2, family = binomial, control = ctrl),
                  error = function(e) NULL)
  row <- function(name, m) {
    b <- if (inherits(m, "glm") && !inherits(m, "glmerMod")) coef(summary(m))["dap", ] else coef(summary(m))["dap", ]
    vc <- if (inherits(m, "glmerMod")) paste(sprintf("%s=%.2f", names(unlist(lapply(VarCorr(m), function(v) diag(v)))), unlist(lapply(VarCorr(m), function(v) diag(v)))), collapse = ", ") else "-"
    sing <- if (inherits(m, "glmerMod")) isSingular(m) else NA
    conv <- if (inherits(m, "glmerMod")) length(m@optinfo$conv$lme4$messages) > 0 else NA
    data.frame(model = name, OR = exp(b[1]), SE = b[2], p = b[4], AIC = AIC(m), 분산 = vc, singular = sing, 수렴경고 = conv)
  }
  out <- rbind(row("① 랜덤효과 없음", m0), row("② 랜덤절편", m1))
  if (!is.null(m2)) out <- rbind(out, row("③ 절편+기울기(상관)", m2))
  if (!is.null(m2u)) out <- rbind(out, row("④ 절편+기울기(무상관)", m2u))
  print(out, digits = 3, row.names = FALSE)
  # 우도비 검정
  lrt1 <- as.numeric(2 * (logLik(m1) - logLik(m0)))
  cat(sprintf("  LRT ①→② (랜덤절편 필요?): chi2=%.2f, p≈%.3g (경계값 보정 p/2=%.3g)\n", lrt1, pchisq(lrt1, 1, lower.tail = FALSE), pchisq(lrt1, 1, lower.tail = FALSE)/2))
  if (!is.null(m2)) { a <- anova(m1, m2); cat(sprintf("  LRT ②→③ (랜덤기울기 필요?): chi2=%.2f, df=%d, p=%.3g\n", a$Chisq[2], a$Df[2], a$`Pr(>Chisq)`[2])) }
  if (!is.null(m2u)) { a <- anova(m1, m2u); cat(sprintf("  LRT ②→④: chi2=%.2f, df=%d, p=%.3g\n", a$Chisq[2], a$Df[2], a$`Pr(>Chisq)`[2])) }
  invisible(list(m1 = m1, m2 = m2))
}
cat("\n=== 결과변수: 정답 여부 ===\n"); r1 <- fit_all("correct")
cat("\n=== 결과변수: '거짓' 응답 여부 ===\n"); r2 <- fit_all("said_false")
# 참고: 동사별 원자료 패턴 (Zero-shot→DAP 조합 빈도)
tab <- with(reshape(c2[, c("verb", "strategy", "correct")], idvar = "verb", timevar = "strategy", direction = "wide"),
            table(ZS = `correct.zero-shot`, DAP = correct.dap))
cat("\n동사별 정답 조합 (행: Zero-shot, 열: DAP)\n"); print(tab)
