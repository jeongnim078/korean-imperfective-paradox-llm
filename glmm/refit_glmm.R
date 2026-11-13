# refit_glmm.R
# 초록 회귀 수치 재검증용: Gemini 제외, 기준범주 명시, (1|verb) 랜덤절편을 실제로 추정하는 GLMM.
# 사용: Rscript refit_glmm.R glmm_input.csv
suppressPackageStartupMessages(library(lme4))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(if (length(args)) args[1] else "glmm_input.csv", encoding = "UTF-8")
d$strategy <- relevel(factor(d$strategy), ref = "zero-shot")
d$model <- relevel(factor(d$model), ref = "gpt")
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

or_table <- function(fit, label) {
  cf <- summary(fit)$coefficients
  ci <- suppressMessages(confint(fit, parm = "beta_", method = "Wald"))
  out <- data.frame(model_spec = label, term = rownames(cf),
                    coef = cf[, 1], se = cf[, 2], p = cf[, 4],
                    OR = exp(cf[, 1]), OR_low = exp(ci[, 1]), OR_high = exp(ci[, 2]),
                    singular = isSingular(fit), row.names = NULL)
  out
}

res <- list()

# M1. Qwen, 그룹 C: 전략 효과 (기준 = zero-shot)
c_q <- subset(d, g == "C" & model == "qwen")
m1 <- glmer(correct ~ strategy + (1 | verb), data = c_q, family = binomial, control = ctrl)
res[[1]] <- or_table(m1, "M1 qwen | group C | correct ~ strategy + (1|verb)")

# M2. Qwen, 그룹 C, 오답 방향: FALSE 응답(과신) 여부
c_q$said_false <- as.integer(c_q$predicted_label == "False")
m2 <- glmer(said_false ~ strategy + (1 | verb), data = c_q, family = binomial, control = ctrl)
res[[2]] <- or_table(m2, "M2 qwen | group C | said_False ~ strategy + (1|verb)")

# M3. Qwen, 그룹 C, DAP vs zero-shot 내 2차 라벨 상호작용
c_q2 <- subset(c_q, strategy %in% c("zero-shot", "dap"))
c_q2$strategy <- droplevels(c_q2$strategy)
c_q2$sec <- relevel(factor(c_q2$sec), ref = "완성동사")
m3 <- glmer(correct ~ strategy * sec + (1 | verb), data = c_q2, family = binomial, control = ctrl)
res[[3]] <- or_table(m3, "M3 qwen | group C | zero-shot vs dap | correct ~ strategy*secondary + (1|verb)")

# M4. Qwen, 그룹 D: 전략 효과 (CoT/Counterfactual 붕괴 확인)
d_q <- subset(d, g == "D" & model == "qwen")
m4 <- glmer(correct ~ strategy + (1 | verb), data = d_q, family = binomial, control = ctrl)
res[[4]] <- or_table(m4, "M4 qwen | group D | correct ~ strategy + (1|verb)")

out <- do.call(rbind, res)
write.csv(out, "glmm_refit.csv", row.names = FALSE, fileEncoding = "UTF-8")
print(out[, c("model_spec", "term", "OR", "OR_low", "OR_high", "p", "singular")], digits = 3)
