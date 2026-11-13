suppressPackageStartupMessages(library(lme4))
d <- read.csv("glmm_input.csv", encoding = "UTF-8"); q <- subset(d, g == "C" & model == "qwen")
q$strategy <- relevel(factor(q$strategy), ref = "zero-shot"); q$said_false <- as.integer(q$predicted_label == "False")
for (y in c("correct","said_false")) {
  m0 <- glm(as.formula(paste(y,"~ strategy")), data=q, family=binomial)
  for (k in c(1, 10)) {
    m <- glmer(as.formula(paste(y,"~ strategy + (1|verb)")), data=q, family=binomial, nAGQ=k, control=glmerControl(optimizer="bobyqa"))
    b <- coef(summary(m))["strategydap",]
    cat(sprintf("%-10s nAGQ=%2d  OR=%.3g [%.3g, %.3g]  동사 SD=%.2f\n", y, k, exp(b[1]), exp(b[1]-1.96*b[2]), exp(b[1]+1.96*b[2]), sqrt(VarCorr(m)$verb[1])))
  }
  m1 <- glmer(as.formula(paste(y,"~ strategy + (1|verb)")), data=q, family=binomial, control=glmerControl(optimizer="bobyqa"))
  lr <- as.numeric(2*(logLik(m1)-logLik(m0))); cat(sprintf("   랜덤절편 LRT chi2=%.1f, p(경계 보정)=%.2g\n", lr, pchisq(lr,1,lower.tail=FALSE)/2))
}
