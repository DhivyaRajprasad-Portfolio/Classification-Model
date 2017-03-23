rm(list=ls())
#Packages Used
library(ggplot2) 
library(car)
library("verification")
library(ROCR)
library(rpart)
library(rattle)
library(mgcv)
library(e1071)
#Data 

german_credit = 
  read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")

colnames(german_credit) = c("chk_acct", "duration", "credit_his", 
                            "purpose", "amount", "saving_acct", 
                            "present_emp",   "installment_rate", 
                            "sex", "other_debtor", "present_resid", 
                            "property", "age", "other_install", 
                            "housing", "n_credits",  "job", 
                            "n_people", "telephone", "foreign", "response")

german_credit$response = german_credit$response - 1

#EDA
#Density Plots
german_credit$response <- factor(german_credit$response)

ggplot(data = melt(german_credit), aes(x = value, color= response)) +
  geom_density(fill="white",alpha=0.55) +
  facet_wrap(~variable, scales = "free")

#Correlation Plots
c <- cor(german_credit[sapply(german_credit, is.numeric)], use="complete.obs", method="pearson")
c

#Pie Chart
pie(table(german_credit$response),c("good","bad"))

#Sampling
set.seed(10857825)
subset <- sample(nrow(german_credit), nrow(german_credit) * 0.75)
credit.train = german_credit[subset, ]
credit.test = german_credit[-subset, ]


#LOGISTIC REGRESSION

#Logit function
creditmod <-glm(response~., family = binomial, credit.train)
summary(creditmod)
AIC(creditmod)
BIC(creditmod)
vif(creditmod)

#Probit Function
probitmod <- glm(response~., family = binomial(link="probit"), data=credit.train)
summary(probitmod)
AIC(probitmod)
BIC(probitmod)
vif(probitmod)
library(car)
#Log Log Function
loglogmod<-glm(response~., family = binomial(link="cloglog"), data=credit.train)
summary(loglogmod)
AIC(loglogmod)
BIC(loglogmod)
vif(loglogmod)

#Model Selection-AIC
creditmod.step <- step(creditmod)
summary(creditmod.step)
hist(predict(creditmod.step, type = "response"))
par(mfrow=c(2,2))
plot(creditmod.step)

#Model Selection-BIC
creditmod.BIC <- step(creditmod, k = log(nrow(credit.train)))
summary(creditmod.BIC)
hist(predict(creditmod.BIC, type = "response"))
#insample
prob.glm1.insample <- predict(creditmod.step, type = "response")
predicted.glm1.insample <- prob.glm1.insample > 0.1666
predicted.glm1.insample <- as.numeric(predicted.glm1.insample)
#Confusion Matrix
(table(credit.train$response, predicted.glm1.insample, dnn = c("Truth", "Predicted")))
#Error Rate
mean(ifelse(credit.train$response != predicted.glm1.insample, 1, 0))

roc.plot(credit.train$response == "1", prob.glm1.insample)
roc.plot(credit.train$response == "1", prob.glm1.insample)$roc.vol

pred <- prediction(predicted.glm1.insample, credit.train$response)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize = TRUE)
auc.perf = performance(pred, measure = "auc")
auc.perf@y.values
creditcost(credit.train$response, predicted.glm1.insample)

#outsample
prob.glm1.outsample <- predict(creditmod.step, credit.test, type = "response")
predicted.glm1.outsample <- prob.glm1.outsample > 0.1666
predicted.glm1.outsample <- as.numeric(predicted.glm1.outsample)

table(credit.test$response, predicted.glm1.outsample, dnn = c("Truth", "Predicted"))

mean(ifelse(credit.test$response != predicted.glm1.outsample, 1, 0))

library("verification")
roc.plot(credit.test$response == "1", predicted.glm1.outsample)
roc.plot(credit.test$response == "1", predicted.glm1.outsample)$roc.vol


library(ROCR)
pred <- prediction(prob.glm1.outsample, credit.test$response)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize = TRUE)
abline(a=0, b= 1)
auc.perf = performance(pred, measure = "auc")
auc.perf@y.values
creditcost(credit.test$response, predicted.glm1.outsample)

#Cross- Validation
searchgrid = seq(0.01, 0.6, 0.02)
result = cbind(searchgrid, NA)
cost1 <- function(r, pi) {
  weight1 = 5
  weight0 = 1
  c1 = (r == 1) & (pi < pcut)  #logical vector - true if actual 1 but predict 0
  c0 = (r == 0) & (pi > pcut)  #logical vecotr - true if actual 0 but predict 1
  return(mean(weight1 * c1 + weight0 * c0))
}
credit.glm1 <- glm(response ~ chk_acct + duration + credit_his + purpose + amount + saving_acct + present_emp + installment_rate +other_debtor +  age + other_install + housing + foreign, family = binomial, german_credit)
for (i in 1:length(searchgrid)) {
  pcut <- result[i, 1]
  result[i, 2] <- cv.glm(data = german_credit, glmfit = credit.glm1, cost = cost1, 
                         K = 3)$delta[2]
}
plot(result, ylab = "CV Cost")


result[which.min((result[,2])),]
#ROC for cross validation
prob.glm1.insample <- predict(creditmod.step, type = "response")
predicted.glm1.insample <- prob.glm1.insample > 0.17
predicted.glm1.insample <- as.numeric(predicted.glm1.insample)
#Confusion Matrix
(table(credit.train$response, predicted.glm1.insample, dnn = c("Truth", "Predicted")))
#Error Rate
mean(ifelse(credit.train$response != predicted.glm1.insample, 1, 0))
library("verification")
roc.plot(credit.train$response == "1", prob.glm1.insample)
roc.plot(credit.train$response == "1", prob.glm1.insample)$roc.vol


library(ROCR)
pred <- prediction(predicted.glm1.insample, credit.train$response)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize = TRUE)
auc.perf = performance(pred, measure = "auc")
auc.perf@y.values


#TREES
set.seed(10857825)
subset <- sample(nrow(german_credit), nrow(german_credit) * 0.75)
credit.train = german_credit[subset, ]
credit.test = german_credit[-subset, ]


#Classification Trees
par(mfrow=c(1,1))
credit.rpart <- rpart(formula = response ~ . , data = credit.train, method = "class", 
                      parms = list(loss = matrix(c(0, 5, 1, 0), nrow = 2)))
credit.rpart
plot(credit.rpart)
text(credit.rpart)

#Prediction using Classification Tree- In Sample
credit.train.pred.tree1 = predict(credit.rpart, type = "class")
table(credit.train$response, credit.train.pred.tree1, dnn = c("Truth", "Predicted"))
cost <- function(r, pi) {
  weight1 = 5
  weight0 = 1
  c1 = (r == 1) & (pi == 0)  #logical vector - true if actual 1 but predict 0
  c0 = (r == 0) & (pi == 1)  #logical vecotr - true if actual 0 but predict 1
  return(mean(weight1 * c1 + weight0 * c0))
}
cost(credit.train$response, credit.train.pred.tree1)
mean(ifelse(credit.train$response != credit.train.pred.tree1, 1, 0))
#ROC Curv-In Sample

credit.rpart2 <- rpart(formula = response ~ ., data = credit.train, method = "class", 
                       cp = 5e-04)
# Probability of getting 1
credit.test.prob.rpart2 = predict(credit.rpart2, credit.train, type = "prob")
pred = prediction(credit.test.prob.rpart2[, 2], credit.train$response)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize = TRUE)
slot(performance(pred, "auc"), "y.values")[[1]]
credit.test.pred.rpart2 = as.numeric(credit.test.prob.rpart2[, 2] > 0.1)
table(credit.test$response, credit.test.pred.rpart2, dnn = c("Truth", "Predicted"))

#Prediction using Classification Tree- Out Sample
credit.test.pred.tree1 = predict(credit.rpart, credit.test,type = "class")
table(credit.test$response, credit.test.pred.tree1, dnn = c("Truth", "Predicted"))
cost <- function(r, pi) {
  weight1 = 5
  weight0 = 1
  c1 = (r == 1) & (pi == 0)  #logical vector - true if actual 1 but predict 0
  c0 = (r == 0) & (pi == 1)  #logical vecotr - true if actual 0 but predict 1
  return(mean(weight1 * c1 + weight0 * c0))
}
cost(credit.test$response, credit.test.pred.tree1)
mean(ifelse(credit.train$response != credit.test.pred.tree1, 1, 0))
#ROC Curve-Out Sample
credit.rpart2 <- rpart(formula = response ~ ., data = credit.train, method = "class", 
                       cp = 5e-04)
# Probability of getting 1
credit.test.prob.rpart2 = predict(credit.rpart2, credit.test, type = "prob")
pred = prediction(credit.test.prob.rpart2[, 2], credit.test$response)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize = TRUE)
slot(performance(pred, "auc"), "y.values")[[1]]
credit.test.pred.rpart2 = as.numeric(credit.test.prob.rpart2[, 2] > 0.1)
table(credit.test$response, credit.test.pred.rpart2, dnn = c("Truth", "Predicted"))

##Creditcost
creditcost <- function(observed, predicted) {
  weight1 = 5
  weight0 = 1
  c1 = (observed == 1) & (predicted == 0)  #logical vector - true if actual 1 but predict 0
  c0 = (observed == 0) & (predicted == 1)  #logical vecotr - true if actual 0 but predict 1
  return(mean(weight1 * c1 + weight0 * c0))
}
creditcost(credit.test$response, credit.test.pred.rpart2)
creditcost(credit.train$response, fittedGAM.insample)

#GAM
gam_formula <- as.formula(paste("response~s(duration)+s(amount)+s(age)"))
fitGAM <- gam(formula = gam_formula, family = binomial, data = credit.train)
summary(fitGAM)
par(mfrow=c(2,2))
plot(fitGAM, shade = TRUE,seWithMean = TRUE, scale = 0)

##insample
fittedGAM.insample.prob <- predict(fitGAM)
fittedGAM.insample <- as.numeric(fittedGAM.insample.prob > 0.16)
table(credit.train$response, fittedGAM.insample, dnn = c("Observation", "Prediction"))
mean(ifelse(credit.train$response != fittedGAM.insample, 1, 0))

##out-sample
fittedGAM.outsample.prob <- predict(fitGAM, credit.test)
fittedGAM.outsample <- as.numeric(fittedGAM.outsample.prob > 0.16)
table(credit.test$response, fittedGAM.outsample, dnn = c("Observation", "Prediction"))
mean(ifelse(credit.test$response != fittedGAM.outsample, 1, 0))

plot(fitGAM, shade = TRUE,  seWithMean = TRUE, scale = 0)

##insample AUC
par(mfrow=c(1,1))
roc.plot(credit.train$response == "1", fittedGAM.insample.prob)
roc.plot(credit.train$response == "1", fittedGAM.insample.prob)$roc.vol 
##outsample AUC
roc.plot(credit.test$response == "1", fittedGAM.outsample.prob)
roc.plot(credit.test$response == "1", fittedGAM.outsample.prob)$roc.vol 

##Creditcost
creditcost <- function(observed, predicted) {
  weight1 = 5
  weight0 = 1
  c1 = (observed == 1) & (predicted == 0)  #logical vector - true if actual 1 but predict 0
  c0 = (observed == 0) & (predicted == 1)  #logical vecotr - true if actual 0 but predict 1
  return(mean(weight1 * c1 + weight0 * c0))
}
creditcost(credit.test$response, fittedGAM.outsample)
creditcost(credit.train$response, fittedGAM.insample)

#LDA

fitLDA <- lda(response~ ., data = credit.train)
lda.in <- predict(fitLDA)
pcut.lda <- 0.16
pred.lda.in <- (lda.in$posterior[, 2] >= pcut.lda) * 1
table(credit.train$response, pred.lda.in, dnn = c("Obs", "Pred"))
mean(ifelse(credit.train$response != pred.lda.in, 1, 0))

lda.out <- predict(fitLDA, newdata = credit.test)
cut.lda <- 0.16
pred.lda.out <- as.numeric((lda.out$posterior[, 2] >= cut.lda))
table(credit.test$response, pred.lda.out, dnn = c("Obs", "Pred"))

mean(ifelse(credit.test$response != pred.lda.out, 1, 0)) ##0.352


##insample AUC
roc.plot(credit.train$response == "1", lda.in$posterior[, 2])
roc.plot(credit.train$response == "1", lda.in$posterior[, 2])$roc.vol 
##outsample AUC
roc.plot(credit.test$response == "1", lda.out$posterior[, 2])
roc.plot(credit.test$response == "1", lda.out$posterior[, 2])$roc.vol 


##Creditcost
creditcost <- function(observed, predicted) {
  weight1 = 5
  weight0 = 1
  c1 = (observed == 1) & (predicted == 0)  #logical vector - true if actual 1 but predict 0
  c0 = (observed == 0) & (predicted == 1)  #logical vecotr - true if actual 0 but predict 1
  return(mean(weight1 * c1 + weight0 * c0))
}
creditcost(credit.test$response, pred.lda.out)
creditcost(credit.train$response, pred.lda.in)

#QDA
#SVM
credit.svm = svm(response ~ ., data = credit.train, cost = 1, gamma = 1/length(credit.train), 
                 probability = TRUE)
#In-Sample
prob.svm = predict(credit.svm, credit.train, probability = TRUE)
pred.svm = as.numeric((prob.svm >= 0.16))
table(credit.train$response, pred.svm, dnn = c("Obs", "Pred"))
mean(ifelse(credit.train$response != pred.svm, 1, 0))
creditcost(credit.train$response, pred.svm)

roc.plot(credit.train$response == "1", prob.svm)
roc.plot(credit.train$response == "1", prob.svm)$roc.vol 

#Out of Sample
prob.svm = predict(credit.svm, credit.test, probability = TRUE)
pred.svm = as.numeric((prob.svm >= 0.16))
table(credit.test$response, pred.svm, dnn = c("Obs", "Pred"))
mean(ifelse(credit.test$response != pred.svm, 1, 0))
creditcost(credit.test$response, pred.svm)
roc.plot(credit.test$response == "1", prob.svm)
roc.plot(credit.test$response == "1", prob.svm)$roc.vol 
