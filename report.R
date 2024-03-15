install.packages('readr', dependencies = TRUE, repos='http://cran.rstudio.com/')
install.packages('skimr')
install.packages('tidyverse')
install.packages('ggplot2')
install.packages('caret')
install.packages('randomForest')

# load dataset
bank <- readr::read_csv("https://www.louisaslett.com/Courses/MISCADA/bank_personal_loan.csv")

# data summary and visualization
library('skimr')
library("tidyverse")
library("ggplot2")


skim(bank)

bank %>%
  ggplot(aes(x=Personal.Loan,y= Income,color= Personal.Loan))+
  geom_boxplot()

bank %>%
  ggplot(aes(x=Personal.Loan,y= Mortgage,color= Personal.Loan))+
  geom_boxplot()

# Reset negative values to absolute in the variable "Experience" 
bank$Experience <- abs(bank$Experience)
# change type from numeric to factor
bank$Personal.Loan <- as.factor(bank$Personal.Loan)
bank$Education <- as.factor(bank$Education)
bank$Securities.Account <- as.factor(bank$Securities.Account)
bank$CD.Account <- as.factor(bank$CD.Account)
bank$Online <- as.factor(bank$Online)
bank$CreditCard <- as.factor(bank$CreditCard)

# split train/test data
library('caret')
set.seed(1)
index <- createDataPartition(bank$Personal.Loan, p = 0.7, list = FALSE)
train <- bank[index,]
test  <- bank[-index,]

## Logistic Regression
lr_model <- glm(Personal.Loan ~.,data = train, family = 'binomial')
summary(lr_model)
# simplified
lr_model_step <- step(lr_model,direction = 'backward',trace = 0)
summary(lr_model_step)
# predict
predicted <- predict(lr_model_step,newdata = test,type = 'response')
test$predicted <- predicted
test$class <- ifelse(test$predicted >=0.5,1,0)
str(test$class)
test$class<- as.factor(test$class)
confusionMatrix(test$class,test$Personal.Loan,positive = "1")

## Decision Tree
library(rpart)
set.seed(1)
dt_model <- rpart(Personal.Loan~.,data = train)
predicted_dt <- predict(dt_model,newdata = test,type = 'class')
test$predicted_dt <- predicted_dt
str(test$predicted_dt)
confusionMatrix(test$predicted_dt,test$Personal.Loan,positive = "1")

## Random Forest
library(randomForest)
set.seed(1)
rf_model <- randomForest(Personal.Loan~.,data = train)
predicted_rf <- predict(rf_model,newdata = test,type = 'class')
test$predicted_rf <- predicted_rf
str(test$predicted_rf)
confusionMatrix(test$predicted_rf,test$Personal.Loan,positive = "1")

## AUC plot
library(pROC)
par(pty = 's')
roc(test$Personal.Loan,test$predicted,
    plot = TRUE,percent = TRUE, 
    legacy.axes = TRUE,xlab = "False Positive (%)", 
    ylab = "True Positive (%)",lwd = 1 ,
    print.auc= TRUE,main = "AUC of Logistic Regression")


set.seed(1)
predicted_dt <- predict(dt_model,newdata = test,type = 'prob')
test$predicted_dt <- predicted_dt[,2]
par(pty = 's')
roc(test$Personal.Loan,test$predicted_dt, 
    plot = TRUE,percent = TRUE, 
    legacy.axes = TRUE,xlab = "False Positive (%)", 
    ylab = "True Positive (%)",lwd = 1 ,
    print.auc= TRUE,main = "AUC of Decision Tree")

set.seed(1)
predicted_rf <- predict(rf_model,newdata = test,type = 'prob')
test$predicted_rf <- predicted_rf[,2]
par(pty = 's')
roc(test$Personal.Loan,test$predicted_rf,
    plot = TRUE,percent = TRUE, 
    legacy.axes = TRUE,xlab = "False Positive (%)", 
    ylab = "True Positive (%)",lwd = 1 ,
    print.auc= TRUE,main = "AUC of Random Forest")



set.seed(123)
seeds <- vector(mode = "list", length = 51)
for(i in 1:50) seeds[[i]] <- sample.int(1000, 22)

## For the last model:
seeds[[51]] <- sample.int(1000, 1)

ctrl <- trainControl(method = "repeatedcv",
                     repeats = 5,
                     seeds = seeds)

set.seed(1)
cv_model <- train(Personal.Loan ~ .,           
                  data = train,          
                  method = "rf",  
                  tuneLength = 12,
                  trControl = ctrl) 


