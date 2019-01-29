############################ SVM Digit Recogniser #################################
# 1. Business Understanding
# 2. Data Understanding
# 3. Data Preparation
# 4. Model Building 
#  4.1 Linear kernel
#  4.2 RBF Kernel
# 5 Hyperparameter tuning and cross validation

#####################################################################################

# 1. Business Understanding: 

#The goal is to develop a model that can correctly identify 
#the digit (between 0-9) written in an image. 


#####################################################################################

# 2. Data Understanding: 
# in mnist_train
# Number of Instances: 60,000
# Number of Attributes: 785 
# in mnist_test
# Number of Instances: 10,000
# Number of Attributes: 785 

#3. Data Preparation: 

#Loading the Required files
library(kernlab)
library(readr)
library(caret)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(caTools)
#Loading train and test Data
digit_all_train<- read.csv("mnist_train.csv",stringsAsFactors = FALSE,header=FALSE)
digit_test<- read.csv("mnist_test.csv",stringsAsFactors=FALSE,header= FALSE)

#Vieing the data
View(digit_all_train)
View(digit_test)
#Both dataset have no column names
colnames(digit_all_train)[1]<-"digit"
colnames(digit_test)[1]<-"digit"

#Structure of the dataset

str(digit_all_train)
str(digit_test)

# Checking for Missing Values if any
sum(sapply(digit_all_train, function(x) sum(is.na(x)))) # No missing values
sum(sapply(digit_test, function(x) sum(is.na(x)))) # No missing values

# Convert digit variable into factor

digit_all_train$digit<-as.factor(digit_all_train$digit)
digit_test$digit<-as.factor(digit_test$digit)

#Checking
class(digit_all_train$digit)
class(digit_test$digit)

#As the data is too big and running the code takes long time,Using only 15% of 
#the data set provided

set.seed(100)

indices <- sample(1: nrow(digit_all_train), 0.15*nrow(digit_all_train))
train <- digit_all_train[indices, ]


#***************************Constructing Model******************************#

#****************************Using Linear Kernel***************************#

Model_linear <- ksvm(digit~ ., data = train, scale = FALSE, kernel = "vanilladot")
print(Model_linear)
#C=1
Eval_linear<- predict(Model_linear, digit_test)

#confusion matrix - Linear Kernel
confusionMatrix(Eval_linear,digit_test$digit)

#***********************************************#
#Overall Accuracy:91.82%
#sensitivity > 84% 
#specificity > 99% 
#***********************************************#

#Hyper parameter tuning and cross validation
#range of C
grid_linear <- expand.grid(C= c(0.5,1,1.5))

#Cross validation and number of folds
trainControl <- trainControl(method="cv", number=5)

#Evaluvation metric is Accuracy
metric <- "Accuracy"

fit_linear <- train(digit~., data=train, method="svmLinear", metric=metric, 
                 tuneGrid=grid_linear, trControl=trainControl)

print(fit_linear)
#Optimum value of C=0.5

plot(fit_linear)
#Checking with C=0.5

Model_linear1 <- ksvm(digit~ ., data = train, scale = FALSE, kernel = "vanilladot",C=0.5)
Eval_linear1<- predict(Model_linear1, digit_test)
confusionMatrix(Eval_linear1,digit_test$digit)
#***********************************#
#Overall Accuracy:91.82%
#Sensitivity >84%
#Specificity> 99%
#**********************************
#Same value as c=1,No much variation found.

#***************************Using RBF Kernel******************************#
Model_RBF <- ksvm(digit~ ., data = train, scale = FALSE, kernel = "rbfdot")

print(Model_RBF)
#sigma=  1.62e-07 
#c=1
Eval_RBF<- predict(Model_RBF, digit_test)

#confusion matrix - RBF Kernel
confusionMatrix(Eval_RBF,digit_test$digit)

#***********************************************#
#Overall Accuracy:95.54%
#sensitivity > 93% high
#specificity > 99% high
#***********************************************#

#Hyper parameter tuning and cross validation
#range of C and sigma
set.seed(7)
grid_rbf <- expand.grid(.sigma=c(1.0e-07,1.62e-07,2.0e-07), .C=c(0.5,1,1.5) )

#Cross validation and number of folds
trainControl <- trainControl(method="cv", number=5)

#Evaluvation metric is Accuracy
metric <- "Accuracy"

rbf_fit <- train(digit~., data=train, method="svmRadial", metric=metric, 
                    tuneGrid=grid_rbf, trControl=trainControl)

print(rbf_fit)
#Sigma =2e-07,C=1.5
plot(rbf_fit)

#Checking with C=1.5,sigma=2e-07
Eval_RBF1 <- predict(rbf_fit, newdata = digit_test)
confusionMatrix(Eval_RBF1, digit_test$digit)


#Accuracy is highest at c=1.5 and sigma=2e-07
#Accuracy of 96.13%
#Sensitivity > 93%
#Specificity > 99

#Final Model
final_model = rbf_fit

#SVM using RBF Kernel (c=1.5 and sigma=2e-07) achieved highest accuracy in predicting digits

