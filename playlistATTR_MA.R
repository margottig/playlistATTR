# ML Predictive Model On Wheter Like/Dislike Music From A Spotify Playlist
# Name: Marcelo Argotti
# Date: 6/20/2020


################################################################################
#chech if packages needed are installed properly
# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", 
                                         repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", 
                                     repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", 
                                          repos = "http://cran.us.r-project.org")
if(!require(ggcorrplot)) install.packages("ggcorrplot", 
                                          repos = "http://cran.us.r-project.org")
if(!require(rattle)) install.packages("rattle", 
                                      repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", 
                                     repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("rpart.plot", 
                                          repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", 
                                     repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", 
                                     repos = "http://cran.us.r-project.org")
if(!require(class)) install.packages("class", 
                                     repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", 
                                            repos = "http://cran.us.r-project.org")
if(!require(MASS)) install.packages("MASS", 
                                    repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", 
                                         repos="http://cran.us.r-project.org")

#load dataset from GitHub repo
# SpotifyClassification 2017 dataset:
# https://github.com/margottig/spotyATR/archive/master.zip

temp <- tempfile()
download.file("https://github.com/margottig/spotyATR/archive/master.zip", temp)

#read downloaded data
data <- read.csv(unz(temp, "spotyATR-master/data.csv"))
unlink(temp)

#get insights into the dataset
str(data)

#Some summary statistics
summary(data)

#check missing values
colSums(is.na(data)) 

#Let see if there is repeated data. Count unique values in the X variable (song_id)
length(unique(data$X)) 

#Let explore how audio features correlate between them or else see which of the features
#correlate the best with the target variable (liked/disliked song)
corr <- round(cor(data[,2:15]),6)
ggcorrplot(corr)


#Transform categorical variables into factors in order to represent data more
# efficiently. Before converting this variables, lets make a copy of the data.
playlist <- data
playlist$target <- factor(playlist$target, levels = c(0,1), labels = c("dislike", "like"))

data$target <- factor(data$target, levels = c(0,1), labels = c("dislike", "like"))
data$mode <- factor(data$mode, levels=c(0,1), labels=c("minor", "major"))
data$key <- factor(playlist$key)
data$duration_ms <- playlist$duration_ms/60000 #convert ms to minutes


#It is know from pitch class integer notation that each number from the key variable
# are relative to a specific letter in elementary music theory. So now we convert 
#the numerical keys to the actual musical keys
levels(data$key)[1] <- "C"
levels(data$key)[2] <- "C#"
levels(data$key)[3] <- "D"
levels(data$key)[4] <- "D#"
levels(data$key)[5] <- "E"
levels(data$key)[6] <- "F"
levels(data$key)[7] <- "F#"
levels(data$key)[8] <- "G"
levels(data$key)[9] <- "G#"
levels(data$key)[10] <- "A"
levels(data$key)[11] <- "A#"
levels(data$key)[12] <- "B"

#convert the duration_ms units to minutes
playlist$duration_ms <- playlist$duration_ms/60000

# identify which class applies for each variable
sapply(playlist, class) 

# At the moment we will not be using character variables, so lets skipped them 
#from our dataset including the song_id variable 'X'
playlist$X <- NULL
playlist$song_title <- NULL
playlist$artist <- NULL



# histogram of target variable
playlist %>% ggplot(aes(target)) + geom_bar(width = 0.4,fill="steelblue")


# Let's see how likely is the musical taste among key notation
#Scattered plot
KeyNotation <- c("C","C#","D", "D#", "E", "F", "G", "G#", "A","A#", "B")
Taste <- c("like", "dislike")
data %>% filter(key%in%KeyNotation & target%in%Taste) %>%
  ggplot(aes(energy, loudness, col = target)) +
  geom_point() + facet_wrap(~key) 


#Let see which key notes are most predominant in data
# histogram key variable
ggplot(data) + geom_bar(aes(key),width = 0.4,fill="steelblue")


###################### DENSITY PLOTS   ########################3
# energy density
plot_energy <- data %>% ggplot(aes(energy, fill = target)) + 
  geom_density(alpha=0.4) + facet_grid(mode~.)
# loudness density
plot_loudness <- data %>% ggplot(aes(loudness, fill = target)) +
  geom_density(alpha=0.4) + facet_grid(mode~.)
# danceability density
plot_danceability <- data %>% ggplot(aes(danceability, fill = target)) +
  geom_density(alpha=0.4) + facet_grid(mode~.)
# acousticness density
plot_acousticness <- data %>% ggplot(aes(acousticness, fill = target)) +
  geom_density(alpha=0.4) +  facet_grid(mode~.)
#valence density
plot_valence <- data %>% ggplot(aes(valence, fill = target)) +
  geom_density(alpha=0.4) +  facet_grid(mode~.)
# speechiness density
plot_speechiness <- data %>% ggplot(aes(speechiness, fill = target)) +
  geom_density(alpha=0.4) +  facet_grid(mode~.)
#liveness density
plot_liveness <- data %>% ggplot(aes(liveness, fill = target)) +
  geom_density(alpha=0.4) +  facet_grid(mode~.)
# tempo density
plot_tempo <- data %>% ggplot(aes(tempo, fill = target)) +
  geom_density(alpha=0.4) +  facet_grid(mode~.)
#combine plots
grid.arrange(plot_energy, plot_danceability, plot_valence, plot_acousticness)
grid.arrange(plot_loudness, plot_tempo, plot_speechiness, plot_liveness)


########## MODELING APPROACH - DATA PARTITION  ################

#Create Data partition into test set and training set 80% and 20%
set.seed(1)
test_index <- createDataPartition(y = playlist$target, times = 1, p = 0.80, 
                                  list = FALSE)
Train <- playlist[-test_index,]
Test <- playlist[test_index,]

#scale data
ScaledTrain <- Train
ScaledTest <- Test

ScaledTrain[, -14]=scale(ScaledTrain[, -14])
ScaledTest[, -14]=scale(ScaledTest[, -14])

# Applying the Decision Tree model:
DecisionT <- rpart(target~., data=ScaledTrain)
# show decision tree plot
prp(DecisionT, type=1,extra=4, main= "Probabilities per class") 
dt_pred <- predict(DecisionT, ScaledTest, type= "class")
cm_decisiontree = confusionMatrix(dt_pred, ScaledTest$target, positive= "like")
cm_decisiontree


# Applying the Logistic Regression model:
LogicRegression = glm(formula = target ~ ., family = binomial,data = ScaledTrain)
# Predicting the Test set results
lr_pred = predict(LogicRegression, type = 'response', newdata = ScaledTest)
y_pred = ifelse(lr_pred > 0.5, "like", "dislike") %>% factor
# Making the Confusion Matrix
cm_glm = confusionMatrix(y_pred, ScaledTest$target, positive = "like")
cm_glm


#Applying the kNN model with k = 5:
# Fitting K-NN to the Training set and Predicting the Test set results
Knearest = knn(train = ScaledTrain[, -14],
               test = ScaledTest[, -14],
               cl = ScaledTrain[, 14],
               k = 5,
               prob = TRUE)
cm_knn = confusionMatrix(Knearest, ScaledTest$target, positive = "like")
cm_knn


#Applying SVM model
# Fitting SVM to the Training set
SVM = svm(formula = target ~ .,
          data = ScaledTrain,
          type = 'C-classification',
          kernel = 'linear')

# Predicting the Test set results
svm_pred = predict(SVM, newdata = ScaledTest[-14])

# Making the Confusion Matrix
cm_svm = confusionMatrix(svm_pred, ScaledTest$target, positive = "like")
cm_svm


# Applying the Naive Bayes model:
NBay = naiveBayes(x = ScaledTrain[-14], y = ScaledTrain$target)
# Predicting the Test set results
nbay_pred = predict(NBay, newdata = ScaledTest[-14])
# Making the Confusion Matrix
cm_naivebayes = confusionMatrix(nbay_pred, ScaledTest$target, positive = "like")
cm_naivebayes


#Applying the Random Forest model, with number of trees limit set to 777:
# Fitting Random Forest Classification to the Training set
RndForest = randomForest(x = ScaledTrain[-14],
                         y = ScaledTrain$target,
                         ntree = 777)
# Predicting the Test set results
rndforest_pred = predict(RndForest, newdata = ScaledTest[-14])
# Making the Confusion Matrix
cm_randomforest = confusionMatrix(rndforest_pred, ScaledTest$target, positive = "like")
cm_randomforest


#Applying the Linear Discriminant Analysis (LDA) model:
train_set = ScaledTrain
test_set = ScaledTest
# Applying LDA
LDA = lda(formula = target ~ ., data = train_set)
train_set = as.data.frame(predict(LDA, train_set))
train_set = train_set[c(4, 1)]
test_set = as.data.frame(predict(LDA, test_set))
test_set = test_set[c(4, 1)]
# Fitting SVM to the Training set
# install.packages('e1071')
classifier_lda = svm(formula = class ~ .,data = train_set,
                     type = 'C-classification', kernel = 'linear')
# Predicting the Test set results
lda_pred = predict(classifier_lda, newdata = test_set[-2])
# Making the Confusion Matrix
cm_lda = confusionMatrix(lda_pred, test_set$class, positive = "like")
cm_lda


################## RESULTS  ################
# Comparison of confusion matrix of used models
confusionmatrix_list <- list(DT = cm_decisiontree,GLM = cm_glm, KNN = cm_knn,
                             SVM = cm_svm, NV = cm_naivebayes,
                             RF= cm_randomforest,LDA = cm_lda)   
# confusionmatrix_list
confusionmatrix_list_results <- sapply(confusionmatrix_list, function(x) x$byClass)
confusionmatrix_list_results %>% knitr::kable()


