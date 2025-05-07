# ---------------------------
#     Supervised Learning
# ---------------------------

# Load all necessary libraries
libraries <- c("tidyverse", "caret", "e1071", "randomForest", "nnet", 
               "xgboost", "ggplot2", "ROCR", "MASS", "mlbench", 
               "reshape2", "gbm", "MLmetrics", "pROC", "rpart.plot")

lapply(libraries, library, character.only = TRUE)

# Load the dataset
obesity_df <- read.csv("data/obesity.csv", header = TRUE)

# Convert the target variable to a factor (multiclass) with correct level order
obesity_df$NObeyesdad <- factor(obesity_df$NObeyesdad,
                                levels = c("Insufficient_Weight", "Normal_Weight",
                                           "Overweight_Level_I", "Overweight_Level_II",
                                           "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"))

# Create dummy variables for predictors (excluding target)
dummies <- dummyVars(NObeyesdad ~ ., data = obesity_df)
df_features <- predict(dummies, newdata = obesity_df) %>% as.data.frame()

# Add the target variable (as factor)
df_features$target <- obesity_df$NObeyesdad

# Split into training and testing sets
set.seed(123)
train_index <- createDataPartition(df_features$target, p = 0.6, list = FALSE)
train_data <- df_features[train_index, ]
test_data  <- df_features[-train_index, ]

# Remove zero variance variables (they don't contribute to prediction)
nzv <- nearZeroVar(train_data, saveMetrics = TRUE)
zero_var_cols <- rownames(nzv[nzv$zeroVar == TRUE, ])

# print the deleted column
cat("Variable with zero variance:", zero_var_cols, "\n")

# remove from train and test
train_data <- train_data[, !names(train_data) %in% zero_var_cols]
test_data  <- test_data[, !names(test_data) %in% zero_var_cols]

# Quick EDA: overview
str(obesity_df)
summary(obesity_df)

# Class distribution
ggplot(obesity_df, aes(x = NObeyesdad)) +
  geom_bar(fill = "limegreen") +
  labs(title = "Distribution of Target Variable", x = "Obesity Level", y = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle=45, hjust=1))

# Numeric variables distribution
obesity_df %>%
  select_if(is.numeric) %>%
  gather(key = "variable", value = "value") %>%
  ggplot(aes(x = value)) +
  geom_histogram(bins = 30, fill = "limegreen", color = "black") +
  facet_wrap(~ variable, scales = "free") +
  theme_minimal() +
  labs(title = "Numeric Variables Distributions")

# Categorical variable
obesity_df %>%
  dplyr::select_if(~!is.numeric(.)) %>%  # select the categorical variable
  gather(key = "variable", value = "value") %>%
  ggplot(aes(x = value)) +
  geom_bar(fill = "limegreen", color = "black") +
  facet_wrap(~ variable, scales = "free", ncol = 2) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle=45, hjust=1)) +
  labs(title = "Categorical Variables Distributions", x = "", y = "Count")

# Distribution of the category in respect to the target
obesity_df %>%
  dplyr::select_if(~!is.numeric(.)) %>%
  gather(key = "variable", value = "value", -NObeyesdad) %>%
  ggplot(aes(x = value, fill = NObeyesdad)) +
  geom_bar(position = "fill") +  
  facet_wrap(~ variable, scales = "free_x", ncol = 2) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle=45, hjust=1)) +
  labs(title = "Proportion of Target by Categorical Variables", y = "Proportion", x = "") +
  scale_fill_manual(
    values = colorRampPalette(c("limegreen", "yellow", "red"))(length(levels(obesity_df$NObeyesdad))),
    name = "Obesity Level"
  )

# Boxplot for numeric variables (to detect outliers visually)
obesity_df %>%
  select_if(is.numeric) %>%
  gather(key = "variable", value = "value") %>%
  ggplot(aes(x = "", y = value)) +
  geom_boxplot(fill = "limegreen") +
  facet_wrap(~ variable, scales = "free") +
  theme_minimal() +
  labs(title = "Boxplots for Numeric Variables", y = "Value", x = "")

# ---------------------------
#            SVM
# ---------------------------

# Train the SVM model (radial kernel) using caret
set.seed(123)
model_svm <- train(
  target ~ ., 
  data = train_data, 
  method = "svmRadial",
  trControl = trainControl(classProbs = TRUE),
  preProcess = c("center", "scale")  # Standardize features
)

# Predict on test set
pred_svm <- predict(model_svm, test_data)

# Confusion Matrix
cat("\nConfusion Matrix - SVM\n")
conf_mat_svm <- confusionMatrix(pred_svm, test_data$target)
print(conf_mat_svm)

# Convert confusion matrix to dataframe for plotting
cm_df_svm <- as.data.frame(conf_mat_svm$table)
colnames(cm_df_svm) <- c("Reference", "Prediction", "Freq")

# Plot Confusion Matrix
ggplot(cm_df_svm, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "black") +
  geom_text(aes(label = Freq), size = 4) +
  scale_fill_gradient(low = "white", high = "purple") +
  labs(
    title = "Confusion Matrix - SVM",
    x = "Actual Class",
    y = "Predicted Class"
  ) +
  theme_minimal()

# Plot variable importance for SVM
var_imp_svm <- varImp(model_svm)$importance
var_imp_svm$Overall <- rowMeans(var_imp_svm)
var_imp_svm$ImportancePercent <- 100 * var_imp_svm$Overall / sum(var_imp_svm$Overall)
var_imp_svm$Variable <- rownames(var_imp_svm)

var_imp_svm <- var_imp_svm %>%
  dplyr::arrange(desc(ImportancePercent))

# Plot
ggplot(var_imp_svm, aes(x = reorder(Variable, ImportancePercent), y = ImportancePercent)) +
  geom_col(fill = "purple") +
  coord_flip() +
  labs(
    title = "Important Variables - SVM",
    x = "Variable",
    y = "Importance (%)"
  ) +
  theme_minimal(base_size = 13)

# ---------------------------
#           CART
# ---------------------------

# Train the CART model for multiclass classification
set.seed(123)
model_cart <- train(
  target ~ .,
  data = train_data,
  method = "rpart",
  trControl = trainControl(classProbs = TRUE)
)

# Predict on test set
pred_cart <- predict(model_cart, test_data)

# Print confusion matrix
cat("\nConfusion Matrix - CART\n")
conf_mat_cart <- confusionMatrix(pred_cart, test_data$target)
print(conf_mat_cart)

# Visualize the confusion matrix with ggplot2
cm_df_cart <- as.data.frame(conf_mat_cart$table)
colnames(cm_df_cart) <- c("Reference", "Prediction", "Freq")

ggplot(cm_df_cart, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "black") +
  geom_text(aes(label = Freq), size = 4) +
  scale_fill_gradient(low = "white", high = "darkgreen") +
  labs(
    title = "Confusion Matrix - CART (Decision Tree)",
    x = "Actual Class",
    y = "Predicted Class"
  ) +
  theme_minimal()

# Plot variable importance
var_imp_cart <- varImp(model_cart)$importance
var_imp_cart$Overall <- rowMeans(var_imp_cart)
var_imp_cart$ImportancePercent <- 100 * var_imp_cart$Overall / sum(var_imp_cart$Overall)
var_imp_cart$Variable <- rownames(var_imp_cart)

var_imp_cart <- var_imp_cart %>%
  arrange(desc(ImportancePercent))

ggplot(var_imp_cart, aes(x = reorder(Variable, ImportancePercent), y = ImportancePercent)) +
  geom_col(fill = "darkgreen") +
  coord_flip() +
  labs(
    title = "Top Important Variables - CART",
    x = "Variable",
    y = "Importance (%)"
  ) +
  theme_minimal(base_size = 13)

# Visualize the tree
rpart.plot(model_cart$finalModel, type = 3, extra = 104, fallen.leaves = TRUE,
           main = "CART Decision Tree")

# ---------------------------
#        Random Forest
# ---------------------------

# Train the Random Forest model for multiclass classification
set.seed(123)
model_rf <- train(
  target ~ ., 
  data = train_data, 
  method = "rf",
  trControl = trainControl(classProbs = TRUE),
  importance = TRUE
)

# Predict on test set
pred_rf <- predict(model_rf, test_data)

# Print confusion matrix
cat("\nConfusion Matrix - RF\n")
conf_mat_rf <- confusionMatrix(pred_rf, test_data$target)
print(conf_mat_rf)

# Visualize the confusion matrix with ggplot2
cm_df_rf <- as.data.frame(conf_mat_rf$table)
colnames(cm_df_rf) <- c("Reference", "Prediction", "Freq")

ggplot(cm_df_rf, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "black") +
  geom_text(aes(label = Freq), size = 4) +
  scale_fill_gradient(low = "white", high = "darkred") +
  labs(
    title = "Confusion Matrix - Random Forest",
    x = "Actual Class",
    y = "Predicted Class"
  ) +
  theme_minimal()

# Plot variable importance
# Extract importance and calculate average across all classes
var_imp <- varImp(model_rf)$importance
var_imp$Overall <- rowMeans(var_imp)

# Convert to percentage
var_imp$ImportancePercent <- 100 * var_imp$Overall / sum(var_imp$Overall)

# Add variable name
var_imp$Variable <- rownames(var_imp)

# Top 20
var_imp <- var_imp %>%
  arrange(desc(ImportancePercent))

# Plot
ggplot(var_imp, aes(x = reorder(Variable, ImportancePercent), y = ImportancePercent)) +
  geom_col(fill = "darkred") +
  coord_flip() +
  labs(
    title = "Most Important Variables - Random Forest",
    x = "Variable",
    y = "Importance (%)"
  ) +
  theme_minimal(base_size = 13)

# -----------------------------------
#   Multinomial Logistic Regression
# -----------------------------------

# Train the Multinomial Logistic Regression model
set.seed(123)
model_log <- train(
  target ~ .,
  data = train_data,
  method = "multinom",
  trControl = trainControl(classProbs = TRUE),
  preProcess = c("center", "scale"),
  trace = FALSE
)

# Predict on test set
pred_log <- predict(model_log, test_data)

# Print confusion matrix
cat("\nConfusion Matrix - Logistic Regression\n")
conf_mat_log <- confusionMatrix(pred_log, test_data$target)
print(conf_mat_log)

# Visualize the confusion matrix with ggplot2
cm_df_log <- as.data.frame(conf_mat_log$table)
colnames(cm_df_log) <- c("Reference", "Prediction", "Freq")

ggplot(cm_df_log, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "black") +
  geom_text(aes(label = Freq), size = 4) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(
    title = "Confusion Matrix - Multinomial Logistic Regression",
    x = "Actual Class",
    y = "Predicted Class"
  ) +
  theme_minimal()

# Plot variable importance
# Extract importance and calculate average across all classes
var_imp_log <- varImp(model_log)$importance
var_imp_log$Overall <- rowMeans(var_imp_log)

# Convert to percentage
var_imp_log$ImportancePercent <- 100 * var_imp_log$Overall / sum(var_imp_log$Overall)
var_imp_log$Variable <- rownames(var_imp_log)

# Top 20
var_imp_log <- var_imp_log %>%
  arrange(desc(ImportancePercent))

ggplot(var_imp_log, aes(x = reorder(Variable, ImportancePercent), y = ImportancePercent)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Most Important Variables - Logistic Regression",
    x = "Variable",
    y = "Importance (%)"
  ) +
  theme_minimal(base_size = 13)

# -----------------------------
#       Gradient Boosting
# -----------------------------

# Train the Gradient Boosting model for multiclass classification
set.seed(123)
model_gbm <- train(
  target ~ .,
  data = train_data,
  method = "gbm",
  trControl = trainControl(classProbs = TRUE),
  verbose = FALSE
)

# Predict on test set
pred_gbm <- predict(model_gbm, test_data)

# Print confusion matrix
cat("\nConfusion Matrix - Gradient Boosting\n")
conf_mat_gbm <- confusionMatrix(pred_gbm, test_data$target)
print(conf_mat_gbm)

# Visualize the confusion matrix with ggplot2
cm_df_gbm <- as.data.frame(conf_mat_gbm$table)
colnames(cm_df_gbm) <- c("Reference", "Prediction", "Freq")

ggplot(cm_df_gbm, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "black") +
  geom_text(aes(label = Freq), size = 4) +
  scale_fill_gradient(low = "white", high = "goldenrod") +
  labs(
    title = "Confusion Matrix - Gradient Boosting",
    x = "Actual Class",
    y = "Predicted Class"
  ) +
  theme_minimal()

# Plot variable importance
var_imp_gbm <- varImp(model_gbm)$importance
var_imp_gbm$Overall <- rowMeans(var_imp_gbm)
var_imp_gbm$ImportancePercent <- 100 * var_imp_gbm$Overall / sum(var_imp_gbm$Overall)
var_imp_gbm$Variable <- rownames(var_imp_gbm)

var_imp_gbm <- var_imp_gbm %>%
  arrange(desc(ImportancePercent))

ggplot(var_imp_gbm, aes(x = reorder(Variable, ImportancePercent), y = ImportancePercent)) +
  geom_col(fill = "goldenrod") +
  coord_flip() +
  labs(
    title = "Top Important Variables - Gradient Boosting",
    x = "Variable",
    y = "Importance (%)"
  ) +
  theme_minimal(base_size = 13)

# ---------------------------
#     Neural Network (nnet)
# ---------------------------

# Train the Neural Network model for multiclass classification
set.seed(123)
model_nnet <- train(
  target ~ .,
  data = train_data,
  method = "nnet",
  trControl = trainControl(classProbs = TRUE),
  preProcess = c("center", "scale"),
  trace = FALSE,
  MaxNWts = 5000,
  linout = FALSE
)

# Predict on test set
pred_nnet <- predict(model_nnet, test_data)

# Print confusion matrix
cat("\nConfusion Matrix - Neural Network\n")
conf_mat_nnet <- confusionMatrix(pred_nnet, test_data$target)
print(conf_mat_nnet)

# Visualize the confusion matrix with ggplot2
cm_df_nnet <- as.data.frame(conf_mat_nnet$table)
colnames(cm_df_nnet) <- c("Reference", "Prediction", "Freq")

ggplot(cm_df_nnet, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "black") +
  geom_text(aes(label = Freq), size = 4) +
  scale_fill_gradient(low = "white", high = "darkorange") +
  labs(
    title = "Confusion Matrix - Neural Network (nnet)",
    x = "Actual Class",
    y = "Predicted Class"
  ) +
  theme_minimal()

# Plot variable importance
var_imp_nnet <- varImp(model_nnet)$importance
var_imp_nnet <- as.data.frame(var_imp_nnet)  
var_imp_nnet$Overall <- rowMeans(var_imp_nnet)
var_imp_nnet$ImportancePercent <- 100 * var_imp_nnet$Overall / sum(var_imp_nnet$Overall)
var_imp_nnet$Variable <- rownames(var_imp_nnet)

var_imp_nnet <- var_imp_nnet %>%
  dplyr::arrange(desc(ImportancePercent))  

# Plot
ggplot(var_imp_nnet, aes(x = reorder(Variable, ImportancePercent), y = ImportancePercent)) +
  geom_col(fill = "darkorange") +
  coord_flip() +
  labs(
    title = "Top Important Variables - Neural Network",
    x = "Variable",
    y = "Importance (%)"
  ) +
  theme_minimal(base_size = 13)

# ---------------------------
#   Compare Model Performance
# ---------------------------

# Create a resamples object
model_list <- resamples(
  list(
    SVM = model_svm,
    CART = model_cart,
    RF = model_rf,
    LogReg = model_log,
    NNet = model_nnet,
    GBM = model_gbm
  )
)

# Summary of cross-validated metrics
summary(model_list)

# Boxplot of Accuracy and Kappa
bwplot(model_list, metric = "Accuracy", main = "Model Accuracy Comparison")
bwplot(model_list, metric = "Kappa", main = "Model Kappa Comparison")

# Dotplot for overall comparison
dotplot(model_list, metric = "Accuracy", main = "Dotplot - Accuracy by Model")

# ---------------------------
#   F1 Macro Score Function
# ---------------------------

f1_macro <- function(true, pred) {
  classes <- levels(true)
  f1_list <- sapply(classes, function(class) {
    F1_Score(y_true = ifelse(true == class, 1, 0),
             y_pred = ifelse(pred == class, 1, 0))
  })
  mean(f1_list, na.rm = TRUE)
}

# ---------------------------
#   RMSE Score Function
# ---------------------------

# Convert target classes to numeric values
true_numeric <- as.numeric(test_data$target)

rmse_score <- function(pred) {
  pred_numeric <- as.numeric(pred)
  RMSE(pred_numeric, true_numeric)
}

# ---------------------------
#   Create Final Comparison Table
# ---------------------------

# Function to extract Accuracy from a model
conf_matrix <- function(model, test_data) {
  pred <- predict(model, test_data)
  cm <- confusionMatrix(pred, test_data$target)
  cm$overall["Accuracy"]
}

# Build the comparison dataframe
comparison_df <- data.frame(
  Model = c("SVM", "CART", "RF", "LogReg", "NNet", "GBM"),
  Accuracy = c(
    conf_matrix(model_svm, test_data),
    conf_matrix(model_cart, test_data),
    conf_matrix(model_rf, test_data),
    conf_matrix(model_log, test_data),
    conf_matrix(model_nnet, test_data),
    conf_matrix(model_gbm, test_data)
  ),
  Kappa = c(
    confusionMatrix(pred_svm, test_data$target)$overall["Kappa"],
    confusionMatrix(pred_cart, test_data$target)$overall["Kappa"],
    confusionMatrix(pred_rf, test_data$target)$overall["Kappa"],
    confusionMatrix(pred_log, test_data$target)$overall["Kappa"],
    confusionMatrix(pred_nnet, test_data$target)$overall["Kappa"],
    confusionMatrix(pred_gbm, test_data$target)$overall["Kappa"]
  ),
  F1_Macro = c(
    f1_macro(test_data$target, pred_svm),
    f1_macro(test_data$target, pred_cart),
    f1_macro(test_data$target, pred_rf),
    f1_macro(test_data$target, pred_log),
    f1_macro(test_data$target, pred_nnet),
    f1_macro(test_data$target, pred_gbm)
  ),
  RMSE = c(
    rmse_score(pred_svm),
    rmse_score(pred_cart),
    rmse_score(pred_rf),
    rmse_score(pred_log),
    rmse_score(pred_nnet),
    rmse_score(pred_gbm)
  )
)

# Round for clarity
comparison_df <- comparison_df %>%
  mutate(across(where(is.numeric), round, 3)) %>%
  arrange(desc(Accuracy))

# Print the final comparison table
print(comparison_df)

# Convert the comparison dataframe to long format
comparison_long <- melt(comparison_df, id.vars = "Model")

model_colors <- c(
  "SVM" = "purple",
  "CART" = "darkgreen",
  "RF" = "darkred",
  "LogReg" = "steelblue",
  "NNet" = "goldenrod",
  "GBM" = "darkorange"
)

# Plot: one chart with facets for each metric
ggplot(comparison_long, aes(x = reorder(Model, value), y = value, fill = Model)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ variable, scales = "free_y") +
  coord_flip() +
  labs(title = "Model Performance Comparison",
       x = "Model",
       y = "Value") +
  scale_fill_manual(values = model_colors) +
  theme_minimal(base_size = 13)

# ---------------------------
#          ROC curve
# ---------------------------

# Predict class probabilities - Random Forest
prob_rf <- predict(model_rf, test_data, type = "prob")

# Compute multiclass AUC
roc_rf <- multiclass.roc(test_data$target, prob_rf)
cat("Multiclass AUC - Random Forest:", round(roc_rf$auc, 3), "\n")

# Get individual one-vs-all ROC curves
rs_rf <- multiclass.roc(test_data$target, prob_rf)
rs_list <- rs_rf$rocs

# Plot ROC curves
plot.roc(rs_list[[1]][[1]], col = "red", main = "ROC Curves - Random Forest")
for (i in 2:length(rs_list)) {
  lines.roc(rs_list[[i]][[1]], col = i)
}
legend(
  "bottomright",
  legend = levels(test_data$target),
  col = 1:length(rs_list),
  lwd = 2
)

# Predict class probabilities - GBM
prob_gbm <- predict(model_gbm, test_data, type = "prob")

# Compute multiclass AUC
roc_gbm <- multiclass.roc(test_data$target, prob_gbm)
cat("Multiclass AUC - GBM:", round(roc_gbm$auc, 3), "\n")

# Get individual one-vs-all ROC curves
rs_gbm <- multiclass.roc(test_data$target, prob_gbm)
rs_list_gbm <- rs_gbm$rocs

# Plot ROC curves
plot.roc(rs_list_gbm[[1]][[1]], col = "red", main = "ROC Curves - GBM")
for (i in 2:length(rs_list_gbm)) {
  lines.roc(rs_list_gbm[[i]][[1]], col = i)
}
legend(
  "bottomright",
  legend = levels(test_data$target),
  col = 1:length(rs_list_gbm),
  lwd = 2
)

# Predict class probabilities - LogReg
prob_log <- predict(model_log, test_data, type = "prob")

# Compute multiclass AUC
roc_log <- multiclass.roc(test_data$target, prob_log)
cat("Multiclass AUC - LogReg:", round(roc_log$auc, 3), "\n")

# Get individual one-vs-all ROC curves
rs_log <- multiclass.roc(test_data$target, prob_log)
rs_list_log <- rs_log$rocs

# Plot ROC curves
plot.roc(rs_list_log[[1]][[1]], col = "red", main = "ROC Curves - LogReg")
for (i in 2:length(rs_list_log)) {
  lines.roc(rs_list_log[[i]][[1]], col = i)
}
legend(
  "bottomright",
  legend = levels(test_data$target),
  col = 1:length(rs_list_log),
  lwd = 2
)

# Predict class probabilities - CART
prob_cart <- predict(model_cart, test_data, type = "prob")

# Compute multiclass AUC
roc_cart <- multiclass.roc(test_data$target, prob_cart)
cat("Multiclass AUC - CART:", round(roc_cart$auc, 3), "\n")

# Get individual one-vs-all ROC curves
rs_cart <- multiclass.roc(test_data$target, prob_cart)
rs_list_cart <- rs_cart$rocs

# Plot ROC curves
plot.roc(rs_list_cart[[1]][[1]], col = "red", main = "ROC Curves - CART")
for (i in 2:length(rs_list_cart)) {
  lines.roc(rs_list_cart[[i]][[1]], col = i)
}
legend(
  "bottomright",
  legend = levels(test_data$target),
  col = 1:length(rs_list_cart),
  lwd = 2
)

