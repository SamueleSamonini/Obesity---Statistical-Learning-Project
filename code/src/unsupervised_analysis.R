# ---------------------------
#     Unsupervised Learning
# ---------------------------

# Load required libraries
library(tidyverse)
library(caret)
library(FactoMineR)
library(factoextra)
library(plotly)
library(reshape2)
library(pheatmap)
library(RColorBrewer)
library(fmsb)

# Load the dataset
obesity_df <- read.csv("data/obesity.csv", header = TRUE)

head(obesity_df)
summary(obesity_df)

# Save 'Gender' column as factor for plotting
gender_col <- obesity_df$Gender

# Remove the target variable
df_data <- dplyr::select(obesity_df, -NObeyesdad)

# Convert categorical variables into dummy variables
df_dummy <- dummyVars(~ ., data = df_data)
df_transformed <- predict(df_dummy, newdata = df_data) %>% as.data.frame()

head(df_transformed)

# Standardize the data
scaled_data <- scale(df_transformed)

head(scaled_data)

# Check for missing values
sum(is.na(scaled_data))

# -------------------------------
#   Exploratory Outlier Analysis
# -------------------------------

# Histogram of numeric variables
numeric_data <- df_data %>% select(where(is.numeric))

numeric_data %>%
  gather(key = "Variable", value = "Value") %>%
  ggplot(aes(x = Value)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "black") +
  facet_wrap(~ Variable, scales = "free") +
  theme_minimal() +
  labs(title = "Histograms of Numeric Variables")

# Boxplot of numeric variables
numeric_data %>%
  gather(key = "Variable", value = "Value") %>%
  ggplot(aes(x = Variable, y = Value, fill = Variable)) +
  geom_boxplot() +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Boxplots of Numeric Variables")

# Histogram of numeric variables
obesity_df %>%
  dplyr::select_if(~!is.numeric(.)) %>%  # select the categorical variable
  gather(key = "variable", value = "value") %>%
  ggplot(aes(x = value)) +
  geom_bar(fill = "steelblue", color = "black") +
  facet_wrap(~ variable, scales = "free", ncol = 2) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle=45, hjust=1)) +
  labs(title = "Categorical Variables Distributions", x = "", y = "Count")

# -------------------------------
#   Correlation Heatmap
# -------------------------------

cor_matrix <- cor(scaled_data, use = "complete.obs")
melted_cor <- melt(cor_matrix)

ggplot(melted_cor, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "darkred", mid = "white", high = "darkblue", midpoint = 0) +
  theme_minimal() +
  labs(title = "Correlation Heatmap", x = "", y = "") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# -------------------------------
#   Principal Component Analysis
# -------------------------------

pca_result <- prcomp(scaled_data, center = TRUE, scale. = TRUE)

# Explained variance
fviz_eig(pca_result, addlabels = TRUE, ylim = c(0, 50))

# Contribution of variables to PCs
fviz_pca_var(pca_result,
             col.var = "contrib",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE)

# PCA by gender
fviz_pca_ind(pca_result,
             geom.ind = "point",
             col.ind = gender_col,
             palette = "jco",
             addEllipses = TRUE,
             legend.title = "Gender")

# Extract first two PCs
pca_data <- pca_result$x[, 1:2]

# -------------------------------
#   Optimal Number of Clusters
# -------------------------------

fviz_nbclust(pca_data, kmeans, method = "wss") +
  labs(subtitle = "Elbow Method")

fviz_nbclust(pca_data, kmeans, method = "silhouette") +
  labs(subtitle = "Silhouette Method")

# -------------------------------
#   K-Means Clustering (k = 7)
# -------------------------------

set.seed(123)
kmeans_7 <- kmeans(pca_data, centers = 7, nstart = 25)
df_clustered <- df_data

# Add cluster column
df_clustered$cluster <- as.factor(kmeans_7$cluster)

# Cluster visualization
fviz_cluster(kmeans_7, data = pca_data,
             ellipse.type = "convex",
             palette = "jco",
             ggtheme = theme_minimal())

# -------------------------------
#   3D PCA Cluster Plot
# -------------------------------

df_pca_3d <- as.data.frame(pca_result$x[, 1:3])
colnames(df_pca_3d) <- c("PC1", "PC2", "PC3")
df_pca_3d$Cluster <- as.factor(kmeans_7$cluster)

plot_ly(df_pca_3d, x = ~PC1, y = ~PC2, z = ~PC3,
        color = ~Cluster,
        colors = c('#E41A1C', '#377EB8', '#4DAF4A', '#984EA3',
                   '#FF7F00', '#FFFF33', '#A65628'),
        type = "scatter3d", mode = "markers")

# -------------------------------
#   Cluster Profile Summary
# -------------------------------

summary_7 <- df_clustered %>%
  group_by(cluster) %>%
  summarise(across(where(is.numeric), mean, na.rm = TRUE))

# Normalize for heatmap
norm_7 <- summary_7 %>%
  column_to_rownames("cluster") %>%
  scale(center = TRUE, scale = TRUE) %>%
  as.data.frame() %>%
  rownames_to_column("cluster") %>%
  pivot_longer(-cluster, names_to = "Variable", values_to = "Value")

# Heatmap

ggplot(norm_7, aes(x = Variable, y = cluster, fill = Value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(Value, 2)), size = 3) +
  scale_fill_gradient2(low = "darkred", mid = "white", high = "darkblue", midpoint = 0) +
  labs(title = "Heatmap - Cluster (z-score)", x = "Variables", y = "Cluster") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# -------------------------------
#   Distribution Plots by Cluster
# -------------------------------

# Age
ggplot(df_clustered, aes(x = Age, fill = cluster)) +
  geom_density(alpha = 0.5) +
  labs(title = "Cluster Age Distribution")


