# Loading all the required packages 

require(ggplot2) # for data visualization
require(stringr) #extracting string patterns
require(Matrix) # matrix transformations
require(glmnet) # ridge, lasso & elastinet
require(xgboost) # gbm
require(randomForest)
require(Metrics) # rmse
require(dplyr) # load this in last so plyr doens't overlap it
require(caret) # one hot encoding
require(scales) # plotting $$
require(e1071) # skewness
require(corrplot) # correlation plot

# Loading the data and combining for pre-processing data. We removed Id and Sales Price from train and Id from test
# as we are not going to use both for pre-processing

train<-read.csv("F:/SP Jain_ extra/Kaggle Datasets/House Prices/train_houseprices.csv",stringsAsFactors=FALSE)
test<-read.csv("F:/SP Jain_ extra/Kaggle Datasets/House Prices/test_houseprices.csv",stringsAsFactors=FALSE)
df.combined <- rbind(within(train, rm('Id','SalePrice')), within(test, rm('Id')))
dim(df.combined)

# To find the columns which have missing values

na.cols <- which(colSums(is.na(df.combined)) > 0)
sort(colSums(sapply(df.combined[na.cols], is.na)), decreasing = TRUE)
paste('There are', length(na.cols), 'columns with missing values')

# Creating a function to visualize the categorical data

plot.categoric <- function(cols, df){
  for (col in cols) {
    order.cols <- names(sort(table(df.combined[,col]), decreasing = TRUE))
    
    num.plot <- qplot(df[,col]) +
      geom_bar(fill = 'red') +
      geom_text(aes(label = ..count..), stat='count', vjust=-0.5) +
      theme_minimal() +
      scale_y_continuous(limits = c(0,max(table(df[,col]))*1.1)) +
      scale_x_discrete(limits = order.cols) +
      xlab(col) +
      theme(axis.text.x = element_text(angle = 30, size=12))
    
    print(num.plot)
  }
}

# Pool Quality

plot.categoric('PoolQC', df.combined)
df.combined[(df.combined$PoolArea > 0) & is.na(df.combined$PoolQC),c('PoolQC','PoolArea')]
df.combined[,c('PoolQC','PoolArea')] %>%
  group_by(PoolQC) %>%
  summarise(mean = mean(PoolArea), counts = n()) 

# There are only 3 PoolQC which have PoolArea >0, therefore for missing values we will fill 
# those values with closest corresponding mean of that PoolArea and rest with 'None' which refers to 
# no pool

df.combined[2421,'PoolQC'] = 'Ex'
df.combined[2504,'PoolQC'] = 'Ex'
df.combined[2600,'PoolQC'] = 'Fa'
df.combined$PoolQC[is.na(df.combined$PoolQC)] = 'None'

# It is obvious that garage would be built when house would be built. So checking NAs and replacing
# with year the house was built
length(which(df.combined$GarageYrBlt == df.combined$YearBuilt))
idx <- which(is.na(df.combined$GarageYrBlt))
df.combined[idx, 'GarageYrBlt'] <- df.combined[idx, 'YearBuilt']
# That leaves 6 garage features in our dataset and 4 of them have at least 157 missing values while GarageArea and GarageCars both only have 1, 
# thus we can assume this particular house does not have a garage at all. For the rest of the houses we 
#can check to see that if the NA's recorded also have 0 GarageArea and 0 GarageCars. If they do we 
#can fill in their missing values with 'None' since having 0 area and 0 cars in their garage will imply 
#that they do not have any at all.
garage.cols <- c('GarageArea', 'GarageCars', 'GarageQual', 'GarageFinish', 'GarageCond', 'GarageType')
df.combined[is.na(df.combined$GarageCond),garage.cols]

# For the house with GarageArea = 360 and GarageCars = 1, but NA's in the other columns, we can use the most frequent values for each columns 
# from houses with a similar area and car count.

idx <- which(((df.combined$GarageArea < 370) & (df.combined$GarageArea > 350)) & (df.combined$GarageCars == 1))
names(sapply(df.combined[idx, garage.cols], function(x) sort(table(x), decreasing=TRUE)[1]))
df.combined[2127,'GarageQual'] = 'TA'
df.combined[2127, 'GarageFinish'] = 'Unf'
df.combined[2127, 'GarageCond'] = 'TA'

# Now we can fill in any missing numeric values with 0 and categoric with 'None' since these houses 
# recorded having 0 area and 0 cars in their garage.
for (col in garage.cols){
  if (sapply(df.combined[col], is.numeric) == TRUE){
    df.combined[sapply(df.combined[col], is.na), col] = 0
  }
  else{
    df.combined[sapply(df.combined[col], is.na), col] = 'None'
  }
}

# Kitchen Quality: Filling with most frequent value as only 1 NA is there

plot.categoric('KitchenQual', df.combined)
df.combined$KitchenQual[is.na(df.combined$KitchenQual)] = 'TA'

# Electrical: Filling with most frequent value as only 1 NA is there

plot.categoric('Electrical', df.combined)
df.combined$Electrical[is.na(df.combined$Electrical)] = 'SBrkr'

# There are 11 basement features each with at least 1 missing value. We can take a look at the subset of 
# just these columns from our data.

bsmt.cols <- names(df.combined)[sapply(names(df.combined), function(x) str_detect(x, 'Bsmt'))]

# Almost all of the missing values for each categoric basement feature comes from houses with 0 on each 
# features corresponding to area. We can fill in these values with 'None' since these houses certainly 
# don't have basements. Rows 949, 1488 and 2349 are the only missing values from BsmtExposure, we can fill
# this with No as that is the most frequent value and these houses most likely don't have any exposure for
# their basements. The rest of the basement columns corresponding to area will be filled with 0 since they 
# likely don't have a basement and the categoric missing values will be filled with NoBsmt.

df.combined[is.na(df.combined$BsmtExposure),bsmt.cols]
plot.categoric('BsmtExposure', df.combined)

df.combined[c(949, 1488, 2349), 'BsmtExposure'] = 'No'

for (col in bsmt.cols){
  if (sapply(df.combined[col], is.numeric) == TRUE){
    df.combined[sapply(df.combined[col], is.na),col] = 0
  }
  else{
    df.combined[sapply(df.combined[col],is.na),col] = 'None'
  }
}

# There is only 1 missing value for Exterior1st and Exterior2nd coming from the same hosue and there 
# aren't any other features that can help us predict what variable should be filled so we can fill this 
# with 'Other' since the NA is likely due to having an exterior cover that is not listed.

idx <- which(is.na(df.combined$Exterior1st) | is.na(df.combined$Exterior2nd))
df.combined[idx,c('Exterior1st', 'Exterior2nd')]
df.combined$Exterior1st[is.na(df.combined$Exterior1st)] = 'Other'
df.combined$Exterior2nd[is.na(df.combined$Exterior2nd)] = 'Other'

# Sale Type: filling with the most frequent missing value

plot.categoric('SaleType', df.combined)
df.combined[is.na(df.combined$SaleType),c('SaleCondition')]
table(df.combined$SaleCondition, df.combined$SaleType)
df.combined$SaleType[is.na(df.combined$SaleType)] = 'WD'

# Functional: filling with the most frequent missing value

plot.categoric('Functional', df.combined)
df.combined$Functional[is.na(df.combined$Functional)] = 'Typ'

# Utilities: It has only one type of value so won't be used for modelling

plot.categoric('Utilities', df.combined)
which(df.combined$Utilities == 'NoSeWa') # in the training data set
col.drops <- c('Utilities')
df.combined <- df.combined[,!names(df.combined) %in% c('Utilities')]

# There are only 4 missing values for MSZoning. We can see what the subclass is for the houses with missing values for Zoning.

df.combined[is.na(df.combined$MSZoning),c('MSZoning','MSSubClass')]
plot.categoric('MSZoning', df.combined)
table(df.combined$MSZoning, df.combined$MSSubClass)
df.combined$MSZoning[c(2217, 2905)] = 'RL'
df.combined$MSZoning[c(1916, 2251)] = 'RM'

# There are 23 missing values for MasVnrArea and 24 for MasVnrType. We can see if both missing values come from the same houses

df.combined[(is.na(df.combined$MasVnrType)) | (is.na(df.combined$MasVnrArea)), c('MasVnrType', 'MasVnrArea')]

# All but one house has missing values for both columns. For houses with NA's on both columns we can 
# fill 0 for the area and None for the type since they likely do not have a masonry veneer. For the house 
# with a MasVnrArea of 198 but NA for MasVnrType we can record the median areas for each type and see 
# which type is closest to 198

na.omit(df.combined[,c('MasVnrType','MasVnrArea')]) %>%
  group_by(na.omit(MasVnrType)) %>%
  summarise(MedianArea = median(MasVnrArea,na.rm = TRUE), counts = n()) %>%
  arrange(MedianArea)

df.combined[2611, 'MasVnrType'] = 'BrkFace'
df.combined$MasVnrType[is.na(df.combined$MasVnrType)] = 'None'
df.combined$MasVnrArea[is.na(df.combined$MasVnrArea)] = 0
df.combined['Nbrh.factor'] <- factor(df.combined$Neighborhood, levels = unique(df.combined$Neighborhood))

# LotFrontage: Grouping according to neigborhood and taking the median and filling missing values accordingly
lot.by.nbrh <- df.combined[,c('Neighborhood','LotFrontage')] %>%
  group_by(Neighborhood) %>%
  summarise(median = median(LotFrontage, na.rm = TRUE))
lot.by.nbrh
idx = which(is.na(df.combined$LotFrontage))

for (i in idx){
  lot.median <- lot.by.nbrh[lot.by.nbrh == df.combined$Neighborhood[i],'median']
  df.combined[i,'LotFrontage'] <- lot.median[[1]]
}

# Fence Quality: Filling with None

plot.categoric('Fence', df.combined)
df.combined$Fence[is.na(df.combined$Fence)] = 'None'

# Misc Feature: Filling with None

table(df.combined$MiscFeature)
df.combined$MiscFeature[is.na(df.combined$MiscFeature)] = 'None'

# FireplaceQu: Filling with None

plot.categoric('FireplaceQu', df.combined)
which((df.combined$Fireplaces > 0) & (is.na(df.combined$FireplaceQu)))
df.combined$FireplaceQu[is.na(df.combined$FireplaceQu)] = 'None'

# Alley: Filling with None

plot.categoric('Alley', df.combined)
df.combined$Alley[is.na(df.combined$Alley)] = 'None'

paste('There are', sum(sapply(df.combined, is.na)), 'missing values left')

# Creating a new data frame with numerical features and categorical features

num_features <- names(which(sapply(df.combined, is.numeric)))
cat_features <- names(which(sapply(df.combined, is.character)))
df.numeric <- df.combined[num_features]

group.df <- df.combined[1:1460,]
group.df$SalePrice <- train$SalePrice

# function that groups a column by its features and returns the median saleprice for each unique feature. 

group.prices <- function(col) {
  group.table <- group.df[,c(col, 'SalePrice', 'OverallQual')] %>%
    group_by_(col) %>%
    summarise(mean.Quality = round(mean(OverallQual),2),
              mean.Price = mean(SalePrice), n = n()) %>%
    arrange(mean.Quality)
  
  print(qplot(x=reorder(group.table[[col]], -group.table[['mean.Price']]), y=group.table[['mean.Price']]) +
          geom_bar(stat='identity', fill='cornflowerblue') +
          theme_minimal() +
          scale_y_continuous(labels = dollar) +
          labs(x=col, y='Mean SalePrice') +
          theme(axis.text.x = element_text(angle = 45)))
  
  return(data.frame(group.table))
}
# functional to compute the mean overall quality for each quality
quality.mean <- function(col) {
  group.table <- df.combined[,c(col, 'OverallQual')] %>%
    group_by_(col) %>%
    summarise(mean.qual = mean(OverallQual)) %>%
    arrange(mean.qual)
  
  return(data.frame(group.table))
}


# function that maps a categoric value to its corresponding numeric value and returns that column to the data frame
map.fcn <- function(cols, map.list, df){
  for (col in cols){
    df[col] <- as.numeric(map.list[df.combined[,col]])
  }
  return(df)
}

# Grouping all quality variables to decide the order of the classes in the variables

qual.cols <- c('ExterQual', 'ExterCond', 'GarageQual', 'GarageCond', 'FireplaceQu', 'KitchenQual', 'HeatingQC', 'BsmtQual')

group.prices('FireplaceQu')
group.prices('BsmtQual')
group.prices('KitchenQual')

# Mapping the classes to a number based on the mean sales price of that particular class

qual.list <- c('None' = 0, 'Po' = 1, 'Fa' = 2, 'TA' = 3, 'Gd' = 4, 'Ex' = 5)
df.numeric <- map.fcn(qual.cols, qual.list, df.numeric)

group.prices('BsmtExposure')
bsmt.list <- c('None' = 0, 'No' = 1, 'Mn' = 2, 'Av' = 3, 'Gd' = 4)
df.numeric = map.fcn(c('BsmtExposure'), bsmt.list, df.numeric)

group.prices('BsmtFinType1')
df.combined[,c('BsmtFinType1', 'BsmtFinSF1')] %>%
  group_by(BsmtFinType1) %>%
  summarise(medianArea = median(BsmtFinSF1), counts = n()) %>%
  arrange(medianArea) %>%
  ggplot(aes(x=reorder(BsmtFinType1,-medianArea), y=medianArea)) +
  geom_bar(stat = 'identity', fill='cornflowerblue') +
  labs(x='BsmtFinType2', y='Median of BsmtFinSF2') +
  geom_text(aes(label = sort(medianArea)), vjust = -0.5) +
  scale_y_continuous(limits = c(0,850)) +
  theme_minimal()

bsmt.fin.list <- c('None' = 0, 'Unf' = 1, 'LwQ' = 2,'Rec'= 3, 'BLQ' = 4, 'ALQ' = 5, 'GLQ' = 6)
df.numeric <- map.fcn(c('BsmtFinType1','BsmtFinType2'), bsmt.fin.list, df.numeric)

group.prices('Functional')
functional.list <- c('None' = 0, 'Sal' = 1, 'Sev' = 2, 'Maj2' = 3, 'Maj1' = 4, 'Mod' = 5, 'Min2' = 6, 'Min1' = 7, 'Typ'= 8)
df.numeric['Functional'] <- as.numeric(functional.list[df.combined$Functional])

group.prices('GarageFinish')
garage.fin.list <- c('None' = 0,'Unf' = 1, 'RFn' = 1, 'Fin' = 2)
df.numeric['GarageFinish'] <- as.numeric(garage.fin.list[df.combined$GarageFinish])


group.prices('Fence')
fence.list <- c('None' = 0, 'MnWw' = 1, 'GdWo' = 1, 'MnPrv' = 2, 'GdPrv' = 4)
df.numeric['Fence'] <- as.numeric(fence.list[df.combined$Fence])

MSdwelling.list <- c('20' = 1, '30'= 0, '40' = 0, '45' = 0,'50' = 0, '60' = 1, '70' = 0, '75' = 0, '80' = 0, '85' = 0, '90' = 0, '120' = 1, '150' = 0, '160' = 0, '180' = 0, '190' = 0)
df.numeric['NewerDwelling'] <- as.numeric(MSdwelling.list[as.character(df.combined$MSSubClass)])

# Computing Correlations and focussing on correlations greater than 0.5 and less than -0.5

corr.df <- cbind(df.numeric[1:1460,], train['SalePrice'])
correlations <- cor(corr.df)
corr.SalePrice <- as.matrix(sort(correlations[,'SalePrice'], decreasing = TRUE))
corr.idx <- names(which(apply(corr.SalePrice, 1, function(x) (x > 0.5 | x < -0.5))))

corrplot(as.matrix(correlations[corr.idx,corr.idx]), type = 'upper', method='color', addCoef.col = 'black', tl.cex = .7,cl.cex = .7, number.cex=.7)

install.packages("GGally")
require(GGally)

lm.plt <- function(data, mapping, ...){
  plt <- ggplot(data = data, mapping = mapping) + 
    geom_point(shape = 20, alpha = 0.7, color = 'darkseagreen') +
    geom_smooth(method=loess, fill="red", color="red") +
    geom_smooth(method=lm, fill="blue", color="blue") +
    theme_minimal()
  return(plt)
}

ggpairs(corr.df, corr.idx[1:6], lower = list(continuous = lm.plt))
ggpairs(corr.df, corr.idx[c(1,7:11)], lower = list(continuous = lm.plt))

# Creating a binary feature for the columns having binary classes

plot.categoric('LotShape', df.combined)
df.numeric['RegularLotShape'] <- (df.combined$LotShape == 'Reg') * 1

plot.categoric('LandContour', df.combined)
df.numeric['LandLeveled'] <- (df.combined$LandContour == 'Lvl') * 1

plot.categoric('LandSlope', df.combined)
df.numeric['LandSlopeGentle'] <- (df.combined$LandSlope == 'Gtl') * 1

plot.categoric('Electrical', df.combined)
df.numeric['ElectricalSB'] <- (df.combined$Electrical == 'SBrkr') * 1

plot.categoric('GarageType', df.combined)
df.numeric['GarageDetchd'] <- (df.combined$GarageType == 'Detchd') * 1

plot.categoric('PavedDrive', df.combined)
df.numeric['HasPavedDrive'] <- (df.combined$PavedDrive == 'Y') * 1

df.numeric['HasWoodDeck'] <- (df.combined$WoodDeckSF > 0) * 1

df.numeric['Has2ndFlr'] <- (df.combined$X2ndFlrSF > 0) * 1

df.numeric['HasMasVnr'] <- (df.combined$MasVnrArea > 0) * 1

plot.categoric('MiscFeature', df.combined)

df.numeric['HasShed'] <- (df.combined$MiscFeature == 'Shed') * 1

# creating a new column for houses that are remodeled

df.numeric['Remodeled'] <- (df.combined$YearBuilt != df.combined$YearRemodAdd) * 1

# creating a new column for houses that are recently remodeled

df.numeric['RecentRemodel'] <- (df.combined$YearRemodAdd >= df.combined$YrSold) * 1

# creating a new column for houses that are built and sold in the same year

df.numeric['NewHouse'] <- (df.combined$YearBuilt == df.combined$YrSold) * 1

# one  hot encoding for area greater than 0 

cols.binary <- c('X2ndFlrSF', 'MasVnrArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'X3SsnPorch', 'ScreenPorch')
for (col in cols.binary){
  df.numeric[str_c('Has',col)] <- (df.combined[,col] == 0) * 1
}

# Analysing which month the houses were sold

ggplot(df.combined, aes(x=MoSold)) +
  geom_bar(fill = 'cornflowerblue') +
  geom_text(aes(label=..count..), stat='count', vjust = -.5) +
  theme_minimal() +
  scale_x_continuous(breaks = 1:12)
df.numeric['HighSeason'] <- (df.combined$MoSold %in% c(5,6,7)) * 1

# Analysing expensive neighborhoods

train[,c('Neighborhood','SalePrice')] %>%
  group_by(Neighborhood) %>%
  summarise(median.price = median(SalePrice, na.rm = TRUE)) %>%
  arrange(median.price) %>%
  mutate(nhbr.sorted = factor(Neighborhood, levels=Neighborhood)) %>%
  ggplot(aes(x=nhbr.sorted, y=median.price)) +
  geom_point() +
  geom_text(aes(label = median.price, angle = 45), vjust = 2) +
  theme_minimal() +
  labs(x='Neighborhood', y='Median price') +
  theme(text = element_text(size=12),
        axis.text.x = element_text(angle=45))

other.nbrh <- unique(df.combined$Neighborhood)[!unique(df.combined$Neighborhood) %in% c('StoneBr', 'NoRidge','NridgHt')]

ggplot(train, aes(x=SalePrice, y=GrLivArea, colour=Neighborhood)) +
  geom_point(shape=16, alpha=.8, size=4) +
  scale_color_manual(limits = c(other.nbrh, 'StoneBr', 'NoRidge', 'NridgHt'), values = c(rep('black', length(other.nbrh)), 'indianred',
                                                                                         'cornflowerblue', 'darkseagreen')) +
  theme_minimal() +
  scale_x_continuous(label=dollar)

# One hot encoding the rich neighborhood

nbrh.rich <- c('Crawfor', 'Somerst, Timber', 'StoneBr', 'NoRidge', 'NridgeHt')
df.numeric['NbrhRich'] <- (df.combined$Neighborhood %in% nbrh.rich) *1

# Mapping the neighborhoods who have higher quality homes and run for larger sale prices

group.prices('Neighborhood')
nbrh.map <- c('MeadowV' = 0, 'IDOTRR' = 1, 'Sawyer' = 1, 'BrDale' = 1, 'OldTown' = 1, 'Edwards' = 1, 
              'BrkSide' = 1, 'Blueste' = 1, 'SWISU' = 2, 'NAmes' = 2, 'NPkVill' = 2, 'Mitchel' = 2,
              'SawyerW' = 2, 'Gilbert' = 2, 'NWAmes' = 2, 'Blmngtn' = 2, 'CollgCr' = 2, 'ClearCr' = 3, 
              'Crawfor' = 3, 'Veenker' = 3, 'Somerst' = 3, 'Timber' = 3, 'StoneBr' = 4, 'NoRidge' = 4, 
              'NridgHt' = 4)

df.numeric['NeighborhoodBin'] <- as.numeric(nbrh.map[df.combined$Neighborhood])

# Sale Condtion
group.prices('SaleCondition')
df.numeric['PartialPlan'] <- (df.combined$SaleCondition == 'Partial') * 1

group.prices('HeatingQC')
heating.list <- c('Po' = 0, 'Fa' = 1, 'TA' = 2, 'Gd' = 3, 'Ex' = 4)

df.numeric['HeatingScale'] <- as.numeric(heating.list[df.combined$HeatingQC])

# Adding all area features to find the total area of the house

area.cols <- c('LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
               'TotalBsmtSF', 'X1stFlrSF', 'X2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 
               'OpenPorchSF', 'EnclosedPorch', 'X3SsnPorch', 'ScreenPorch', 'LowQualFinSF', 'PoolArea')

df.numeric['TotalArea'] <- as.numeric(rowSums(df.combined[,area.cols]))
df.numeric['AreaInside'] <- as.numeric(df.combined$X1stFlrSF + df.combined$X2ndFlrSF)

# To calculate the age of the house and how long the house was in the market

df.numeric['Age'] <- as.numeric(2010 - df.combined$YearBuilt)
df.numeric['TimeSinceSold'] <- as.numeric(2010 - df.combined$YrSold)

# how many years since the house was remodelled and sold 
df.numeric['YearSinceRemodel'] <- as.numeric(df.combined$YrSold - df.combined$YearRemodAdd)

# Computing correlation plot with overall quality of the house

corr.OverallQual <- as.matrix(sort(correlations[,'OverallQual'], decreasing = TRUE))
corr.idx <- names(which(apply(corr.OverallQual, 1, function(x) (x > 0.5 | x < -0.5))))
corrplot(as.matrix(correlations[corr.idx, corr.idx]), type = 'upper',
         method = 'color', addCoef.col = 'black', tl.cex =.7, cl.cex = .7,
         number.cex = .7)

# Outliers Handling

train.test.df <- rbind(dplyr::select(train,-SalePrice), test)
train.test.df$type <- c(rep('train',1460),rep('test',1459))

ggplot(train, aes(x=GrLivArea)) +
  geom_histogram(fill='indianred',color='black') +
  theme_minimal()

outlier_values <- boxplot.stats(train$GrLivArea)$out  # outlier values.
boxplot(train$GrLivArea, main="GrLivArea", boxwex=0.1)
mtext(paste("Outliers: ", paste(outlier_values[outlier_values>4000], collapse=", ")), cex=0.6)

ggplot(train.test.df, aes(x=type, y=GrLivArea, fill=type)) +
  geom_boxplot() +
  theme_minimal()
# There are 4 outliers, thus removing them as they will cause skewness

idx.outliers <- which(train$GrLivArea > 4000)
df.numeric <- df.numeric[!1:nrow(df.numeric) %in% idx.outliers,]
df.combined <- df.combined[!1:nrow(df.combined) %in% idx.outliers,]

dim(df.numeric)

require(psych)
# linear models assume normality from dependant variables 
# transform any skewed data into normal

skewed <- apply(df.numeric, 2, skewness)
skewed <- skewed[(skewed > 0.8) | (skewed < -0.8)]

kurtosis <- apply(df.numeric, 2, kurtosis)
kurtosis <- kurtosis[(kurtosis > 3.0) | (kurtosis < -3.0)]

# not very useful in our case
ks.p.val <- NULL
for (i in 1:length(df.numeric)) {
  test.stat <- ks.test(df.numeric[i], rnorm(1000))
  ks.p.val[i] <- test.stat$p.value
}

for(col in names(skewed)){
  if(0 %in% df.numeric[, col]) {
    df.numeric[,col] <- log(1+df.numeric[,col])
  }
  else {
    df.numeric[,col] <- log(df.numeric[,col])
  }
}

# normalize the data

scaler <- preProcess(df.numeric)
df.numeric <- predict(scaler, df.numeric)

# one hot encoding for categorical data

dummy <- dummyVars(" ~ .",data=df.combined[,cat_features])
df.categoric <- data.frame(predict(dummy,newdata=df.combined[,cat_features]))

# binning houses into sequences of 20, which will give us 7 different bins for a feature built in a year to fall into. 
# This will allow us to differentiate the different times/era houses, garages and remodeling took place. 

year.map = function(col.combined, col.name) {
  for (i in 1:7) {
    year.seq = seq(1871+(i-1)*20, 1871+i*20-1)
    idx = which(df.combined[,col.combined] %in% year.seq)
    df.categoric[idx,col.name] = i
  }
  return(df.categoric)
}
df.categoric['GarageYrBltBin'] = 0
df.categoric <- year.map('GarageYrBlt', 'GarageYrBltBin')
df.categoric['YearBuiltBin'] = 0
df.categoric <- year.map('YearBuilt','YearBuiltBin')
df.categoric['YearRemodAddBin'] = 0
df.categoric <- year.map('YearRemodAdd', 'YearRemodAddBin')

bin.cols <- c('GarageYrBltBin', 'YearBuiltBin', 'YearRemodAddBin')

for (col in bin.cols) {
  df.categoric <- cbind(df.categoric, model.matrix(~.-1, df.categoric[col]))
}

# lets drop the orginal 'GarageYrBltBin', 'YearBuiltBin', 'YearRemodAddBin' from our dataframe
df.categoric <- df.categoric[,!names(df.categoric) %in% bin.cols]
df <- cbind(df.numeric, df.categoric)

# Distribution of sale prices

require(WVPlots)
y.true <- train$SalePrice[which(!1:1460 %in% idx.outliers)]

qplot(y.true, geom='density') +# +(train, aes(x=SalePrice)) +
  geom_histogram(aes(y=..density..), color='black', 
                 fill='cornflowerblue', alpha=.5, bins = 75) +
  geom_line(aes(y=..density..), color='cornflowerblue', lwd = 1, stat = 'density') + 
  stat_function(fun = dnorm, colour = 'indianred', lwd = 1, args = 
                  list(mean(train$SalePrice), sd(train$SalePrice))) +
  scale_x_continuous(breaks = seq(0,800000,100000), labels = dollar) +
  scale_y_continuous(labels = comma) +
  theme_minimal() +
  annotate('text', label = paste('skewness =', signif(skewness(train$SalePrice),4)),
           x=500000,y=7.5e-06)

qqnorm(train$SalePrice)
qqline(train$SalePrice)

# Log Transformation as data is not normal

y_train <- log(y.true+1)

qplot(y_train, geom = 'density') +
  geom_histogram(aes(y=..density..), color = 'black', fill = 'cornflowerblue', alpha = .5, bins = 75) +
  scale_x_continuous(breaks = seq(0,800000,100000), labels = comma) +
  geom_line(aes(y=..density..), color='dodgerblue4', lwd = 1, stat = 'density') + 
  stat_function(fun = dnorm, colour = 'indianred', lwd = 1, args = 
                  list(mean(y_train), sd(y_train))) +
  scale_y_continuous(labels = comma) +
  theme_minimal() +
  annotate('text', label = paste('skewness =', signif(skewness(y_train),4)),
           x=13,y=1) +
  labs(x = 'log(SalePrice + 1)') 

paste('The dataframe has', dim(df)[1], 'rows and', dim(df)[2], 'columns')

# Removing zero variance factors from the data with nearZeroVar() from caret package

nzv.data <- nearZeroVar(df, saveMetrics = TRUE)
drop.cols <- rownames(nzv.data)[nzv.data$nzv == TRUE]
df <- df[,!names(df) %in% drop.cols]
paste('The dataframe now has', dim(df)[1], 'rows and', dim(df)[2], 'columns')

# splitting back into train and test

x_train <- df[1:1456,]
head(x_train)
x_test <- df[1457:nrow(df),]


install.packages("xgboost")
require(xgboost)

dtrain <- xgb.DMatrix(as.matrix(x_train), label = y_train)
dtest <- xgb.DMatrix(as.matrix(x_test))
summary(dtrain)

# Parameter tuning for avoiding overfitting 

cv.ctrl <- trainControl(method = "repeatedcv", repeats = 1,number = 4, 
                        allowParallel=T)
xgb.grid <- expand.grid(nrounds = 50,
                        eta = c(0.01,0.005,0.001),
                        max_depth = c(4,6,8),
                        colsample_bytree=c(0,1,10),
                        min_child_weight = 2,
                        subsample=c(0,0.2,0.4,0.6),
                        gamma=0.01)
set.seed(45)

require(plyr)
require(dplyr)

xgb_tune <- train(as.matrix(x_train),
        y_train,
        method="xgbTree",
        trControl=cv.ctrl,
        tuneGrid=xgb.grid,
        verbose=T,
        metric="RMSE",
        nthread =1)

xgb_params <- list(
  booster = 'gbtree',
  objective = 'reg:linear',
  colsample_bytree=1,
  eta=0.005,
  max_depth=4,
  min_child_weight=3,
  alpha=0.3,
  lambda=0.4,
  gamma=0.01, # less overfit
  subsample=0.6,
  seed=5,
  silent=TRUE)

xgb.cv(xgb_params, dtrain, nrounds = 5000, nfold = 4, early_stopping_rounds = 500)

# for estimating 10 most important features

bst <- xgb.train(xgb_params,dtrain, nrounds = 1000)#, early_stopping_rounds = 300, watchlist = list(train=dtrain))
model.names <- dimnames(dtrain)[[2]]
importance_matrix <- xgb.importance(model.names, model = bst)
xgb.plot.importance(importance_matrix[1:10])

# RMSE  evaluation

rmse_eval <- function(y.true, y.pred) {
  mse_eval <- sum((y.true - exp(y.pred)-1)^2) / length(y.true)
  return(sqrt(mse_eval))
}
y_pred.xgb <- predict(bst, dtrain)
rmse_eval(y.true, y_pred.xgb)

# ridge, lasso, elasticnet

glm.cv.ridge <- cv.glmnet(as.matrix(x_train), y_train, alpha = 0)
glm.cv.lasso <- cv.glmnet(as.matrix(x_train), y_train, alpha = 1)
glm.cv.net <- cv.glmnet(data.matrix(x_train), y_train, alpha = 0.001)

# use the lamdba that minimizes the error

penalty.ridge <- glm.cv.ridge$lambda.min
penalty.lasso <- glm.cv.lasso$lambda.min
penalty.net <- glm.cv.net$lambda.min

glm.ridge <- glmnet(x = as.matrix(x_train), y = y_train, alpha = 0, lambda = penalty.ridge )
glm.lasso <- glmnet(x = as.matrix(x_train), y = y_train, alpha = 1, lambda = penalty.lasso)
glm.net <- glmnet(x = as.matrix(x_train), y = y_train, alpha = 0.001, lambda = penalty.net)

y_pred.ridge <- as.numeric(predict(glm.ridge, as.matrix(x_train)))
y_pred.lasso <- as.numeric(predict(glm.lasso, as.matrix(x_train)))
y_pred.net <- as.numeric(predict(glm.net, as.matrix(x_train)))

rmse_eval(y.true, y_pred.ridge)
rmse_eval(y.true, y_pred.lasso)
rmse_eval(y.true, y_pred.net)

y_pred.ridge <- as.double(predict(glm.ridge, as.matrix(x_test)))
y_pred.lasso <- as.double(predict(glm.lasso, as.matrix(x_test)))
y_pred.net <- as.double(predict(glm.net, as.matrix(x_test)))

y_pred.ridge <- as.double(exp(y_pred.ridge) - 1)
y_pred.lasso <- as.double(exp(y_pred.lasso) - 1)
y_pred.net <- as.double(exp(y_pred.net) - 1)

y_pred.xgb <- as.double(predict(bst, dtest))
y_pred.xgb <- as.double(exp(y_pred.xgb) - 1)

# take the average of our predictions for our ensemble

y_pred <- (y_pred.xgb + y_pred.ridge + y_pred.lasso + y_pred.net)/4.0
head(y_pred)


