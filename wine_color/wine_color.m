%% Wine colour prediction via machine learning
% Wine Quality Data Set by UCL is used for this project.

%% Load the data 
redWine   = readtable("wine_red.csv");
whiteWine = readtable("wine_white.csv");

% Examine both tables
head(redWine)
head(whiteWine)

% Add a column to indicate colour (1 for RED, 0 for WHITE)
redWine.Colour  = ones(size(redWine,1),1); 
whiteWine.Colour = zeros(size(whiteWine,1),1);

% Concatenate white and red wines at single table
wineData = [redWine; whiteWine];

% Clean the data from duplicated recordings
clean_wineData = unique(wineData,'rows');

%% Analysis of the data
% Examine the summary statistics
summary(clean_wineData)

% Use PCA to find importance of variables
% Firstly, data should be numeric
numData = clean_wineData{:,1:end-1};

% Apply PCA
[~,score,~,~,pexp] = pca(numData);

% Create a pareto chart
headers = {'fixedAcidity', 'volatileAcidity', 'citricAcid',...
    'residualSugar', 'chlorides', 'freeSulfurDioxide',...
    'totalSulfurDioxide', 'density', 'pH', 'sulphates',...
    'alcohol', 'quality'};
figure(1)
subplot(131)
pareto(pexp,headers)
title('PCA for raw data')

% Normalize the data
% Determine max. values
max_vals = max(numData);

% Divide each column by their max. value
normData = bsxfun(@rdivide,numData,max_vals);

% Try PCA for normalized data
% Apply PCA
[~,nscore,~,~,npexp] = pca(normData);

% Create a pareto chart
figure(1)
subplot(1,3,2:3)
pareto(npexp,headers)
title('PCA for normalized data')

% Create a correlation matrix between features
cormat = corr(normData);
figure(2)
heatmap(headers,headers,cormat)
colormap turbo

%% Machine Learning part
% Determine input and output variables
X = normData;
Y = clean_wineData.Colour;

% Seperate to train and test samples (%70 for training)
cvpart = cvpartition(Y,'holdout',0.3);
Xtrain = X(training(cvpart),:);
Ytrain = Y(training(cvpart),:);
Xtest  = X(test(cvpart),:);
Ytest  = Y(test(cvpart),:);

% Fit a model
mdl = fitcensemble(Xtrain,Ytrain,'Method','Bag','NumLearningCycles',10)

% Plot misclassification of the test data
figure(4)
plot(loss(mdl,Xtest,Ytest,'mode','cumulative'))
xlabel('Number of trees')
ylabel('Test classification error')

%% Let's try to predict
% Input is the same with first row of the cleaned_wineData, white(0) is
% expected
Ypred = predict(mdl,[0.23,0.19,0.012,0.16,0.05,0.06,0.25,0.95,0.93,0.22,0.83,0.66])

% Now try a red(1) wine, 10th row of the cleaned_wineData
Ypred = predict(mdl,[0.28,0.32,0.09,0.03,0.08,0.02,0.14,0.95,0.97,0.28,0.87,0.44])

%% end