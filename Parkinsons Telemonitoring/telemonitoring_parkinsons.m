%% Parkinsons Telemonitoring
% Data is acquired from  https://data.world/uci/parkinsons/workspace/file?filename=parkinsons.names.txt.
%% Load the data
parkinsons = readtable('telemonitoring_parkinsons.csv');

% Examine data
head(parkinsons)

% Remove subject and sex column since it is not relevant for the model
parkinsons = parkinsons(:,2:end);
parkinsons = parkinsons(:,[1 3:end]);

% Seperate data by  UPDRS whic is a scale of Parkinson
X = [parkinsons(:,1:2) parkinsons(:,5:end)];
Y = parkinsons(:,3:4);

% Transfrom table to double
X = table2array(X);
Y = table2array(Y);

%% Machine Learning
% Randomly split the data for train and test
idx_train = randperm(4875);
idx_test  = randperm(1000);

Xtrain = X(idx_train,:);    
Xtest  = X(idx_test,:);        
Ytrain = Y(idx_train,:);
Ytest  = Y(idx_test,:);

% For the model, make output a single column
Ytrain = Ytrain(:,2);
Ytest  = Ytest(:,2);

% Fit a model
mdl = fitcknn(Xtrain,Ytrain);

%% Visualize classifications of the fitted model
% Make predictions
Ypredicted = predict(mdl,Xtest);

% Create a plot to examine the orignal and predicted values
figure(1)
plot(Ytest,'o','MarkerSize',12),hold on
plot(Ypredicted,'o','MarkerSize',12), legend({'Original','Predicted'})

% Make sure of error rate
diff    = abs(Ytest-Ypredicted); % difference
errRate = nnz(diff) *100/ length(Ytest);
figure(2)
plot(diff),title('Difference between predicted and original value')

%% end