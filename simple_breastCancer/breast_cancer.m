%% Breast Cancer Detection
% Breast Cancer Wisconsin (Diagnostic) Data Set by UCL Machine Learning in
% kaggle
%% Load the data
bc_data = readtable('breastCancer.csv','Range','A1:AF570');

% Take a look 
head(bc_data)

% Make diagnosis values categorical
bc_data.diagnosis = categorical(bc_data.diagnosis);

%% Create a model 
% %70 of the data will be seperated for training
pt      = cvpartition(bc_data.diagnosis,'HoldOut',0.3);
bcTrain = bc_data(training(pt),:);
bcTest  = bc_data(test(pt),:);

% Fit a knn model
mdl_c = fitcknn(bcTrain,"diagnosis","NumNeighbors",5);
errRate = loss(mdl_c,bcTest);

%% Visualize classifications of the fitted model
% Make predictions
bcPredicted = predict(mdl_c,bcTest);

% Calculate the ratio of false negative which means the tumor is predictd benign but
% malignant in the real data
falseNeg = mean((bcTest.diagnosis == "M") & (bcPredicted == "B"));

% Create a confusion chart to see
figure(1)
subplot(121)
confusionchart(bcTest.diagnosis,bcPredicted)
title('Confusion matrix of the knn model')

%% Let's try an another model to see if we get better result
% Fit an ensemble model
mdl_c = fitcensemble(bcTrain,"diagnosis");
errRate_c = loss(mdl_c,bcTest);

% Make predictions
bcPredicted_c = predict(mdl_c,bcTest);

% Calculate the ratio of false negative
falseNeg_c = mean((bcTest.diagnosis == "M") & (bcPredicted_c == "B"));

% Create a confusion chart to see
figure(1)
subplot(122)
confusionchart(bcTest.diagnosis,bcPredicted_c)
title('Confusion matrix of the ensemble model')

%% end