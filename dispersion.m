%% Computing dispersion
% The distributions
N = 10001; % number of data points
nbins = 30; % number of histogram bins

d1 = randn(N,1) - 1;
d2 = 3*randn(N,1);
d3 = randn(N,1) + 1;

% Need their histograms
[y1,x1] = histcounts(d1,nbins);
x1 = (x1(1:end-1)+x1(2:end))/2;

[y2,x2] = histcounts(d2,nbins);
x2 = (x2(1:end-1)+x2(2:end))/2;

[y3,x3] = histcounts(d3,nbins);
x3 = (x3(1:end-1)+x3(2:end))/2;

% Plot
figure(1)
subplot(5,1,2:4), hold on

plot(x1,y1,'b','linew',2)
plot(x2,y2,'r','linew',2)
plot(x3,y3,'k','linew',2)

xlabel('Data values')
ylabel('Data counts')

%% Overlay the mean
% Compute means
mean_d1 = sum(d1) / length(d1);
mean_d2 = mean(d2);
mean_d3 = mean(d3);

% Plot
plot([1 1]*mean_d1,[0 max(y1)],'b:','linew',3)
plot([1 1]*mean_d2,[0 max(y2)],'r:','linew',3)
plot([1 1]*mean_d3,[0 max(y3)],'k:','linew',3)

%% Standard deviation
% Initialize
stds = zeros(3,1);

% Compute standard deviations in a weird loop
for i=1:3
    eval(sprintf('stds(%g) = std(d%g);',i,i))
end

% Plot on top
plot([mean_d1-stds(1) mean_d1+stds(1)],[.4 .4]*max(y1),'b','linew',10)
plot([mean_d2-stds(2) mean_d2+stds(2)],[.5 .5]*max(y2),'r','linew',10)
plot([mean_d3-stds(3) mean_d3+stds(3)],[.6 .6]*max(y3),'k','linew',10)

%% Different variance measures
vars = 1:10;
N = 300;

varmeasures = zeros(4,length(vars));

for i=1:length(vars)
    % Create data and mean-center
    data = randn(N,1) * vars(i);
    datacent = data - mean(data);
    
    % Variance
    varmeasures(1,i) = sum(datacent.^2) / (N-1);
    
    % "Biased" variance
    varmeasures(2,i) = sum(datacent.^2) / N;
    
    % Standard deviation
    varmeasures(3,i) = sqrt( sum(datacent.^2) / (N-1) );
    
    % MAD (mean absolute difference)
    varmeasures(4,i) = sum(abs(datacent)) / (N-1);
end

% Show
figure(2)
plot(vars,varmeasures,'linew',3)
legend({'Var';'biased var';'Std';'MAD'})

%% Fano factor and coefficient of variation (CV)
% Beed positive-valued data
data = poissrnd(3,300,1); % "Poisson noise"

figure(3)
subplot(211)
plot(data,'s','markerfacecolor','k')
title('Poisson noise')

subplot(212)
histogram(data,30)

%% Compute fano factor and CV for a range of lambda parameters
% List of parameters
lambdas = linspace(1,12,15);

% Initialize output vectors
[cv,fano] = deal( zeros(length(lambdas),1) );

for li = 1:length(lambdas)
    % Generate new data
    data = poissrnd(lambdas(li),1000,1);
    
    % Compute the metrics
    cv(li) = std(data) / mean(data);
    fano(li) = var(data) / mean(data);
end

% Plot
figure(4), hold on
plot(lambdas,cv,'bs-','markersize',8,'markerfacecolor','b')
plot(lambdas,fano,'ro-','markersize',8,'markerfacecolor','r')
legend({'CV';'Fano'})
xlabel('$\lambda$','Interpreter','latex')
ylabel('CV or Fano')

%% end.