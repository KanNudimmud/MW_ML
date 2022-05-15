% Bootstrapping confidence intervals
%% Simulate Data
popN = 1e7;

% Data 
population = (4*randn(popN,1)).^2;

% We can calculate the exact population mean
popMean = mean(population);

% Let's see it
figure(1)
subplot(211)

% Only plot every 1000th sample
plot(population(1:1000:end),'ks')
xlabel('Data index'), ylabel('Data value')

subplot(212)
histogram(population,'BinMethod','fd')
ylabel('Count'), xlabel('Data value')

%% Draw a Random Sample
% Parameters
samplesize = 40;
confidence = 95; % in percent

% Compute sample mean
randSamples = randi(popN,samplesize,1);
sampledata  = population(randSamples);
samplemean  = mean(sampledata);
samplestd   = std(population(randSamples)); % used later for analytic solution

%%% Bootstrapping
numBoots  = 1000;
bootmeans = zeros(numBoots,1);

% Resample with replacement
for booti=1:numBoots
    bootmeans(booti) = mean( randsample(sampledata,samplesize,true) );
end

% Find confidence intervals
confint(1) = prctile(bootmeans,(100-confidence)/2);
confint(2) = prctile(bootmeans,100-(100-confidence)/2);

%% Graph the Outcomes
figure(2), hold on

% Start with a histogram of the resampled means
[y,x] = histcounts(bootmeans,40);
y=y./max(y);
x = (x(1:end-1)+x(2:end))/2;
bar(x,y)

patch(confint([1 1 2 2]),[0 1 1 0],'g','facealpha',.5,'edgecolor','none')
plot([1 1]*popMean,[0 1.5],'k:','linew',2)
plot([1 1]*samplemean,[0 1],'r--','linew',3)
set(gca,'xlim',[popMean-30 popMean+30],'ytick',[])
xlabel('Data values')
legend({'Empirical distribution';[ num2str(confidence) '% CI region' ];'True mean';'Sample mean'},'box','off')

%% Compare Against the Analytic Confidence Interval
% Compute confidence intervals
citmp = (1-confidence/100)/2;
confint2 = samplemean + tinv([citmp 1-citmp],samplesize-1) * samplestd/sqrt(samplesize);

fprintf('\n Empirical: %g - %g',round(confint,2))
fprintf('\n Analytic:  %g - %g\n',round(confint2,2))

%% end.