%% Compute confidence intervals by formula
%% Simulate Data
popN = 1e7; 

% The Data
population = (4*randn(popN,1)).^2;

% Calculate the exact population mean
popMean = mean(population);

% Let's see it
figure(1)
subplot(211)
% only plot every 1000th sample
plot(population(1:1000:end),'ks')
xlabel('Data index'), ylabel('Data value')

subplot(212)
histogram(population,'BinMethod','fd')
ylabel('Count'), xlabel('Data value')

%% Random Sample
% Parameters
samplesize = 40;
confidence = 95; % in percent

% Compute sample mean
randSamples = randi(popN,samplesize,1);
samplemean  = mean(population(randSamples));
samplestd   = std(population(randSamples));

% Compute confidence intervals
citmp = (1-confidence/100)/2;
confint = samplemean + tinv([citmp 1-citmp],samplesize-1) * samplestd/sqrt(samplesize);

% Graph everything
figure(2), hold on
patch(confint([1 1 2 2]),[0 1 1 0],'g','facealpha',.5,'edgecolor','none')
plot([1 1]*popMean,[0 1.5],'k:','linew',2)
plot([1 1]*samplemean,[0 1],'r--','linew',3)
set(gca,'xlim',[popMean-30 popMean+30],'ytick',[])
xlabel('Data values')
legend({[ num2str(confidence) '% CI region' ];'True mean';'Sample mean'},'box','off')

%% Large Number of Samples
% Parameters
samplesize = 50;
confidence = 95; % in percent
numExperiments = 5000;

withinCI = zeros(numExperiments,1);

% Part of the CI computation can be done outside the loop
citmp = (1-confidence/100)/2;
CI_T  = tinv([citmp 1-citmp],samplesize-1);
sqrtN = sqrt(samplesize);

for expi=1:numExperiments
    % Compute sample mean and CI as above
    randSamples = ceil(popN*rand(samplesize,1));
    samplemean  = mean(population(randSamples));
    samplestd   = std(population(randSamples));
    confint     = samplemean + CI_T * samplestd/sqrtN;
    
    % Determine whether the True mean is inside this CI
    if popMean>confint(1) && popMean<confint(2)
        withinCI(expi) = 1;
    end
end

fprintf('%g%% of sample C.I.''s contained the true population mean.\n',100*mean(withinCI))

%% end.