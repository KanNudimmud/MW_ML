%% Sampling variability
% Theoretical normal distribution
x = linspace(-5,5,10101);
theoNormDist = normpdf(x);
% normalize to pdf
% theoNormDist = theoNormDist*mean(diff(x));

% Now for our experiment
numSamples = 40;

% Initialize
sampledata = zeros(numSamples,1);

% Run the experiment!
for expi = 1:numSamples
    sampledata(expi) = randn;
end

% Show the results
figure(1), hold on
histogram(sampledata,'Normalization','probability')
plot(x,theoNormDist,'r','linew',3)
xlabel('Data values'), ylabel('Probability')

%% Show the mean of samples of a known distribution
% Generate population data with known mean
populationN = 1000000;
population  = randn(populationN,1);
population  = population - mean(population); % demean

% Now we draw a random sample from that population
samplesize = 30;

% The random indices to select from the population
sampleidx = randi(populationN,samplesize,1);
samplemean = mean(population( sampleidx ));

%% Repeat for different sample sizes

samplesizes = 30:1000;

samplemeans = zeros(size(samplesizes));

for sampi=1:length(samplesizes)
    % Nearly the same code as above
    sampleidx = randi(populationN,samplesizes(sampi),1);
    samplemeans(sampi) = mean(population( sampleidx ));
end

% Show the results
figure(2), hold on
plot(samplesizes,samplemeans,'s-')
plot(get(gca,'xlim'),[1 1]*mean(population),'r','linew',3)
xlabel('sample size')
ylabel('mean value')
legend({'Sample means';'Population mean'})

%% end.