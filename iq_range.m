%% Inter-Quartile Range
%% "Discrete" Entropy
% Generate data
N = 1000;
numbers = ceil( 8*rand(N,1).^2 );

% Get counts and probabilities
u = unique(numbers);
probs = zeros(length(u),1);

for ui=1:length(u)
    probs(ui) = sum(numbers==u(ui)) / N;
end

% Compute entropy
entropee = -sum( probs.*log2(probs+eps) );

% Plot
figure(1)
bar(u,probs)
title([ 'Entropy = ' num2str(entropee) ])
xlabel('Data value'), ylabel('Probability')

%% For Random Variables
% Create Brownian noise
N = 1123;
brownnoise = cumsum( sign(randn(N,1)) );

figure(1)
subplot(211)
plot(brownnoise)
xlabel('Data index'), ylabel('Data value')
title('Brownian noise')

subplot(212)
histogram(brownnoise,30,'Normalization','probability')
xlabel('Data value'), ylabel('Probability')

%% Compute entropy
% Number of bins
nbins = 50;

% Bin the data and convert to probability
[nPerBin,bins] = histcounts(brownnoise,nbins);
probs = nPerBin ./ sum(nPerBin);

% Compute entropy
entro = -sum( probs.*log2(probs+eps) );
title([ 'Entropy = ' num2str(entro) ])

%% end.