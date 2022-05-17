%% Central limit theorem in action
%% Create data from a power-law distribution
% Data
N = 1000000;
data = randn(N,1).^2;
% alternative data
% data = sin(linspace(0,10*pi,N));

% Show the distribution
figure(1)
subplot(211)
plot(data,'.')

subplot(212)
histogram(data,40)

%% Repeated samples of the mean
samplesize   = 30;
numberOfExps = 500;
samplemeans  = zeros(numberOfExps,1);

for expi=1:numberOfExps
    % Get a sample and compute its mean
    sampleidx = randi(N,samplesize,1);
    samplemeans(expi) = mean(data( sampleidx ));
end

% Show its distribution
figure(2)
histogram(samplemeans,30)
xlabel('Mean estimate'), ylabel('Count')

%% Linear mixtures
% Create two datasets with non-Gaussian distributions
x = linspace(0,6*pi,10001);
s = sin(x);
u = 2*rand(size(x))-1;

figure(3)
subplot(231)
plot(x,s,'linew',3,'color','b')
title('Signal')

subplot(234)
[y,xx] = hist(s,200);
plot(xx,y,'linew',3,'color','b')
title('Distribution')

subplot(232)
plot(x,u,'linew',3,'color','m')
title('Signal')

subplot(235)
[y,xx] = hist(u,200);
plot(xx,y,'linew',3,'color','m')
title('Distribution')

subplot(233)
plot(x,s+u,'linew',3,'color','k')
title('Combined signal')

subplot(236)
[y,xx] = hist(s+u,200);
plot(xx,y,'linew',3,'color','k')
title('Combined distribution')

%% end.