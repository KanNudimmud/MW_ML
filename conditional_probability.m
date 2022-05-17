%% Conditional Probability
%% Generate two long-spike time series
N = 10000;
spikeDur  = 10; % a.u. but must be an even number
spikeNumA = .01; % in proportion of total number of points
spikeNumB = .05; % in proportion of total number of points

% Initialize to zeros
[spike_tsA,spike_tsB] = deal( zeros(N,1) );

% Populate time series 1
spiketimesA = randi(N,round(N*spikeNumA),1);

% Flesh out spikes (loop per spike)
for spikei=1:length(spiketimesA)
    % Find boundaries
    bnd_pre = max(1,spiketimesA(spikei)-spikeDur/2);
    bnd_pst = min(N,spiketimesA(spikei)+spikeDur/2);
    
    % Fill in with ones
    spike_tsA(bnd_pre:bnd_pst) = 1;
end

% Repeat for time series 2
spiketimesB = randi(N,round(N*spikeNumB),1);
% spiketimesB(1:length(spiketimesA)) = spiketimesA; % induce strong conditional probability

% Flesh out spikes (loop per spike)
for spikei=1:length(spiketimesB) 
    % Find boundaries
    bnd_pre = max(1,spiketimesB(spikei)-spikeDur/2);
    bnd_pst = min(N,spiketimesB(spikei)+spikeDur/2);
    
    % Fill in with ones
    spike_tsB(bnd_pre:bnd_pst) = 1;
end

%% Let's see what they look like
figure(1)
plot(1:N,spike_tsA, 1:N,spike_tsB)
set(gca,'ylim',[0 1.2])

%% Compute their probabilities and intersection
% Probabilities
probA = sum(spike_tsA==1) / N;
probB = mean(spike_tsB);

% Joint probability
probAB = mean(spike_tsA+spike_tsB==2);

%% Compute the conditional probabilities
% p(A|B)
pAgivenB = probAB/probB;

% p(B|A)
pBgivenA = probAB/probA;

% Print a little report
disp('  ')
disp([ 'P(A)   = ' num2str(probA) ])
disp([ 'P(A|B) = ' num2str(pAgivenB) ])
disp([ 'P(B)   = ' num2str(probB) ])
disp([ 'P(B|A) = ' num2str(pBgivenA) ])

%% end.