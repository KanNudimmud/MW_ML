%% Data trimming to remove outliers
%% Create some data
N = 40;
data = randn(N,1);
data(data<-2) = -data(data<-2).^2;
data(data>2) = data(data>2).^2;

% Mean-centered data
dataMC = data - mean(data);

% Plot them
figure(1)
subplot(5,1,2:4), hold on
plot(data,'k^','markerfacecolor','y','markersize',12);
set(gca,'xtick',[],'ytick',[])
xlabel('Data index')
ylabel('Data value')

%% Show the mechanism
figure(2)
subplot(121)
plot(data,'k^','markerfacecolor','y','markersize',12);
set(gca,'xtick',[]), axis square
title('Original data')
xlabel('Original data index')

subplot(122)
plot(sort(abs(dataMC)),'ko','markerfacecolor','g','markersize',12)
set(gca,'xtick',[]), axis square
title('Sorted data')
xlabel('Sorted data index')

%% Option 1: remove k% of the data
% Percent of "extreme" data values to remove
trimPct = 5; % in percent

% Identify the cut-off (note the abs() )
datacutoff = prctile(abs(dataMC),100-trimPct);

% Find the exceedance data values
data2cut = find( abs(dataMC)>datacutoff );

% Mark those off
figure(1)
plot(data2cut,data(data2cut),'rx','markersize',15,'linew',3)

%% Option 2: remove k most extreme values
% Number of "extreme" data values to remove
k2remove = 3; % in number

% Find the exceedance data values
[~,datasortIdx] = sort(abs(dataMC),'descend');
data2cut = datasortIdx(1:k2remove);

% Mark those off
plot(data2cut,data(data2cut),'go','markersize',15,'linew',3)

% Finally, add a legend
legend({'All data';[ num2str(100-trimPct) '% threshold' ];[ num2str(k2remove) '-value threshold' ]})

%% end.