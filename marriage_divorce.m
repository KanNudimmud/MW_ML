%% Example with real marriage/divorce data
%% data urls
marriage_url = 'https://www.cdc.gov/nchs/data/dvs/state-marriage-rates-90-95-99-19.xlsx';
divorce_url  = 'https://www.cdc.gov/nchs/data/dvs/state-divorce-rates-90-95-99-19.xlsx';

%% Import Marriage Data
% Raw data
data = table2cell( webread(marriage_url) );

% Initialize matrices
M       = zeros(51,23);
statesM = cell(51,1);
yearM   = zeros(23,1);

% Import the data; start with a loop over columns
for coli = 1:length(yearM) 
    % Get year (same for all rows...)
    % some columns are text, others are numeric
    yearval = data{5,coli+1};
    if isnumeric(yearval)
        yearM(coli) = yearval;
    else
        yearM(coli) = str2double(yearval);
    end

    % Loop over rows
    for rowi = 1:51 
        % Get value from this cell and convert to number
        val = data{rowi+6,coli+1};
        if isnumeric(val)
            M(rowi,coli) = val;
        else
            M(rowi,coli) = str2double(val);
        end
        
        % Get state label (only in first colum)
        if coli==1
            statesM{rowi} = data{rowi+6,1};
        end
    end % end row loop
end % end column loop

%% Clean the Marriage Data
figure(1)
imagesc(yearM,[],M)
colorbar
set(gca,'clim',[0 10])
xlabel('Year'), ylabel('State')

% Replace with column median
[nanrow,nancol] = find(isnan(M));
for i=1:length(nanrow)
    M(nanrow(i),nancol(i)) = nanmedian(M(:,nancol(i)));
end

%% Plot time courses of each state
figure(2)
subplot(311)
plot(yearM,M)
ylabel('M. rate (per 1k)')
title('Marriage rates over time')

subplot(312)
plot(yearM,zscore(M,[],2))
ylabel('M. rate (z-norm)')

% Notice that x-axis is non-constant
subplot(313)
plot(yearM,mean(M),'ks-','markerfacecolor','w','markersize',8)
xlabel('Year')
ylabel('State-average M. rate')

figure(1)
imagesc(yearM,[],zscore(M,[],2))
xlabel('Year')
ylabel('State index')

%% Barplot of average marriage rate
% Average over time
meanMarriageRate = mean(M,2);

% Sort index
[~,sidx_M] = sort(meanMarriageRate);

figure(3)
subplot(211)
bar(meanMarriageRate(sidx_M))
set(gca,'xtick',1:51,'XTickLabel',statesM(sidx_M))
xtickangle(60)
ylabel('M. rate (per 1k)')
title('Marriage rates per state')

%% Correlation Matrix
figure(4)
imagesc(yearM,yearM,corr(M))
axis square
colorbar
set(gca,'clim',[.9 1])

%% PCA
[coeffM,scoreM,latentM,~,explainedM] = pca(M);

% Scree plot
figure(5), clf
plot(explainedM,'ks-','linew',2,'markerfacecolor','w','markersize',12)
ylabel('Percent variance explained')
xlabel('Component number')
title('PCA scree plot of marriage data')

%% Repeat for divorce rate data
%% Import data
% Raw data
data = table2cell( webread(divorce_url) );

% Initialize matrices
D       = zeros(51,23);
statesD = cell(51,1);
yearD   = zeros(23,1);

% Import the data; start with a loop over columns
for coli = 1:length(yearD)
    % Get year (same for all rows...)
    % some columns are text, others are numeric
    yearval = data{5,coli+1};
    if isnumeric(yearval)
        yearD(coli) = yearval;
    else
        yearD(coli) = str2double(yearval);
    end
    
    % Loop over rows
    for rowi = 1:51 
        % Get value from this cell and convert to number
        val = data{rowi+6,coli+1};
        if isnumeric(val)
            D(rowi,coli) = val;
        else
            D(rowi,coli) = str2double(val);
        end
        
        % Get state label (only in first colum)
        if coli==1
            statesD{rowi} = data{rowi+6,1};
        end
    end % end row loop
end % end column loop

%% Check if marriage and divorce datasets have the same year/state order
% should be zero
sum(yearD-yearM)

% should also be zero
sum( strcmpi(statesM,statesD)==0 )

% compare
[statesM statesD]

% find the difference
find(strcmpi(statesM,statesD)==0)

%% Clean Data
figure(6)
imagesc(yearD,[],D)
colorbar
set(gca,'clim',[0 10])

% Replace with column median
[nanrow,nancol] = find(isnan(D));
for i=1:length(nanrow)
    D(nanrow(i),nancol(i)) = nanmedian(D(:,nancol(i)));
end

%% Plot time courses of each state and average in black
figure(7)
subplot(311)
plot(yearM,D)
ylabel('D. rate (per 1k)')

subplot(312)
plot(yearM,zscore(D,[],2))
ylabel('D. rate (z-norm)')

% notice that x-axis is non-constant
subplot(313)
plot(yearM,mean(D),'ks-','markerfacecolor','w','markersize',8)
xlabel('Year')
ylabel('State-average D. rate')

figure(6)
imagesc(yearM,[],zscore(D,[],2))
xlabel('Year')
ylabel('State index')

%% Barplot
meanDivorceRate = mean(D,2);
[~,sidx_D] = sort(meanDivorceRate);

figure(3)
subplot(212)
bar(meanDivorceRate(sidx_D))
set(gca,'xtick',1:51,'XTickLabel',statesM(sidx_D))
xtickangle(60)
ylabel('M. rate (per 1k)')
title('Divorce rates per state')

%% PCA
[coeffD,scoreD,latentD,~,explainedD] = pca(D);

% Scree plot
figure(8)
plot(explainedD,'ks-','linew',2,'markerfacecolor','w','markersize',12)
ylabel('Percent variance explained')
xlabel('Component number')
title('PCA scree plot of divorce data')

%% More Visualization
%% Correlation Matrices
figure(9)
subplot(121)
imagesc(corr(M))
axis square
title('Correlation of marriages')
colorbar
set(gca,'clim',[.9 1])
set(gca,'xtick',1:3:length(yearM),'xticklabel',yearM(1:3:end),...
        'ytick',1:3:length(yearM),'yticklabel',yearM(1:3:end))
xtickangle(60)

subplot(122)
imagesc(corr(D))
axis square
title('Correlation of divorces')
colorbar
set(gca,'clim',[.7 1])
set(gca,'xtick',1:3:length(yearM),'xticklabel',yearM(1:3:end),...
        'ytick',1:3:length(yearM),'yticklabel',yearM(1:3:end),...
        'XTickLabelRotation',60)

%% Correlate over states
figure(10)
imagesc(corr(D'))
axis square
set(gca,'clim',[0 1])
colorbar
set(gca,'ytick',1:51,'yticklabel',statesD,'xtick',1:51,'xticklabel',statesD,'XTickLabelRotation',-60)

%% Inferential Statistics
%% Correlate M and D over time per state
% Bonferroni correction
pvalThresh = .05/51;

figure(11), hold on
color = 'rg';

for si=1:length(statesM)
    % Compute correlation
    [r,p] = corr(M(si,:)',D(si,:)');
    
    % Plot the data point
    plot([r 1],[si si],'-','color',ones(1,3)*.5)
    plot(r,si,'ks','markersize',12,'markerfacecolor',color((p<pvalThresh)+1))
end

ylabel('State'), xlabel('Correlation')
title('Marriage-divorce correlations per state')
set(gca,'ytick',1:51,'yticklabel',statesD,'ylim',[0 52])
set(gca,'xlim',[-1 1],'YAxisLocation','right')
plot([0 0],get(gca,'ylim'),'k--')

%% Have marriage/divorce rates really declined over time?
figure(12), hold on

% Initialize marriage vs. divorce rates
MvsD = zeros(size(M,1),1);

for rowi=1:size(M,1) 
    % Run regression
    lM = fitlm(yearM,M(rowi,:));
    lD = fitlm(yearD,D(rowi,:));
    
    % Normalize beta coefficients
    bM = lM.Coefficients.Estimate(2) / lM.Coefficients.SE(2);
    bD = lD.Coefficients.Estimate(2) / lD.Coefficients.SE(2);
    
    % Extract p-values
    pM = lM.Coefficients.pValue(2);
    pD = lD.Coefficients.pValue(2);
    
    % Plot
    subplot(211), hold on
    plot([rowi rowi],[bM bD],'k')
    plot(rowi,bM,'ko','markerfacecolor',color((pM<pvalThresh)+1))
    plot(rowi,bD,'ks','markerfacecolor',color((pD<pvalThresh)+1))
    
    subplot(212), hold on
    plot([rowi rowi],[bM-bD 0],'k-','color',ones(1,3)*.7)
    plot([rowi rowi],bM-bD,'ko','markerfacecolor','k')
    
    MvsD(rowi) = bM-bD;
end

for i=1:2
    subplot(2,1,i)
    set(gca,'xtick',1:51,'xticklabel',statesD,'xlim',[0 52],'XTickLabelRotation',60)
    plot(get(gca,'xlim'),[0 0],'k--')
end

subplot(211), ylabel('Decrease per year (norm.)')
subplot(212), ylabel('$\Delta$M - $\Delta$D','Interpreter','latex')

% t-test of marriage vs. divorce rate
[~,p,~,t] = ttest(MvsD);
title([ 'Marriage vs. divorce: t(' num2str(t.df,3) ')=' num2str(t.tstat) ', p=' num2str(p,3) ])

%% end.