%% Kendall Correlation
%% Generate Data
N = 40;

% Movie ratings
docuRatings = randi(5,N,1);

% Education level (1-4, correlated with docuRatings)
eduLevel = ceil( (docuRatings + randi(4,N,1) )/9 * 4 );

% Compute correlations
cr(1) = corr(docuRatings,eduLevel,'type','kendall');
cr(2) = corr(docuRatings,eduLevel,'type','pearson');
cr(3) = corr(docuRatings,eduLevel,'type','spearman');

% Plot the data
figure(1)
plot(eduLevel+randn(N,1)/30,docuRatings+randn(N,1)/30,'ks','markersize',15)
set(gca,'xlim',[.5 4.5],'xtick',1:4,'ytick',1:5,'ylim',[.5 5.5])
grid
xlabel('Education level')
ylabel('Documentary ratings')
title(sprintf('$r_k$=%g, $\\quad r_p$=%g, $\\quad r_s$=%g',round(cr,4)),'Interpreter','latex')

%% Correlation Estimation Errors Under H0
numExprs = 1000;
nValues = 50;
nCategories = 6;

c = zeros(numExprs,3);

for i=1:numExprs
    % Create data
    x = randi(nCategories,nValues,1);
    y = randi(nCategories,nValues,1);
    
    % Store correlations
    c(i,:) = [ corr(x,y,'type','k') corr(x,y,'type','p') corr(x,y,'type','s') ];
end

% Show the graph
figure(2), hold on
bar(mean(c.^2,1))
errorbar(mean(c.^2,1),std(c.^2,[],1),'.')
set(gca,'xtick',1:3,'XTickLabel',{'Kendall';'Pearson';'Spearman'})
ylabel('Squared correlation error')
title('Noise correlation ($r^2$) distributions','Interpreter','latex')

figure(3)
subplot(211)
plot(c(1:100,:),'s-')
xlabel('Experiment number')
ylabel('Correlation value')
legend({'K','P','S'})

subplot(212)
imagesc(corr(c)), axis square
set(gca,'clim',[.9 1])
set(gca,'xtick',1:3,'XTickLabel',{'K','P','S'},'ytick',1:3,'YTickLabel',{'K','P','S'})
colorbar
title('Correlation matrix')

%% end.