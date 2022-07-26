% ROC

% BCC=final_B;
% NS=final_N;
% BCC=B_test_1;
% NS=N_test_1;

all=[BCC;NS];
BCC_sort=sort(BCC);
NS_sort=sort(NS);
all_sort=sort(all);
min_bound=min(BCC);
max_bound=max(NS);
while all_sort(1)<min_bound
    all_sort(1)=[];
end
while all_sort(end)>=max_bound
    all_sort(end)=[];
end

TPR=[];
FPR=[];
for i=1:length(all_sort)
    threshold=all_sort(i);
    FP=0;
    TP=0;
    for j=1:length(BCC_sort)
        if BCC_sort(j)<threshold
            FP=FP+1;
        else
            TP=TP+1;
        end
    end
    TN=0;
    FN=0;
    for j=1:length(NS_sort)
        if NS_sort(j)<=threshold
            TN=TN+1;
        else
            FN=FN+1;
        end
    end
    
    TPR=[TPR,TP/(TP+FN)];
    FPR=[FPR,FP/(FP+TN)];
end

% TPR=[TPR];
% FPR=[FPR];
FPR=sort(FPR);
TPR=sort(TPR);
FPR=[0,0,FPR,FPR(end),1];
TPR=[0,TPR(1),TPR,1,1];
% ROC_curve=[TPR;FPR];
auc=100*sum((FPR(2:length(TPR))-FPR(1:length(TPR)-1)).*TPR(2:length(FPR)))
plot(sort(FPR),sort(TPR),'linewidth',1);
grid on
set(gca, 'XTick', [0:0.1:1]);
set(gca, 'YTick', [0:0.1:1]);
hold on
