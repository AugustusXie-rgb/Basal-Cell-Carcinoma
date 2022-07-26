% svm_test
test_1=cut;
test_4=cut1;
test_16=cut2;
test_416=[test_4,test_4,test_16(1:4:end),test_16(2:4:end),test_16(3:4:end),test_16(4:4:end)];
[~,val_416]=predict(model_416,test_416);
% val_416=abs(val_416)/max(max(val_416));
% label_416=2*label_416-1;
% pre_416=0.5*(1+label_416.*val_416(:,1));
pre_416=1./(1+exp(val_416(:,1)));

test_14=[test_1,test_1,pre_416(1:4:end),pre_416(2:4:end),pre_416(3:4:end),pre_416(4:4:end)];
[~,val_14]=predict(model_14,test_14);
% val_14=abs(val_14)/max(max(val_14));
% label_14=2*label_14-1;
% pre_14=0.5*(1+label_14.*val_14(:,1));
pre_14=1./(1+exp(val_14(:,1)));
plot(pre_14);
temp=cut;
clear cut cut1 cut2