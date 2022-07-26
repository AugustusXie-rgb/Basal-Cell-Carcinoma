% svm_1_4_16
% input B_1, B_4, B_16, N_1, N_4, N_16, B_test_1, B_test_4, B_test_16, N_test_1, N_test_4, N_test_16
B_comb=[B_4,B_16(1:4:end),B_16(2:4:end),B_16(3:4:end),B_16(4:4:end)];
N_comb=[N_4,N_16(1:4:end),N_16(2:4:end),N_16(3:4:end),N_16(4:4:end)];
input_x=[B_comb;N_comb];
input_y=[ones(length(B_comb),1);zeros(length(N_comb),1)];

% model_416=fitcsvm(input_x,input_y,'KernelFunction','polynomial','PolynomialOrder',2);
model_416=fitcsvm(input_x,input_y,'KernelFunction','linear');

[label_416,val_416]=predict(model_416,input_x);
pre_416=1./(1+exp(val_416(:,1)));
% val_416=abs(val_416)/max(max(val_416));
% label_416=2*label_416-1;
% pre_416=0.5*(1+label_416.*val_416(:,1));
pre_416_B=pre_416(1:length(B_comb));
pre_416_N=pre_416(length(B_comb)+1:end);

B_comb=[B_1,pre_416_B(1:4:end),pre_416_B(2:4:end),pre_416_B(3:4:end),pre_416_B(4:4:end)];
N_comb=[N_1,pre_416_N(1:4:end),pre_416_N(2:4:end),pre_416_N(3:4:end),pre_416_N(4:4:end)];
input_x=[B_comb;N_comb];
input_y=[ones(length(B_comb),1);zeros(length(N_comb),1)];

% model_14=fitcsvm(input_x,input_y,'KernelFunction','polynomial','PolynomialOrder',2);
model_14=fitcsvm(input_x,input_y,'KernelFunction','linear');

B_test=[B_test_4,B_test_16(1:4:end),B_test_16(2:4:end),B_test_16(3:4:end),B_test_16(4:4:end)];
N_test=[N_test_4,N_test_16(1:4:end),N_test_16(2:4:end),N_test_16(3:4:end),N_test_16(4:4:end)];
test_x=[B_test;N_test];
[~,val]=predict(model_416,test_x);
% label=2*label-1;
% val=abs(val)/max(max(val));
% test_416=0.5*(1+label.*val(:,1));
test_416=1./(1+exp(val(:,1)));
test_416_B=test_416(1:length(B_test));
test_416_N=test_416(length(B_test)+1:end);

B_test=[B_test_1,test_416_B(1:4:end),test_416_B(2:4:end),test_416_B(3:4:end),test_416_B(4:4:end)];
N_test=[N_test_1,test_416_N(1:4:end),test_416_N(2:4:end),test_416_N(3:4:end),test_416_N(4:4:end)];
[~,val_B]=predict(model_14,B_test);
[~,val_N]=predict(model_14,N_test);
% label=2*label-1;
% val=abs(val)/max(max(val));
% final=0.5*(1+label.*val(:,1));
final_B=1./(1+exp(val_B(:,1)));
final_N=1./(1+exp(val_N(:,1)));