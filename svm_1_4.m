% svm_1_4
% input B_1, B_4, N_1, N_4, B_test_1, B_test_4, N_test_1, N_test_4

B_comb=[B_1,B_4(1:4:end),B_4(2:4:end),B_4(3:4:end),B_4(4:4:end)];
N_comb=[N_1,N_4(1:4:end),N_4(2:4:end),N_4(3:4:end),N_4(4:4:end)];
input_x=[B_comb;N_comb];
input_y=[ones(length(B_comb),1);zeros(length(N_comb),1)];

% model_14=fitcsvm(input_x,input_y,'KernelFunction','polynomial','PolynomialOrder',2);
model_14=fitcsvm(input_x,input_y,'KernelFunction','linear');

B_test=[B_test_1,B_test_4(1:4:end),B_test_4(2:4:end),B_test_4(3:4:end),B_test_4(4:4:end)];
N_test=[N_test_1,N_test_4(1:4:end),N_test_4(2:4:end),N_test_4(3:4:end),N_test_4(4:4:end)];
[~,val_B]=predict(model_14,B_test);
[~,val_N]=predict(model_14,N_test);
% label=2*label-1;
% val=abs(val)/max(max(val));
% final=0.5*(1+label.*val(:,1));
final_B=1./(1+exp(val_B(:,1)));
final_N=1./(1+exp(val_N(:,1)));