% svm_running

BCC_y=ones(1,7908);
NS_y=zeros(1,7047);

input_x=[BCC,NS]';
input_y=[BCC_y,NS_y]';

model=svmtrain(input_x,input_y,'linear',Inf);