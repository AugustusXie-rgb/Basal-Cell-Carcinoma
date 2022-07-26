% create_val

imgPath='D:\Basal_Cell_Carcinoma\Normal skin sort\bmp\NS_1_train\';
val_output_path='D:\Basal_Cell_Carcinoma\Normal skin sort\bmp\NS_1_val\';
test_output_path='D:\Basal_Cell_Carcinoma\Normal skin sort\bmp\NS_1_test\';
imgDir=dir([imgPath '*.bmp']);
val_length=round(0.1*length(imgDir));
test_length=round(0.1*length(imgDir));

list_num=randperm(length(imgDir),val_length);
for i=1:val_length
    temp=imread([imgPath imgDir(list_num(i)).name]);
%     outputname=erase(imgDir(list_num(i)).name,'.tif');
    imwrite(temp,[val_output_path imgDir(list_num(i)).name]);
    delete([imgPath imgDir(list_num(i)).name]);
    ['val ',int2str(i)]
end

imgDir=dir([imgPath '*.bmp']);
list_num=randperm(length(imgDir),test_length);
for i=1:test_length
    temp=imread([imgPath imgDir(list_num(i)).name]);
    imwrite(temp,[test_output_path imgDir(list_num(i)).name]);
    delete([imgPath imgDir(list_num(i)).name]);
    ['test ',int2str(i)];
end