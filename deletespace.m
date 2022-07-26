% deletespace

imgPath='F:\Basal_Cell_Carcinoma\BCC_subject_val\BCC_test_bmp\';
imgDir=dir([imgPath '*.bmp']);

for i=1:length(imgDir)
    im=imread([imgPath imgDir(i).name]);
    outputname=erase(imgDir(i).name,' ');
    imwrite(im,[imgPath outputname]);
    delete([imgPath imgDir(i).name]);
    i
end