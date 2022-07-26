% imseq_resize

seq_len = 36;

folder_path = 'F:\Basal_Cell_Carcinoma\comp_data\seq\N_seq_1cut\';
output_path = 'D:\Basal Cell Carcinoma\imseq_resize\comp\N\';
subdir = dir(folder_path);

for i=3:length(subdir)
    imdir = dir([folder_path, subdir(i).name, '\', '*.bmp']);
    mkdir(output_path, subdir(i).name);
    total_length = length(imdir);
    count = 1;
    for j=1:(total_length-1)/(seq_len-1):total_length
        ind = round(j);
        im = imread([folder_path, subdir(i).name, '\', imdir(ind).name]);
        im = imresize(im, [224, 224]);
        imwrite(im, [output_path, subdir(i).name, '\', int2str(count), '.bmp']);
        count = count+1;
    end
    i
end
        
    