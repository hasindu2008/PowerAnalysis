clear;clc;

traces=10000;
samplepoints=100000;
start=49200;
stop=50800;

input='wave.dat';
output='waveout.dat';

file_in=fopen(input,'r');
file_out=fopen(output,'w');

data=fread(file_in,[samplepoints,traces],'float');
data=data(start:stop,:);

fwrite(file_out,data,'float');

fprintf('sample size is : %d\r\n',stop-start+1);

fclose(file_in);
fclose(file_out);