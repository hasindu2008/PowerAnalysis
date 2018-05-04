#step value for number of traces
i=1000

#minimum number of traces to test
min=1000

#maximum number of traces to test
max=20000

while [ $min -le $max ]
do
echo "CUDA : doing for traces $min" 
sed "s/#define SAMPLES 1000/#define SAMPLES $min/g" <kernel.cu >kerneltemp.cu
nvcc kerneltemp.cu helpers.cu -arch=sm_20 
./a.out ../wave.txt ../plain.txt > "$min".txt
cp all.txt all_"$min".txt
rm all.txt
echo ""
min=$(echo $min+$i | bc)
done

