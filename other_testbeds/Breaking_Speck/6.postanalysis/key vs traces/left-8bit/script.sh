#step value for number of traces
i=10

#minimum number of traces to test
min=10

#maximum number of traces to test
max=200

while [ $min -le $max ]
do
echo "CUDA : doing for traces $min" 
sed "s/#define SAMPLES 1000/#define SAMPLES $min/g" <kernelback.cu >kernel.cu
nvcc kernel.cu helpers.cu -arch=sm_20 
./a.out wave.txt plain.txt > "$min".txt
cp all.txt all_"$min".txt
rm all.txt
echo ""
min=$(echo $min+$i | bc)
done

