[ $# -ne 4 ] && echo "Please enter all args. eg. ./script.sh <numtraces> <wavefile> <plaintextfile> <result>" && exit

num=$1
wave=$2
plain=$3
result=$4

echo "CUDA : doing for $num traces" 
sed "s/#define SAMPLES 1000/#define SAMPLES $num/g" <kernel.cu >kerneltemp.cu
nvcc kerneltemp.cu helpers.cu -arch=sm_20 
./a.out $wave $plain > $result 
echo ""


