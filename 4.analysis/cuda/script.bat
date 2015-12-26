::	The A to Z of Building a Testbed for Power Analysis Attacks
::	Windows batch file to run the correlation power analysis algorithm 

::  Authors : Hasindu Gamaarachchi, Harsha Ganegoda and Roshan Ragel, 
::  Department of Computer Engineering, 
::  Faculty of Engineering, University of Peradeniya, 22 Dec 2015
 
::  For more information read 
::  Hasindu Gamaarachchi, Harsha Ganegoda and Roshan Ragel, 
::  "The A to Z of Building a Testbed for Power Analysis Attacks", 
::  10th IEEE International Conference on Industrial and Information Systems 2015 (ICIIS)]
 
::  Any bugs, issues or suggestions please email to hasindu2008@live.com

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

cls
@echo off

:: There are 3 methods to run from cmd

:: 1) script.bat
::		This will run the program with default arguments
::  2) script.bat <num_of_traces> <result_file>
::		This will run the program with <num_of_traces> number of power traces with other arguments set to default
::  3) script.bat <num_of_traces> <power_trace_file> <plain_text_file> <result_file> <power_trace_file_format> <sample_points> <sample_point_partition_size>
::		This will run with arguments as provided by you. FOr description of those arguments see below.
		
	
:: The number of power traces to be used for calculation. If your input has more than num_of_traces, the first num_of_traces will be used for calulation	
set num_of_traces=200

::File containing power traces
set power_trace_file=wave.dat

::File containing plain text samples
set plain_text_file=plain.txt

::file to output the result which is the most five correlated keys
set result_file=results.txt

::input file format for the power trace file
::#put ascii for ASCII format power traces. 
::#put binary to save power traces in binary. 
set power_trace_file_format=binary

::The number of sampling points in a single power trace
::This value can be found by inspecting the stat.txt generated after collecting a set of power traces*/
set sample_points=100000

::Global memory on a GPU is limited (RAM as well) and hence if the power traces are large sized, all the things won't fit at once to memory
::If you experience a memory allocation failure when running, reduce WAVELENGTH value
::This will force the program to read the power traces part by part 
::But when using this implementation make sure that ALLWAVELENGTH is divisible by WAVELENGTH 
set sample_point_partition_size=100000

::check number of args entered and handle appropriately
IF "%1" == "" GOTO noargs
IF "%3" == "" GOTO twoargs
IF "%8" == "" GOTO allargs

::if only two args (number of power traces and result file) are given	
:twoargs
	set num_of_traces=%1
	set result_file=%2
	echo.
	echo Only two arguments were entered.
	echo Hence assigning them for number of samples and result file and assuming default values for other arguments"
	echo If you do not like default args use as :  "eg. script.bat <num_of_traces> <power_trace_file> <plain_text_file> <result_file> <power_trace_file_format> <sample_points> <sample_point_partition_size>" 
	echo Check the script file comments for description of these arguments"
	echo.
	echo.
	goto next
	
::#if all arguments are provided
:allargs
	set	num_of_traces=%1
	set	power_trace_file=%2
	set	plain_text_file=%3
	set	result_file=%4
	set	power_trace_file_format=%5
	set	sample_points=%6
	set	sample_point_partition_size=%7
	goto next
	
::arguments are given wrong	
:noargs
@echo off
	echo.
	echo Arguments were not properly entered
	echo Hence using default arguments
	echo If you do not like default args use as :  "eg. script.bat <num_of_traces> <power_trace_file> <plain_text_file> <result_file> <power_trace_file_format> <sample_points> <sample_point_partition_size>"
	echo Check the script file comments for description of these arguments
	echo.
	echo.
	goto next

:next 

::print used parameters
echo Parameters being used are : 
echo num_of_traces : %num_of_traces%
echo power_trace_file : %power_trace_file%
echo plain_text_file : %plain_text_file%
echo result_file : %result_file%
echo power_trace_file_format : %power_trace_file_format%
echo sample_points : %sample_points%
echo sample_point_partition_size : %sample_point_partition_size%
echo.

::check if sed is installed
where sed > nul 2>nul
if %errorlevel%==1 (
	echo sed command is required
	echo Install mingw32 from "http://www.mingw.org/"
	echo Make sure you select msys and set the path to msys bin folder "(default is C:\mingw\msys\1.0\bin)"
	goto end
)
echo.

:: Generate a temporary script based on the provided parameters
copy cpa.cu cpatemp.cu
sed -i "s/#define SAMPLES .*/#define SAMPLES %num_of_traces%/" cpatemp.cu
if "%power_trace_file_format%" == "binary" (
	sed -i "s/#define WAVFORMAT .*/#define WAVFORMAT 1/" cpatemp.cu
)
if "%power_trace_file_format%" == "ascii" (
	sed -i "s/#define WAVFORMAT .*/#define WAVFORMAT 0/" cpatemp.cu
)
sed -i "s/#define ALLWAVELEN .*/#define ALLWAVELEN %sample_points%/" cpatemp.cu
sed -i "s/#define WAVELENGTH .*/#define WAVELENGTH %sample_point_partition_size%/" cpatemp.cu
	
::check if nvcc ins installed	
where nvcc > nul 2>nul
if %errorlevel%==1 (
	echo "nvcc command is required"
	echo Install CUDA toolkit
	echo Make sure you have set the paths
	goto end
)	
::compile using cuda. Please change sm_20 to match the appropriate compute capability of your graphics card
nvcc cpatemp.cu helpers.cu -arch=sm_20 -o cpa.exe

::run the CPA algorithm
cpa.exe %power_trace_file% %plain_text_file% > %result_file% 


:end
echo.

