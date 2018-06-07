/*
   
   The A to Z of Building a Testbed for Power Analysis Attacks
   CCS PIC C source code for AES cryptographic algorithms 
   Configured for PIC18F2550
   
   Communication with the computer happens using USB through a USB to RS232 TTL converter such as FTDI 232RL
   Sits in an infinite loop to, 
   accept a 128 bit plain text sample from the computer, 
   encrypt using 128 bit AES key and sends the encrypted text back to the computer.
 
   Authors : Hasindu Gamaarachchi, Harsha Ganegoda and Roshan Ragel, 
   Department of Computer Engineering, 
   Faculty of Engineering, University of Peradeniya, 22 Dec 2015
 
   For more information read 
   Hasindu Gamaarachchi, Harsha Ganegoda and Roshan Ragel, 
   "The A to Z of Building a Testbed for Power Analysis Attacks", 
   10th IEEE International Conference on Industrial and Information Systems 2015 (ICIIS)]
 
   Any bugs, issues or suggestions please email to hasindu2008@live.com

   VERSION 2 : NEW ADDITIONS
   - key is not hardcoded anymore. Use the switch connected to INT1 to set the key to EEPROM
   - Switch to reset the device. MCLR has been enabled and a switch is connected.
   - power up timer enabled
   - use a 16MHz crystal instead of 8MHz
   

*/

/************************************************ DEVICE DEPENDENT CONFIGURATION *******************************************************/

// The header file for the microcontroller. Change this if your microcontroller is different
#include <18F2550.h>

//configurations bits. Note that these changes depending on the microcontroller
#fuses HSPLL,NOWDT,NOPROTECT,NOLVP,NODEBUG,PLL5,CPUDIV1,NOVREGEN,NOBROWNOUT,MCLR,PUT  
/*
HSPLL - High Speed Crystal/Resonator with PLL enabled. HSPLL requires the crystal to be >=4MHz
NOWDT - disable watch dog timer      
NOPROTECT - Code not protected from reading
NOLVP - No low voltage programming, BB5 used for I/O
NODEBUG - No Debug mode for ICD
PLL5 - Divide By 2(20MHz oscillator input). The input crystal frequency must be divided and brought to 4MHz to be fed to the PLL. PLL converts the 4MHz signal to 96MHz. Since our crustal is 20MHz we divide by 5 to bring it to 4MHz by specifying PLL5
CPUDIV1 - No System Clock Postscaler.
NOVREGEN - Internal voltage regulator disabled
NOBROWNOUT - No brownout reset
MCLR - master clear reset enabled
PUT - power up timer enabled
*/
//configuration is such that a 16MHz crystal input is converted to operate at 48MHz

//the effective clock frequency (48MHz) to be used for things like serial port communication, sleep etc
#use delay(clock=48000000)
//settings for the UART
#use rs232(UART1,baud=9600,parity=N,bits=8)

#include "kpd.c"
#include <stdlib.h>

/********************************************* AES IMPLEMENTATION *********************************************************************/
//This AES implementation is based on F. Finfe. (2014, Aug.) Advanced encryption standard (aes) on embedded system. [Online]. 
//Available: http://www.on4jx.net/microcontroller/AESonEmbedded.php

// Key size in bits (could be equal 128, 192 or 256)
#define key_size 128

// rappel: round_key = Nb(Nr+1)*4
// 256 bit = 240 bytes (15 keys of 16 bit)
// 192 bit = 208 bytes (13 keys)
// 128 bit = 176 bytes (11 keys)

// The number of 32 bit words in the key.
#define Nk                 (key_size / 32)
// The number of rounds in AES Cipher.
#define Nr                 (Nk + 6)

// This function produces Nb(Nr+1) round keys. The round keys are used in each round to encrypt the states.
void KeyExpansion();
// This function adds the round key to state.
// The round key is added to the state by an XOR function.
void AddRoundKey(unsigned char round);
// Cipher is the main function that encrypts the PlainText.
void Cipher();
// InvCipher is the main function that decrypts the CipherText.
void InvCipher();
// MixColumns function mixes the columns of the state matrix.
void InvMixColumns();
void MixColumns();
void ShiftRows();
void SubBytes();
void InvShiftRows();
unsigned char getS(unsigned char num);

const unsigned char sbox[256] =   {
    //0     1    2      3     4    5     6     7      8    9     A      B    C     D     E     F
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76, //0
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0, //1
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15, //2
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75, //3
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84, //4
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf, //5
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8, //6
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2, //7
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73, //8
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb, //9
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, //A
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08, //B
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a, //C
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e, //D
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf, //E
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16 };



const unsigned char Roundcon[255] = {
    0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a,
    0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39,
    0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a,
    0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8,
    0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef,
    0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc,
    0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b,
    0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3,
    0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94,
    0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20,
    0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35,
    0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f,
    0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04,
    0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63,
    0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd,
    0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb  };

// decryption
const unsigned char rsbox[256] = {
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d };


// The number of columns comprising a state in AES. This is a constant in AES. Value=4
#define Nb 4

// in - it is the array that holds the plain text to be encrypted.
// out - it is the array that holds the output CipherText after encryption.
// state - the array that holds the intermediate results during encryption.
unsigned char in[16] = {0};
unsigned char out[16], state[4][4];

// The array that stores the round keys.
unsigned char RoundKey[Nb*(Nr+1)*4];

// The Key input to the AES Program
unsigned char Key[key_size/8];


#define getSBoxValue(num)                sbox[num]
#define getSBoxInvert(num)                rsbox[num]


unsigned char getS(unsigned char num){

return sbox[num];

}
// The round constant word array, Roundcon[i], contains the values given by
// x to th e power (i-1) being powers of x (x is denoted as {02}) in the field GF(28)
// Note that i starts at 1, not 0).


// This function produces Nb(Nr+1) round keys. The round keys are used in each round to encrypt the states.
void KeyExpansion()
{
    unsigned char i,j;
    unsigned char temp[4],k;

    // The first round key is the key itself.
    for(i=0; i<Nk; i++)
    {
        RoundKey[i*4]=Key[i*4];
        RoundKey[i*4+1]=Key[i*4+1];
        RoundKey[i*4+2]=Key[i*4+2];
        RoundKey[i*4+3]=Key[i*4+3];
    }

    // All other round keys are found from the previous round keys.
    while (i < (Nb * (Nr+1)))
    {
        for(j=0;j<4;j++)
        {
            temp[j]=RoundKey[(i-1) * 4 + j];
        }
        if (i % Nk == 0)
        {
            // This function rotates the 4 bytes in a word to the left once.
            // [a0,a1,a2,a3] becomes [a1,a2,a3,a0]

            // Function RotWord()
            {
                k = temp[0];
                temp[0] = temp[1];
                temp[1] = temp[2];
                temp[2] = temp[3];
                temp[3] = k;
            }

            // SubWord() is a function that takes a four-byte input word and
            // applies the S-box to each of the four bytes to produce an output word.

            // Function Subword()
            {
                temp[0]=getSBoxValue(temp[0]);
                temp[1]=getSBoxValue(temp[1]);
                temp[2]=getSBoxValue(temp[2]);
                temp[3]=getSBoxValue(temp[3]);
            }

            temp[0] =  temp[0] ^ Roundcon[i/Nk];
        }
        else if (Nk > 6 && i % Nk == 4)
        {
            // Function Subword()
            {
                temp[0]=getSBoxValue(temp[0]);
                temp[1]=getSBoxValue(temp[1]);
                temp[2]=getSBoxValue(temp[2]);
                temp[3]=getSBoxValue(temp[3]);
            }
        }
        RoundKey[i*4+0] = RoundKey[(i-Nk)*4+0] ^ temp[0];
        RoundKey[i*4+1] = RoundKey[(i-Nk)*4+1] ^ temp[1];
        RoundKey[i*4+2] = RoundKey[(i-Nk)*4+2] ^ temp[2];
        RoundKey[i*4+3] = RoundKey[(i-Nk)*4+3] ^ temp[3];
        i++;
    }
}

// This function adds the round key to state.
// The round key is added to the state by an XOR function.
void AddRoundKey(unsigned char round)
{
    int i,j;
    for(i=0;i<4;i++)
    {
        for(j=0;j<4;j++)
        {
            state[j][i] ^= RoundKey[round * Nb * 4 + i * Nb + j];
        }
    }
}

// The SubBytes Function Substitutes the values in the
// state matrix with values in an S-box.
void SubBytes()
{
    int i,j;
    for(i=0;i<4;i++)
    {
        for(j=0;j<4;j++)
        {
            state[i][j] = getSBoxValue(state[i][j]);

        }
    }
}

// The ShiftRows() function shifts the rows in the state to the left.
// Each row is shifted with different offset.
// Offset = Row number. So the first row is not shifted.
void ShiftRows()
{
    unsigned char temp;

    // Rotate first row 1 columns to left
    temp=state[1][0];
    state[1][0]=state[1][1];
    state[1][1]=state[1][2];
    state[1][2]=state[1][3];
    state[1][3]=temp;

    // Rotate second row 2 columns to left
    temp=state[2][0];
    state[2][0]=state[2][2];
    state[2][2]=temp;

    temp=state[2][1];
    state[2][1]=state[2][3];
    state[2][3]=temp;

    // Rotate third row 3 columns to left
    temp=state[3][0];
    state[3][0]=state[3][3];
    state[3][3]=state[3][2];
    state[3][2]=state[3][1];
    state[3][1]=temp;
}

// xtime is a macro that finds the product of {02} and the argument to xtime modulo {1b}
#define xtime(x)   ((x<<1) ^ ((x>>7) * 0x1b))

/*
unsigned char xtime(unsigned char x){

return ((x<<1) ^ ( ((x>>7) & 1) * 0x1b) );
}
*/
// MixColumns function mixes the columns of the state matrix
// The method used may look complicated, but it is easy if you know the underlying theory.
// Refer the documents specified above.
void MixColumns()
{
    unsigned char i;
    unsigned char Tmp,Tm,t;
    for(i=0;i<4;i++)
    {
        t=state[0][i];
        Tmp = state[0][i] ^ state[1][i] ^ state[2][i] ^ state[3][i] ;

                Tm = state[0][i] ^ state[1][i] ;
                Tm = xtime(Tm);
                state[0][i] ^= Tm ^ Tmp ;

        Tm = state[1][i] ^ state[2][i] ;
                Tm = xtime(Tm);
                state[1][i] ^= Tm ^ Tmp ;


        Tm = state[2][i] ^ state[3][i] ;
                Tm = xtime(Tm);
                state[2][i] ^= Tm ^ Tmp ;

                Tm = state[3][i] ^ t ;
                Tm = xtime(Tm);
                state[3][i] ^= Tm ^ Tmp ;
    }
}


// Cipher is the main function that encrypts the PlainText.
void Cipher()
{
    unsigned char i,j,round=0;

    //Copy the input PlainText to state array.
    for(i=0;i<4;i++)
    {
        for(j=0;j<4;j++)
        {
            state[j][i] = in[i*4 + j];
        }
    }

/***************************************************************SET THE TRIGGER TO GO HIGH*************************************************/   
   output_high (PIN_B0);  //sets the pin B0 to go high at this point. If your intermediate value for the attack is different change this
    
   // Add the First round key to the state before starting the rounds.
    AddRoundKey(0);

    // There will be Nr rounds.
    // The first Nr-1 rounds are identical.
    // These Nr-1 rounds are executed in the loop below.
    for(round=1;round<Nr;round++)
    {
        SubBytes();
        if(round==1){
            output_low (PIN_B0); //sets the pin B0 to go low at this point. If your intermediate value for the attack is different change this
        }
/**************************************************************TRIGGER IS NOW LOW AGAIN*****************************************************/      
        ShiftRows();
        MixColumns();
        AddRoundKey(round);
    }

    // The last round is given below.
    // The MixColumns function is not here in the last round.
    SubBytes();
    ShiftRows();
    AddRoundKey(Nr);

    // The encryption process is over.
    // Copy the state array to output array.
    for(i=0;i<4;i++)
    {
        for(j=0;j<4;j++)
        {
            out[i*4+j]=state[j][i];
        }
    }
}

// The SubBytes Function Substitutes the values in the
// state matrix with values in an S-box.
void InvSubBytes()
{
    unsigned char i,j;
    for(i=0;i<4;i++)
    {
        for(j=0;j<4;j++)
        {
            state[i][j] = getSBoxInvert(state[i][j]);

        }
    }
}


// Multiplty is a macro used to multiply numbers in the field GF(2^8)
//#define Multiply(x,y) (((y & 1) * x) ^ ((y>>1 & 1) * xtime(x)) ^ ((y>>2 & 1) * xtime(xtime(x))) ^ ((y>>3 & 1) * xtime(xtime(xtime(x)))) ^ ((y>>4 & 1) * xtime(xtime(xtime(xtime(x))))))

unsigned char Multiply(unsigned char x, unsigned char y) {
unsigned char result=0;
/*unsigned char xtime1, xtime2, xtime3,xtime4;


xtime1 = xtime(x);
xtime2 = xtime(xtime1);
xtime3 = xtime(xtime2);
xtime4 = xtime(xtime3);
*/

do{
        //result^=(y&1) * x;
        if (y&1) result ^= x;
        x=xtime(x);
        y>>=1;
}while(y != 0);


return result;
//return (((y & 1) * x) ^ ((y>>1 & 1) * xtime1) ^ ((y>>2 & 1) * xtime2) ^ ((y>>3 & 1) * xtime3) ^ ((y>>4 & 1) * xtime4));

//return (((y & 1) * x) ^ ((y>>1 & 1) * xtime) ^ ((y>>2 & 1) * xtime(xtime(x))) ^ ((y>>3 & 1) * xtime(xtime(xtime(x)))) ^ ((y>>4 & 1) * xtime(xtime(xtime(xtime(x))))));

}

// MixColumns function mixes the columns of the state matrix.
// The method used to multiply may be difficult to understand for beginners.
// Please use the references to gain more information.
void InvMixColumns()
{
    unsigned char i;
    unsigned char a,b,c,d;
    for(i=0;i<4;i++)
    {

        a = state[0][i];
        b = state[1][i];
        c = state[2][i];
        d = state[3][i];


        state[0][i] = Multiply(a, 0x0e) ^ Multiply(b, 0x0b) ^ Multiply(c, 0x0d) ^ Multiply(d, 0x09);
        state[1][i] = Multiply(a, 0x09) ^ Multiply(b, 0x0e) ^ Multiply(c, 0x0b) ^ Multiply(d, 0x0d);
        state[2][i] = Multiply(a, 0x0d) ^ Multiply(b, 0x09) ^ Multiply(c, 0x0e) ^ Multiply(d, 0x0b);
        state[3][i] = Multiply(a, 0x0b) ^ Multiply(b, 0x0d) ^ Multiply(c, 0x09) ^ Multiply(d, 0x0e);
    }
}

// The ShiftRows() function shifts the rows in the state to the left.
// Each row is shifted with different offset.
// Offset = Row number. So the first row is not shifted.
void InvShiftRows()
{
    unsigned char temp;

    // Rotate first row 1 columns to right
        temp=state[1][3];
    state[1][3]=state[1][2];
    state[1][2]=state[1][1];
    state[1][1]=state[1][0];
    state[1][0]=temp;

    // Rotate second row 2 columns to right
        temp=state[2][0];
    state[2][0]=state[2][2];
    state[2][2]=temp;

    temp=state[2][1];
    state[2][1]=state[2][3];
    state[2][3]=temp;

    // Rotate third row 3 columns to right
    temp=state[3][0];
    state[3][0]=state[3][1];
    state[3][1]=state[3][2];
    state[3][2]=state[3][3];
    state[3][3]=temp;
}


// InvCipher is the main function that decrypts the CipherText.
void InvCipher()
{
    unsigned char i,j,round;

    //Copy the input CipherText to state array.
    for(i=0;i<4;i++)
    {
        for(j=0;j<4;j++)
        {
            state[j][i] = in[i*4 + j];
        }
    }

    // Add the First round key to the state before starting the rounds.
       AddRoundKey(Nr);



            // There will be Nr rounds.
    // The first Nr-1 rounds are identical.
    // These Nr-1 rounds are executed in the loop below.
    for(round=Nr-1;round>0;round--)
    {
        InvShiftRows();
        InvSubBytes();
        AddRoundKey(round);
        InvMixColumns();
    }

        // The last round is given below.
    // The MixColumns function is not here in the last round.
    InvShiftRows();
    InvSubBytes();
    AddRoundKey(0);

    // The decryption process is over.
    // Copy the state array to output array.
    for(i=0;i<4;i++)
    {
        for(j=0;j<4;j++)
        {
            out[i*4 +j]=state[j][i];
        }
    }
}

/*************************************************************END OF AES*****************************************************************/

//return the value of a ascii character in hexa decimal
int convertdigit(char digit){
   
   unsigned char value=-1;
   switch (digit){
   
   case '0':
      value=0;
      break;
   case '1':
      value=1;
      break;
   case '2':
      value=2;
      break;
   case '3':
      value=3;
      break;
   case '4':
      value=4;
      break;
   case '5':
      value=5;
      break;
   case '6':
      value=6;
      break;
   case '7':
      value=7;
      break;      
   case '8':
      value=8;
      break;
   case '9':
      value=9;
      break;
   case 'A':
      value=10;
      break;
   case 'B':
      value=11;
      break;   
   case 'C':
      value=12;
      break;
   case 'D':
      value=13;
      break;
   case 'E':
      value=14;
      break;
   case 'F':
      value=15;
      break;   
   }

   return value;
}


/******************************************************************Get the key****************************************************/

void getkey(){

   int i;
   
   //get the key from eeprom
   for (i=0;i<16;i++){
      Key[i]=read_eeprom(i);
   }   
      
   //print the key to verify
   /*printf("\r\nKey is : ");
   for (i=0;i<16;i++){
       printf("%2X", Key[i] );
   }
   printf("\r\n");*/
      
}



//interrupt for key change (INT1 pin)
#INT_EXT1
void ext1_isr()
{
   delay_ms(500); //button hold time
   
   if(input_state(pin_B1)==1){
   
      printf("\r\nKey change mode... \r\n");
      delay_ms(1000); //button hold time
      
      char buffer[33];  //space to read the ASCII characters coming through the serial in
      char hex[2]; //space for keeping hexadecimal ASCII representation of an 8 bit number   
      unsigned char keybyte;  //space for the keybyte
      int i;     
      
      if(input_state(pin_B1)==1){
         printf("\r\nPlease Enter the new key through keypad... \r\n");
         
         output_high(PIN_A4);
         
         kbd_init(); 
         char k=0;
         
         //read the key as 32 characters
         for (i=0;i<32;i++){
            k=0;
            while(k==0){
               k=kbd_getc();
            }
            buffer[i]=k;
         }
         buffer[i]=0;
         
         output_low(PIN_A4);
         delay_ms(100);
         output_high(PIN_A4);
         delay_ms(100);
         output_low(PIN_A4);
         
         
      }
      else{
          printf("\r\nPlease Enter the new key through keyboard: \r\n");
         
         //read the key as 32 characters
         for (i=0;i<32;i++){
            buffer[i]=getc();
         }
         buffer[i]=0;     
      }
      
      //convert the input string to unsigned chars and put to eep rom
      for(i=0;i<16;i++){
         hex[0]=buffer[i*2];
         hex[1]=buffer[i*2+1];
         keybyte=convertdigit(hex[1])+16*convertdigit(hex[0]);
         write_eeprom(i,keybyte); 
      }
   
      //prints the inserted key
      printf("\r\nEntered key is : \r\n");
      for (i=0;i<16;i++){
         printf("%2X", read_eeprom(i));
      }
   

   }
   //trigger mode
   else{
      printf("\r\nIn trigger mode. Press any key after done.\r\n");
      
       
      //We need to repeatedly do the encryption on the plain text sample until the host computer aquires the power trace via the oscilloscope
      //hence repeatedly do the encryption until host sends a signal to stop so
      while(1){
            
            //if the host computer has sent a signal, get it and behave appropriately
               if(kbhit()){
                  char temp=getc();              
                  break;
               }
            
            //if the host computer has sent no signal, repeatedly do the encryption
               else{
                
                  // The KeyExpansion routine must be called before encryption.
                  KeyExpansion();
                  
                  // encrypts the PlainText with the Key using AES algorithm.
                  Cipher();
                  
               //just keep a delay
                  delay_ms(5);
               }
     }   
 
       
   }
   
   printf("\r\nRestarting...\r\n");
   delay_ms(100);
   reset_cpu();

}


/***********************************************************************MAIN FUNCTION******************************************************/
void main()
{

   //printf("\r\nStarting...");

   //setting key change interrupts
   ext_int_edge(1,L_TO_H);       //setect low to high
   enable_interrupts(INT_EXT1);
   enable_interrupts(GLOBAL);   

   //arrays and variables 
   extern unsigned char in[16];   //space for the plain text
   extern unsigned char out[16];  //space for the cipher text
   extern unsigned char Key[16];  //space for the key
   char buffer[33];  //space to read the ASCII characters coming through the serial in
   char hex[2]; //space for keeping hexadecimal ASCII representation of an 8 bit number   
   int i;
   char temp=0;
 
   //get the key from eeprom
   getkey(); 
     
   //infinitely take plain text, encrypt and send cipher text back    
   while(1){
 
         //get the input character string to buffer. Since a plain text block is 128 bits it is 32 characters
         for (i=0;i<32;i++){
            buffer[i]=getc();
         
         //some error correction mechanism. If the host sends a 'y' some issue has occurred, clean all the things in the buffer
            if(buffer[i]=='y'){
               while(kbhit()){
                    temp=getc();
               }
            }
         
         }
         buffer[i]=0; //terminating character
         
         //convert the input string to a byte array
         for(i=0;i<16;i++){
            hex[0]=buffer[i*2];
            hex[1]=buffer[i*2+1];
            in[i]=convertdigit(hex[1])+16*convertdigit(hex[0]);
         }

         //prints the plain text via the serial port. The computer can check if communication happen properly
         for (i=0;i<16;i++){
               printf("%2X", in[i] );
         }
         
       //We need to repeatedly do the encryption on the plain text sample until the host computer aquires the power trace via the oscilloscope
       //hence repeatedly do the encryption until host sends a signal to stop so
         while(1){
         
         //if the host computer has sent a signal, get it and behave appropriately
            if(kbhit()){
            temp=getc();
               
            //if the host sends 'z' thats the stopping signal and hence stop encryption and get ready to goto next round
            if(temp=='z'){
                  break;
               }
            
            //if something other than 'z' is received clean everything in the buffers and get ready for the next round
               else{
                  while(kbhit()){
                     temp=getc();
                  }
                  break;
               }
            }
         
         //if the host computer has sent no signal, repeatedly do the encryption
            else{
             
               // The KeyExpansion routine must be called before encryption.
               KeyExpansion();
               
               // encrypts the PlainText with the Key using AES algorithm.
               Cipher();
               
            //just keep a delay
               delay_ms(5);
            }
         }   
 
         //prints the cipher text to verify by the host whether cryptosystem is encrypting properly
         for (i=0;i<16;i++){
               printf("%2X", out[i] );
         }
       
       //just keep a delay
         delay_ms(5);
 
   }
}
