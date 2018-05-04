/* 
 * File:   main.c
 * Author: USER
 *
 * Created on October 25, 2015, 8:47 PM
 */

#include <stdio.h>
#include <stdlib.h>
#include <p32xxxx.h>
#include <stdlib.h>
#include <plib.h>


//  Configuration Bit settings
//  System Clock = 40 MHz,  Peripherial Bus = 40 MHz
//  Internal Osc w/PLL FNOSC = FRCPLL
//  Input Divider    2x Divider FPLLIDIV
//  Multiplier      20x Multiplier FPLLMUL
//  Output divider   2x Divider FPLLODIV
//  peripherial bus divider FPBDIV = 1
//  WDT disabled
//  Other options are don't care
//to run at 40MHz using fasr RC oscillator
//#pragma config FNOSC = FRCPLL, POSCMOD = HS, FPLLIDIV = DIV_2, FPLLMUL = MUL_20, FPBDIV = DIV_1, FPLLODIV = DIV_2
// frequency we're running at
//#define	SYS_FREQ 40000000

//use the internal fast RC oscillator to work at 8MHz. FRC is 8MHz so prescalers postscalers dont matter
#pragma config FNOSC = FRC, POSCMOD = HS, FPLLIDIV = DIV_2, FPLLMUL = MUL_20, FPBDIV = DIV_1, FPLLODIV = DIV_2
#pragma config FWDTEN = OFF                  //disable the watchdong time
//#pragma config FVBUSONIO = OFF             // USB VBUS ON Selection (Controlled by Port Function)
//#pragma config JTAGEN = OFF                // JTAG Enable (JTAG Disabled)
//#pragma config PWP = OFF                  // Program Flash Write Protect (Disable)
//#pragma config BWP = OFF                  // Boot Flash Write Protect bit (Protection Disabled)
//#pragma config CP = OFF                   // Code Protect (Protection Disabled)

//frequency
#define	SYS_FREQ 8000000

// UART parameters
#define BAUDRATE 9600 // must match PC end
#define PB_DIVISOR (1 << OSCCONbits.PBDIV) // read the peripheral bus divider, FPBDIV
#define PB_FREQ SYS_FREQ/PB_DIVISOR // periperhal bus frequency


/*********************************************************************************************/

unsigned long long pt[2];
unsigned long long ct[2];
unsigned long long K[2];
unsigned long long ans=0;


void generalDelay(){
    int i;
    for(i=0;i<100;i++);
}


void _shiftR(unsigned long long x,unsigned long long r){
   ans=(x>>r);
}

void _shiftL(unsigned long long x,unsigned long long r){
   ans= (x<<r);
}

void _or(unsigned long long x,unsigned long long y){
   ans= (x | y);
}

void _add(unsigned long long x,unsigned long long y){
   ans=x+y;
}


void _xor(unsigned long long x,unsigned long long y){
   ans=x^y;
}


void ROR(unsigned long long x, unsigned long long r){
   _shiftR(x,r);
   unsigned long long r1=ans;
   _shiftL(x,64-r);
   unsigned long long r2=ans;
   _or(r1,r2);
}

void ROL(unsigned long long x, unsigned long long r){
   _shiftL(x,r);
   unsigned long long r1=ans;
   _shiftR(x,64-r);
   unsigned long long r2=ans;
   _or(r1,r2);
}


void R(unsigned long long x, unsigned long long y, unsigned long long k){
   ROR(x,8);
   x=ans;
   _add(x,y);
   x=ans;
   _xor(x,k);
   x=ans;
   ROL(y,3);
   y=ans;
   _xor(x,y);
   y=ans;
}


void encrypt(){

   unsigned long long i;
   unsigned long long B = K[1];
   unsigned long long A = K[0];
   ct[0] = pt[0];
   ct[1] = pt[1];

   TRISBbits.TRISB0=0;

   for(i = 0; i < 32; i++){

      ROR(ct[1],8);
      ct[1]=ans;
      _add(ct[1],ct[0]);
      ct[1]=ans;

      _xor(ct[1],A);

      generalDelay();
      if(i==0){
         LATBbits.LATB0 = 1;
      }
      /*if(i==1){
         output_high (PIN_B1);
      } */

      ct[1]=ans;

      ROL(ct[0],3);
      ct[0]=ans;

      _xor(ct[1],ct[0]);
      ct[0]=ans;

      if(i==0){
          LATBbits.LATB0 = 0;
      }
      /*if(i==1){
         output_low (PIN_B1);
      } */
      generalDelay();

      ROR(B,8);
      B=ans;

      _add(B,A);
      B=ans;

      _xor(B,i);
      B=ans;

      ROL(A,3);
      A=ans;
      _xor(B,A);
      A=ans;

   }
}

 void convert(unsigned char block[],unsigned long long ct[]){
  block[0] = (unsigned char) ((ct[0] >> 56)& (unsigned long long)0xFF);
  block[1] = (unsigned char) ((ct[0] >> 48)& (unsigned long long)0xFF);
  block[2] = (unsigned char) ((ct[0] >> 40)& (unsigned long long)0xFF);
  block[3] = (unsigned char) ((ct[0] >> 32)& (unsigned long long)0xFF);
  block[4] = (unsigned char) ((ct[0] >> 24)& (unsigned long long)0xFF);
  block[5] = (unsigned char) ((ct[0] >> 16)& (unsigned long long)0xFF);
  block[6] = (unsigned char) ((ct[0] >> 8)& (unsigned long long)0xFF);
  block[7] = (unsigned char) ((ct[0])& (unsigned long long)0xFF);

  block[8] = (unsigned char) ((ct[1] >> 56)& (unsigned long long)0xFF);
  block[9] = (unsigned char) ((ct[1] >> 48)& (unsigned long long)0xFF);
  block[10] = (unsigned char)((ct[1] >> 40)& (unsigned long long)0xFF);
  block[11] = (unsigned char)((ct[1] >> 32)& (unsigned long long)0xFF);
  block[12] = (unsigned char)((ct[1] >> 24)& (unsigned long long)0xFF);
  block[13] = (unsigned char)((ct[1] >> 16)& (unsigned long long)0xFF);
  block[14] = (unsigned char)((ct[1] >> 8)& (unsigned long long)0xFF);
  block[15] = (unsigned char)((ct[1]) & (unsigned long long)0xFF);
}

 void convertback(unsigned long long ct[],unsigned char block[]){
    ct[0]= (((unsigned long long)block[0])<<56) | (((unsigned long long)block[1])<<48) | (((unsigned long long)block[2])<<40) | (((unsigned long long)block[3])<<32) | (((unsigned long long)block[4])<<24) | (((unsigned long long)block[5])<<16) | (((unsigned long long)block[6])<<8) | (((unsigned long long)block[7]));
    ct[1]= (((unsigned long long)block[8])<<56) | (((unsigned long long)block[9])<<48) | (((unsigned long long)block[10])<<40) | (((unsigned long long)block[11])<<32) | (((unsigned long long)block[12])<<24) | (((unsigned long long)block[13])<<16) | (((unsigned long long)block[14])<<8) | (((unsigned long long)block[15])) ;
}

void convertbyte(unsigned char block[16],unsigned long long ct){
  block[0] = (unsigned char) (ct >> 56);
  block[1] = (unsigned char) (ct >> 48);
  block[2] = (unsigned char) (ct >> 40);
  block[3] = (unsigned char) (ct >> 32);
  block[4] = (unsigned char) (ct >> 24);
  block[5] = (unsigned char) (ct >> 16);
  block[6] = (unsigned char) (ct >> 8);
  block[7] = (unsigned char) (ct);

}


void blockToSTring(unsigned char block[16]){
   int i;
   for(i=0;i<16;i++){
      //sprintf(&str[2*i],"%.2x",(unsigned char)block[i]);
      printf("%.2X ",block[i]);
   }
   //str[16*2]=0;
   printf("\n");

}


void blockToSTringbyte(unsigned char block[8]){
   int i;
   for(i=0;i<8;i++){
      //sprintf(&str[2*i],"%.2x",(unsigned char)block[i]);
      printf("%.2X ",block[i]);
   }
   //str[16*2]=0;
   printf("\n");

}

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

/*void setkey(){
        int i;
        for(i=0;i<16;i++){
                K0[i]=0;
                K1[i]=0;
        }
}*/

void setkey(){
    int i;
    /*K0[0]=0x8;
    K0[1]=0x7;
    K0[2]=0x6;
    K0[3]=0x5;
    K0[4]=0x4;
    K0[5]=0x3;
    K0[6]=0x2;
    K0[7]=0x1;

    K1[0]=0x4;
    K1[1]=0x3;
    K1[2]=0x2;
    K1[3]=0x5;
    K1[4]=0x1;
    K1[5]=0x1;
    K1[6]=0x2;
    K1[7]=0x3;*/

    K[0]=0x0807060504030201;
    K[1]=0x0403020501010203;

}


void main()
{
   __XC_UART = 2;

   // specify PPS group, signal, logical pin name
   PPSInput (2, U2RX, RPB11); //Assign U2RX to pin RPB11 -- Physical pin 22 on 28 PDIP
   PPSOutput(4, RPB10, U2TX); //Assign U2TX to pin RPB10 -- Physical pin 21 on 28 PDIP

   ANSELA =0; //make sure analog is cleared
   ANSELB =0;

   // init the uart2
   UARTConfigure(UART2, UART_ENABLE_PINS_TX_RX_ONLY);
   UARTSetLineControl(UART2, UART_DATA_SIZE_8_BITS | UART_PARITY_NONE | UART_STOP_BITS_1);
   UARTSetDataRate(UART2, PB_FREQ, BAUDRATE);
   UARTEnable(UART2, UART_ENABLE_FLAGS(UART_PERIPHERAL | UART_RX | UART_TX));

   char buffer[33];
   char hex[2];
   int i;
   char temp=0;
   unsigned char block[16];   
     
   setkey(); 
     
   while(1){

         //get the input string
         for (i=0;i<32;i++){
            // wait for a character 
            while(!UARTReceivedDataIsAvailable(UART2)){}; 
            //receive it
            buffer[i]=UARTGetDataByte(UART2);
            if(buffer[i]=='y'){
               while(UARTReceivedDataIsAvailable(UART2)){
                    temp=UARTGetDataByte(UART2);
               }
            }
         }
         buffer[i]=0;
       
         //convert the input string
         for(i=0;i<16;i++){
            hex[0]=buffer[i*2];
            hex[1]=buffer[i*2+1];
            if(i<16){
               block[i]=convertdigit(hex[1])+16*convertdigit(hex[0]);
            }
         }

         //prints the plain text
         for (i=0;i<16;i++){
               printf("%.2X", block[i] );
         }
        convertback(pt,block);
         
         while(1){
         
             if(UARTReceivedDataIsAvailable(UART2)){
               temp=UARTGetDataByte(UART2);;
               if(temp=='z'){
                  break;
               }
               else{
                  while(UARTReceivedDataIsAvailable(UART2)){
                     temp=UARTGetDataByte(UART2);
                  }
                  break;
               }
             }
             else{
 
               //output_high (PIN_B0);
               encrypt();
               //output_low (PIN_B0);
               //generalDelay();
             }
         }   
 
         //prints the cipher text
		convert(block,ct);       
         for (i=0;i<16;i++){
               printf("%.2X", block[i] );
         }
         generalDelay();
      
   }

}
