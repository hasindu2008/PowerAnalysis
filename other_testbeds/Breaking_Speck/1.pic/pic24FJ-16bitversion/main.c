
#include <24FJ32GA002.h>

//configure a 8MHz crystal to operate at 8MHz
//#fuses HS,NOWDT,NOPROTECT,NOLVP,NODEBUG,USBDIV,CPUDIV1,NOVREGEN,NOBROWNOUT
  #fuses PR,XT,NOWDT,NOPROTECT,NODEBUG,NOJTAG

#pin_select U1RX=PIN_B9 
#pin_select U1TX=PIN_B8 
#use delay(clock=8000000)
#use rs232(UART1,baud=9600,parity=N,bits=8)
//#use rs232(baud=9600,parity=N,xmit=PIN_B8,rcv=PIN_B9,bits=8)

#include <stdlib.h>

unsigned int64 pt[2];
unsigned int64 ct[2];
unsigned int64 K[2];

unsigned int64 ans=0;

void _shiftR(unsigned int64 x,unsigned int64 r){
   ans=(x>>r);
} 

void _shiftL(unsigned int64 x,unsigned int64 r){
   ans= (x<<r);
} 
 
void _or(unsigned int64 x,unsigned int64 y){
   ans= (x | y);
} 

void _add(unsigned int64 x,unsigned int64 y){
   ans=x+y;
}
 

void _xor(unsigned int64 x,unsigned int64 y){
   ans=x^y;
}


/*void _xor(unsigned int64 x,unsigned int64 y){

#asm MY:
   NOP
   NOP
   NOP
   NOP
   NOP
   NOP
   NOP
   NOP   
   PUSH    0x894
   NOP
   NOP   
   POP     0x836
   NOP
   NOP   
   MOV     0x88C,W0
   NOP
   NOP   
   XOR     0x836
   NOP
   NOP   
   PUSH    0x896
   NOP
   NOP   
   POP     0x838
   NOP
   NOP   
   MOV     0x88E,W0
   NOP
   NOP   
   XOR     0x838
   NOP
   NOP   
   PUSH    0x898
   NOP
   NOP   
   POP     0x83A
   NOP
   NOP   
   MOV     0x890,W0
   NOP
   NOP   
   XOR     0x83A
   NOP
   NOP   
   PUSH    0x89A
   NOP
   NOP   
   POP     0x83C
   NOP
   NOP   
   MOV     0x892,W0
   NOP
   NOP   
   XOR     0x83C
   NOP
   NOP
   NOP
   NOP
   NOP
#endasm   
   
   } */


void ROR(unsigned int64 x, unsigned int64 r){
   _shiftR(x,r);
   unsigned int64 r1=ans;
   _shiftL(x,64-r);
   unsigned int64 r2=ans;
   _or(r1,r2);
}

void ROL(unsigned int64 x, unsigned int64 r){
   _shiftL(x,r);
   unsigned int64 r1=ans;
   _shiftR(x,64-r);
   unsigned int64 r2=ans;
   _or(r1,r2);
}


void R(unsigned int64 x, unsigned int64 y, unsigned int64 k){   
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
   
   unsigned int64 i;
   unsigned int64 B = K[1];
   unsigned int64 A = K[0];
   ct[0] = pt[0]; 
   ct[1] = pt[1];   
   
   for(i = 0; i < 32; i++){

      ROR(ct[1],8);
      ct[1]=ans;
      _add(ct[1],ct[0]);
      ct[1]=ans;
     
      _xor(ct[1],A); 
      
      delay_ms(1);
      if(i==0){
         output_high (PIN_B0);
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
         output_low (PIN_B0);
      }
      /*if(i==1){
         output_low (PIN_B1);
      } */ 
      delay_ms(1);
      
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
 
 void convert(unsigned char block[],unsigned int64 ct[]){
  block[0] = (unsigned char) ((ct[0] >> 56)& (unsigned int64)0xFF);
  block[1] = (unsigned char) ((ct[0] >> 48)& (unsigned int64)0xFF);
  block[2] = (unsigned char) ((ct[0] >> 40)& (unsigned int64)0xFF);
  block[3] = (unsigned char) ((ct[0] >> 32)& (unsigned int64)0xFF);
  block[4] = (unsigned char) ((ct[0] >> 24)& (unsigned int64)0xFF);
  block[5] = (unsigned char) ((ct[0] >> 16)& (unsigned int64)0xFF);
  block[6] = (unsigned char) ((ct[0] >> 8)& (unsigned int64)0xFF);
  block[7] = (unsigned char) ((ct[0])& (unsigned int64)0xFF);  
  
  block[8] = (unsigned char) ((ct[1] >> 56)& (unsigned int64)0xFF);
  block[9] = (unsigned char) ((ct[1] >> 48)& (unsigned int64)0xFF);
  block[10] = (unsigned char)((ct[1] >> 40)& (unsigned int64)0xFF);
  block[11] = (unsigned char)((ct[1] >> 32)& (unsigned int64)0xFF);
  block[12] = (unsigned char)((ct[1] >> 24)& (unsigned int64)0xFF);
  block[13] = (unsigned char)((ct[1] >> 16)& (unsigned int64)0xFF);
  block[14] = (unsigned char)((ct[1] >> 8)& (unsigned int64)0xFF);
  block[15] = (unsigned char)((ct[1]) & (unsigned int64)0xFF); 
}

 void convertback(unsigned int64 ct[],unsigned char block[]){
    ct[0]= (((unsigned int64)block[0])<<56) | (((unsigned int64)block[1])<<48) | (((unsigned int64)block[2])<<40) | (((unsigned int64)block[3])<<32) | (((unsigned int64)block[4])<<24) | (((unsigned int64)block[5])<<16) | (((unsigned int64)block[6])<<8) | (((unsigned int64)block[7])); 
    ct[1]= (((unsigned int64)block[8])<<56) | (((unsigned int64)block[9])<<48) | (((unsigned int64)block[10])<<40) | (((unsigned int64)block[11])<<32) | (((unsigned int64)block[12])<<24) | (((unsigned int64)block[13])<<16) | (((unsigned int64)block[14])<<8) | (((unsigned int64)block[15])) ;
}

void convertbyte(unsigned char block[16],unsigned int64 ct){
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
   char buffer[33];
   char hex[2];
   int i;
   char temp=0;
   unsigned char block[16];   
     
   setkey(); 
     
   while(1){

         //get the input string
         for (i=0;i<32;i++){
            buffer[i]=getc();
            if(buffer[i]=='y'){
               while(kbhit()){
                    temp=getc();
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
               printf("%2X", block[i] );
         }
       convertback(pt,block);
         
         while(1){
         
             if(kbhit()){
               temp=getc();
               if(temp=='z'){
                  break;
               }
               else{
                  while(kbhit()){
                     temp=getc();
                  }
                  break;
               }
             }
             else{
 
               //output_high (PIN_B0);
               encrypt();
               //output_low (PIN_B0);
               delay_ms(5);
             }
         }   
 
         //prints the cipher text
      convert(block,ct);       
         for (i=0;i<16;i++){
               printf("%2X", block[i] );
         }
         delay_ms(5);
  
      
   }
}
