#include <18F2550.h>

//configure a 20MHz crystal to operate at 48MHz
#fuses HSPLL,NOWDT,NOPROTECT,NOLVP,NODEBUG,PLL2,CPUDIV1,NOVREGEN,NOBROWNOUT,NOMCLR
//#fuses   USBDIV, PLL1, CPUDIV1, PROTECT, NOCPD, noBROWNOUT,HSPLL,NOWDT,nolvp, VREGEN
#use delay(clock=48000000)
#use rs232(UART1,baud=9600,parity=N,bits=8)
 
#include <stdlib.h>

int addr=0;
unsigned char pt0[8];
unsigned char pt1[8];
unsigned char ct0[8];
unsigned char ct1[8];
unsigned char K0[8];
unsigned char K1[8];
 
void copy(unsigned char ans[],unsigned char x[]){
   int i;
   for(i=0;i<8;i++){
      ans[i]=x[i];
   }

}
 
void _shiftR(unsigned char ans[],unsigned char x[],unsigned char r){
   int i;
   int shiftbytes;
   int shiftbits;
   unsigned char temp[8];
   
   shiftbytes=r/8;
   shiftbits=r%8;
   
   for(i=0;i<shiftbytes;i++){
      ans[i]=0;
   }
   for(i=shiftbytes;i<8;i++){
      ans[i]=x[i-shiftbytes];
   }
   
   for(i=0;i<8;i++){
      temp[i]=ans[i];
   }
   
   if(shiftbits>0){
      for(i=shiftbytes;i<8;i++){
         if(i==shiftbytes){
            ans[i]=(temp[i]>>shiftbits);
         }
         else{
            ans[i]=(temp[i]>>shiftbits)|(temp[i-1]<<(8-shiftbits));
         }
      }
   }
   
} 

void _shiftL(unsigned char ans[],unsigned char x[],unsigned char r){
   int i;
   int shiftbytes;
   int shiftbits;
   unsigned char temp[8];
 
   shiftbytes=r/8;
  shiftbits=r%8;
   
   for(i=0;i<shiftbytes;i++){
      ans[7-i]=0;
   }
   for(i=shiftbytes;i<8;i++){
      ans[7-i]=x[7-i+shiftbytes];
   }
   
   for(i=0;i<8;i++){
      temp[i]=ans[i];
   }   
   
   if(shiftbits>0){
      for(i=shiftbytes;i<8;i++){
         if(i==shiftbytes){
            ans[7-i]=(temp[7-i]<<shiftbits);
         }
         else{
            ans[7-i]=(temp[7-i]<<shiftbits)|(temp[8-i]>>(8-shiftbits));
         }
      }
   }   
} 
 
void _or(unsigned char ans[],unsigned char x[],unsigned char y[]){
   int i=0;
   for(i=0;i<8;i++){
      ans[i]=(x[i])|(y[i]);
   }
} 

void _add(unsigned char ans[],unsigned char x[],unsigned char y[]){
   int i;
   unsigned char q=0;
  
 for(i=0;i<8;i++){
      unsigned int32 add;
      add=(unsigned int32)x[7-i]+(unsigned int32)y[7-i]+(unsigned int32)q;
      ans[7-i]=(unsigned char)(add%256);
      q=(unsigned char)(add/256);
  }
}
 
void _xor(unsigned char ans[],unsigned char x[],unsigned char y[]){
   int i=0;
   for(i=0;i<8;i++){
      ans[i]=(x[i])^(y[i]);
   }
} 
void ROR(unsigned char ans[],unsigned char x[], unsigned char  r){
   unsigned char r1[8];
   unsigned char r2[8];
   _shiftR(r1,x,r);
   _shiftL(r2,x,64-r);
   _or(ans,r1,r2);
}

void ROL(unsigned char ans[],unsigned char x[], unsigned char  r){
   unsigned char r1[8];
   unsigned char r2[8];
   _shiftL(r1,x,r);
   _shiftR(r2,x,64-r);
   _or(ans,r1,r2);
}

 
void encrypt()
{
   unsigned char i;      
   unsigned char ans[8];
   unsigned char B[8];
   unsigned char A[8];
   
   for(i=0;i<8;i++){
   B[i] = K1[i];
   A[i] = K0[i];
   ct0[i] = pt0[i]; 
   ct1[i] = pt1[i];
   }

 for(i = 0; i < 32; i++){
     
      int j;
      unsigned char index[8];
      
      ROR(ans,ct1,8);   
      copy(ct1,ans);      
      _add(ans,ct1,ct0);

      copy(ct1,ans);
      _xor(ans,ct1,A);  
      copy(ct1,ans);
   
      ROL(ans,ct0,3);
      copy(ct0,ans);

      delay_ms(1);
      if(i==0){
         output_high (PIN_B0);
      }
      if(i==1){
         output_high (PIN_B1);
      } 
      _xor(ans,ct1,ct0);
      if(i==0){
         output_low (PIN_B0);
      }
      if(i==1){
         output_low (PIN_B1);
      }
      delay_ms(1);          
      
      copy(ct0,ans);
      
      ROR(ans,B,8);
      copy(B,ans);
      
      _add(ans,B,A);
      copy(B,ans);
   
      for(j=0;j<7;j++){
         index[j]=0;
      }
      index[7]=(unsigned char)i;
      _xor(ans,B,index);
      copy(B,ans);
         
      ROL(ans,A,3);
      copy(A,ans);
      _xor(ans,B,A);
      copy(A,ans);
    
     
     
  }
}

void blockToEEP(unsigned char block1[],unsigned char block2[]){
   int i;
   for(i=0;i<8;i++){
      write_eeprom(addr,block1[i]);
      addr++;
   }
   for(i=8;i<16;i++){
      write_eeprom(addr,block2[i-8]);
      addr++;   
   }   
   
}

void blockToEEP2(){
   int i;
   for(i=0;i<8;i++){
      write_eeprom(addr,pt0[i]);
      addr++;
   }
   for(i=8;i<16;i++){
      write_eeprom(addr,pt1[i-8]);
      addr++;   
   }   
   
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
    K0[0]=0x8;
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
    K1[7]=0x3;

    /*K0[0]=0xFF;
    K0[1]=0xEE;
    K0[2]=0xDD;
    K0[3]=0xCC;
    K0[4]=0xBB;
    K0[5]=0xAA;
    K0[6]=0x99;
    K0[7]=0x88;
   
    K1[0]=0xEE;
    K1[1]=0xDD;
    K1[2]=0xCC;
    K1[3]=0xBB;
    K1[4]=0xAA;
    K1[5]=0x99;
    K1[6]=0x88;
    K1[7]=0x77; */
    
   
}

void main()
{
   char buffer[33];
   char hex[2];
   int i;
   char temp=0;
     
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

         /*prints the key
         for (i=0;i<8;i++){
               printf("%2X", K0[i] );
         }
         for (i=0;i<8;i++){
               printf("%2X", K1[i] );
         }*/
         
         //convert the input string
         for(i=0;i<16;i++){
            hex[0]=buffer[i*2];
            hex[1]=buffer[i*2+1];
            if(i<8){
               pt0[i]=convertdigit(hex[1])+16*convertdigit(hex[0]);
            }
            else{
               pt1[i-8]=convertdigit(hex[1])+16*convertdigit(hex[0]);               
            }
         }

         //prints the plain text
         for (i=0;i<8;i++){
               printf("%2X", pt0[i] );
         }
         for (i=0;i<8;i++){
               printf("%2X", pt1[i] );
         }
         
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
         for (i=0;i<8;i++){
               printf("%2X", ct0[i] );
         }
         for (i=0;i<8;i++){
               printf("%2X", ct1[i] );
         }
         delay_ms(5);
 
      
   }
}
