#include <24FJ32GA002.h>

//configure a 20MHz crystal to operate at 48MHz
//#fuses HS,NOWDT,NOPROTECT,NOLVP,NODEBUG,USBDIV,CPUDIV1,NOVREGEN,NOBROWNOUT
  #fuses PR_PLL,XT,NOWDT,NOPROTECT,NODEBUG,NOJTAG 

#pin_select U1RX=PIN_B9 
#pin_select U1TX=PIN_B8 
#use delay(clock=32000000)
#use rs232(UART1,baud=9600,parity=N,bits=8)
//#use rs232(baud=9600,parity=N,xmit=PIN_B8,rcv=PIN_B9,bits=8)

// Includes all USB code and interrupts, as well as the CDC API
#include <stdlib.h>

unsigned char pt0[8];
unsigned char pt1[8];
unsigned char ct0[8];
unsigned char ct1[8];
unsigned char K0[8];
unsigned char K1[8];
 
void print(unsigned char pr[8]){
   int i;
   for(i=0;i<8;i++){
      printf("%.2X ",pr[i]);
   }
   printf("\r\n");

}

void copy(unsigned char ans[],unsigned char x[]){
   int i;
   for(i=0;i<8;i++){
      ans[i]=x[i];
   }

}
 
void _shiftR(unsigned char ans[],unsigned char x[],unsigned char r){
   unsigned char i;
   unsigned char shiftbytes;
   unsigned char shiftbits;
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
            ans[i]=(unsigned char)(((unsigned int32)temp[i])>>shiftbits);
         }
         else{
            ans[i]=(unsigned char)(((unsigned int32)temp[i])>>shiftbits)|(((unsigned int32)temp[i-1])<<(8-shiftbits));
         }
      }
   }
   
} 

void _shiftL(unsigned char ans[],unsigned char x[],unsigned char r){
   unsigned char i;
   unsigned char shiftbytes;
   unsigned char shiftbits;
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
            ans[7-i]=(unsigned char)(((unsigned int32)temp[7-i])<<shiftbits);
         }
         else{
            ans[7-i]=(unsigned char)(((unsigned int32)temp[7-i])<<shiftbits)|(((unsigned int32)temp[8-i])>>(8-shiftbits));
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
      unsigned int16 add;
      add=(unsigned int16)x[7-i]+(unsigned int16)y[7-i]+(unsigned int16)q;
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
     
     // print(ct0);print(ct1);print(A);print(B);printf("\r\n");
     
      ROR(ans,ct1,8);   
      copy(ct1,ans);      
      _add(ans,ct1,ct0); 
      copy(ct1,ans);
     // print(ct0);print(ct1);print(A);print(B);printf("\r\n");        
     
      _xor(ans,ct1,A);
      delay_ms(1);
      if(i==0){
         output_high (PIN_B0);
      }
      if(i==1){
         output_high (PIN_B1);
      }   	  
	  
     // print(ct0);print(ct1);print(A);print(B);printf("\r\n");        
      copy(ct1,ans);
     // print(ct0);print(ct1);print(A);print(B);printf("\r\n");  
    
      ROL(ans,ct0,3);
      copy(ct0,ans);
       
      _xor(ans,ct1,ct0);      
      copy(ct0,ans);
     //  print(ct0);print(ct1);print(A);print(B);printf("\r\n");  
     
      ROR(ans,B,8);
      copy(B,ans);
     // print(ct0);print(ct1);print(A);print(B);printf("\r\n");  
      
      _add(ans,B,A);
      copy(B,ans);
     // print(ct0);print(ct1);print(A);print(B);printf("\r\n");        
   
     if(i==0){
         output_low (PIN_B0);
      }
      if(i==1){
         output_low (PIN_B1);
      }
      delay_ms(1);
   
      for(j=0;j<7;j++){
         index[j]=0;     
      }
      index[7]=(unsigned char)i;
      _xor(ans,B,index);
      copy(B,ans);
     // print(ct0);print(ct1);print(A);print(B);printf("\r\n");  
      
      ROL(ans,A,3);
      copy(A,ans);
      _xor(ans,B,A);
      copy(A,ans);
      //print(ct0);print(ct1);print(A);print(B);printf("\r\n");  
     // printf("\r\n");
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
    /*for(i=0;i<8;i++){
      K0[i]=0;
      K1[i]=0;
    }*/
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

         //prints the key
         for (i=0;i<8;i++){
               printf("%2X", K0[i] );
         }
         for (i=0;i<8;i++){
               printf("%2X", K1[i] );
         }
         
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
