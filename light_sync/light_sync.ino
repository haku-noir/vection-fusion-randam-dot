#include "BluetoothSerial.h"
BluetoothSerial SerialBT;
int INPIN=18;//タイミングを取る指標が出てきたらフォトトラから電圧出力される
int OUTPIN=19;
int initflag=1; //1ならば初期状態判定を行う
void getStartTiming(){
  if(initflag==1){
    Serial.println(1);
    digitalWrite(OUTPIN,HIGH);
  }
}
void init(){
  initflag=1;
  digitalWrite(OUTPIN,LOW);
}
void setup() {
  Serial.begin(9600);
  SerialBT.begin("ESP32_Unity");
  // put your setup code here, to run once:
  pinMode(OUTPIN,OUTPUT);
  pinMode(INPIN,INPUT);
  // ピン番号から割り込み番号への変換には専用の関数を使用
  attachInterrupt(digitalPinToInterrupt(INPIN), getStartTiming, RISING);
  init();
}
void loop() {
  // put your main code here, to run repeatedly:
  //SerialBTを受け取って，init状態であるかを設定すること
 // delay(1000);
//  init();
  if(SerialBT.available()){
    char receivedChar = SerialBT.read();
    Serial.print("Received: ");
    Serial.println(receivedChar);
    if(receivedChar ='r'){
      //reset
      init();
    }
  }
}