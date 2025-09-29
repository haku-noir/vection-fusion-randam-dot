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
  // put your setup code here, to run once:
  pinMode(OUTPIN,OUTPUT);
  pinMode(INPIN,INPUT);
  // ピン番号から割り込み番号への変換には専用の関数を使用
  attachInterrupt(digitalPinToInterrupt(INPIN), getStartTiming, RISING);
  init();
}

void loop() {
  // シリアル入力がある場合
  if (Serial.available()) {
    char receivedChar = Serial.read();
    Serial.print("Received: ");
    Serial.println(receivedChar);

    if (receivedChar == 'r') {
      // reset
      init();
    } else if (receivedChar == 's') {
      // s入力で強制的にHIGHにする
      digitalWrite(OUTPIN, HIGH);
      Serial.println("OUTPIN set HIGH by 's' command");
    }
  }
}
