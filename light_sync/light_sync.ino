// 共有変数は volatile で宣言する
volatile bool readyFlag = false;
volatile bool interruptFired = false; // 割り込みが発生したことを知らせるフラグ

int INPIN = 18;  // タイミングを取る指標が出てきたらフォトトラから電圧出力される
int OUTPIN = 19;

// ISRはフラグを立てるだけにする
void IRAM_ATTR getStartTiming() { // ESP32ではISRに IRAM_ATTR を付けると高速化され安定します
  interruptFired = true;
}

void setup() {
  Serial.begin(9600);
  pinMode(OUTPIN, OUTPUT);
  pinMode(INPIN, INPUT);
  
  attachInterrupt(digitalPinToInterrupt(INPIN), getStartTiming, RISING);
  
  readyFlag = false;
  digitalWrite(OUTPIN, LOW);
}

void loop() {
  // loop()内でフラグをチェックし、実際の処理を行う
  if (interruptFired) {
    interruptFired = false; // フラグをすぐに下ろす

    if (readyFlag) {
      digitalWrite(OUTPIN, HIGH);
      readyFlag = false;
    }
  }

  // シリアル入力の処理
  if (Serial.available()) {
    char receivedChar = Serial.read();
    Serial.print("Received: ");
    Serial.println(receivedChar);

    if (receivedChar == 'r') {
      digitalWrite(OUTPIN, LOW);
      readyFlag = true;
    } else if (receivedChar == 's') {
      digitalWrite(OUTPIN, HIGH);
      Serial.println("OUTPIN set HIGH by 's' command");
    } else if (receivedChar == 'x') {
      digitalWrite(OUTPIN, LOW);
      readyFlag = false;
      Serial.println("OUTPIN set LOW by 'x' command");
    } else if (receivedChar == 'd') {
      Serial.print("Debug - INPIN: ");
      Serial.print(digitalRead(INPIN) == HIGH ? "HIGH" : "LOW");
      Serial.print(", OUTPIN: ");
      Serial.print(digitalRead(OUTPIN) == HIGH ? "HIGH" : "LOW");
      Serial.print(", readyFlag: ");
      Serial.print(readyFlag);
      Serial.print("\n");
    }
  }
}
