int x;
#define R1 2
#define G1 3
 
 
#define R2 4
#define G2 5
int v= 100;
void setup() {
  Serial.begin(9600);
pinMode(R1,OUTPUT);
pinMode(G1,OUTPUT);
 
pinMode(R2,OUTPUT);
pinMode(G2,OUTPUT);
 
 
}
 
void loop() {
if(Serial.available()>0){
      String x = Serial.readString();
      Serial.println(x);
  if(x.toInt() == 1){
    Serial.println("okay");
digitalWrite(G1,HIGH);
digitalWrite(R1,LOW);
 
digitalWrite(R2,HIGH);
digitalWrite(G2,LOW);
delay(5000);
}
  else if(x.toInt() == 2){
    Serial.println("okay");
 
digitalWrite(G2,HIGH);
digitalWrite(R1,HIGH);
 
digitalWrite(G1,LOW);
digitalWrite(R2,LOW);
delay(5000);
 
 
}
 
 
   }
else{
 
  digitalWrite(R1,HIGH);
  digitalWrite(G1,LOW);
  digitalWrite(R2,LOW);
  digitalWrite(G2,HIGH);
 
 
  for(int i =0;i<v;i++){
    delay(10);
    if(Serial.available()>0){
      break;
    }
    }
  digitalWrite(R1,LOW);
  digitalWrite(G1,HIGH);
  digitalWrite(R2,HIGH);
  digitalWrite(G2,LOW);
  
for(int i =0;i<v;i++){
      delay(10);
 
    if(Serial.available()>0){
      break;
    }
    }   
 
}
}