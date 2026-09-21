/*
 * TurmeriScan AI — Physical Sorting & Alert Station (Silent / No Buzzer)
 * Hardware Actuator for Arduino Uno
 * 
 * Pinout:
 * - Green LED: Pin 5
 * - Red LED:   Pin 6
 * - Servo:     Pin 9 (PWM)
 * - I2C LCD:   SDA -> A4, SCL -> A5, VCC -> 5V, GND -> GND
 */

#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <Servo.h>

// 16x2 LCD at standard I2C address 0x27 (or 0x3F)
LiquidCrystal_I2C lcd(0x27, 16, 2);
Servo sorterServo;

// Pin Definitions
const int PIN_GREEN_LED = 5;
const int PIN_RED_LED   = 6;
const int PIN_SERVO     = 9;

// Servo Angles for sorting
const int ANGLE_CENTER = 90;
const int ANGLE_PURE   = 45;   // Swings Left (Pure Tray)
const int ANGLE_REJECT = 135;  // Swings Right (Rejection Tray)

void setup() {
  Serial.begin(9600);

  pinMode(PIN_GREEN_LED, OUTPUT);
  pinMode(PIN_RED_LED, OUTPUT);

  sorterServo.attach(PIN_SERVO);
  sorterServo.write(ANGLE_CENTER);

  // Initialize LCD Screen
  lcd.init();
  lcd.backlight();
  
  // Run startup self-test
  runSelfTest();
}

void runSelfTest() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("TurmeriScan AI");
  lcd.setCursor(0, 1);
  lcd.print("Hardware Ready..");

  // Flash Green then Red LED
  digitalWrite(PIN_GREEN_LED, HIGH);
  delay(300);
  digitalWrite(PIN_GREEN_LED, LOW);
  digitalWrite(PIN_RED_LED, HIGH);
  delay(300);
  digitalWrite(PIN_RED_LED, LOW);

  // Test servo movement
  sorterServo.write(ANGLE_PURE);
  delay(400);
  sorterServo.write(ANGLE_REJECT);
  delay(400);
  sorterServo.write(ANGLE_CENTER);

  delay(800);
  showIdleScreen();
}

void showIdleScreen() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("TURMERISCAN AI");
  lcd.setCursor(0, 1);
  lcd.print("Awaiting Sample");
  digitalWrite(PIN_GREEN_LED, LOW);
  digitalWrite(PIN_RED_LED, LOW);
}

void handlePure(String confidence) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("VERDICT: PURE");
  lcd.setCursor(0, 1);
  lcd.print("Pass: " + confidence + "%");

  digitalWrite(PIN_GREEN_LED, HIGH);
  digitalWrite(PIN_RED_LED, LOW);

  // Move sorting arm to Pure tray
  sorterServo.write(ANGLE_PURE);
  delay(2500);

  // Return to center
  sorterServo.write(ANGLE_CENTER);
  delay(800);
  showIdleScreen();
}

void handleAdulterated(String confidence, String adulterant) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("! ADULTERATED !");
  lcd.setCursor(0, 1);
  if (adulterant.length() > 0) {
    lcd.print(adulterant.substring(0, 16));
  } else {
    lcd.print("Risk: " + confidence + "%");
  }

  digitalWrite(PIN_GREEN_LED, LOW);

  // Move sorting arm to Reject tray
  sorterServo.write(ANGLE_REJECT);

  // Warning flashing on Red LED
  for (int i = 0; i < 5; i++) {
    digitalWrite(PIN_RED_LED, HIGH);
    delay(200);
    digitalWrite(PIN_RED_LED, LOW);
    delay(200);
  }

  delay(1500);
  sorterServo.write(ANGLE_CENTER);
  delay(800);
  showIdleScreen();
}

void loop() {
  if (Serial.available() > 0) {
    String msg = Serial.readStringUntil('\n');
    msg.trim();

    if (msg.startsWith("PURE")) {
      int comma = msg.indexOf(',');
      String conf = (comma != -1) ? msg.substring(comma + 1) : "100";
      handlePure(conf);
    } 
    else if (msg.startsWith("ADULTERATED")) {
      int comma1 = msg.indexOf(',');
      int comma2 = msg.indexOf(',', comma1 + 1);
      String conf = "100";
      String adult = "Synthetic Dye";
      if (comma1 != -1) {
        if (comma2 != -1) {
          conf = msg.substring(comma1 + 1, comma2);
          adult = msg.substring(comma2 + 1);
        } else {
          conf = msg.substring(comma1 + 1);
        }
      }
      handleAdulterated(conf, adult);
    }
  }
}
