/*
 * TurmeriScan AI — Physical Sorting & Alert Station (v2.1 Enhanced)
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
const int ANGLE_PURE   = 40;   // Swings Left (Pure / Approved Bin)
const int ANGLE_REJECT = 140;  // Swings Right (Adulterated / Reject Bin)

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
  lcd.print("Hardware Init OK");

  // Flash Green then Red LED for 1 full second each so wiring is obvious
  digitalWrite(PIN_GREEN_LED, HIGH);
  delay(800);
  digitalWrite(PIN_GREEN_LED, LOW);

  digitalWrite(PIN_RED_LED, HIGH);
  delay(800);
  digitalWrite(PIN_RED_LED, LOW);

  // Clear visual sweep of the servo arm
  sorterServo.write(ANGLE_PURE);   // 40 degrees Left
  delay(700);
  sorterServo.write(ANGLE_REJECT); // 140 degrees Right
  delay(700);
  sorterServo.write(ANGLE_CENTER); // 90 degrees Center
  delay(500);

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
  sorterServo.write(ANGLE_CENTER);
}

void handlePure(String confidence) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("VERDICT: PURE");
  lcd.setCursor(0, 1);
  lcd.print("Pass: " + confidence + "%");

  digitalWrite(PIN_GREEN_LED, HIGH);
  digitalWrite(PIN_RED_LED, LOW);

  // Swing sorting arm to the LEFT (Pure bin)
  sorterServo.write(ANGLE_PURE);
  
  // Keep verdict and arm active for 6 seconds so user/judges see it clearly!
  delay(6000);

  // Return to center standby
  sorterServo.write(ANGLE_CENTER);
  delay(500);
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

  // Swing sorting arm to the RIGHT (Reject bin)
  sorterServo.write(ANGLE_REJECT);

  // Warning flashing on Red LED for 6 seconds
  for (int i = 0; i < 12; i++) {
    digitalWrite(PIN_RED_LED, HIGH);
    delay(250);
    digitalWrite(PIN_RED_LED, LOW);
    delay(250);
  }

  sorterServo.write(ANGLE_CENTER);
  delay(500);
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
    else if (msg == "TEST_GREEN") {
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("TEST: GREEN LED");
      lcd.setCursor(0, 1);
      lcd.print("Pin 5 ON (5 sec)");
      digitalWrite(PIN_GREEN_LED, HIGH);
      delay(5000);
      digitalWrite(PIN_GREEN_LED, LOW);
      showIdleScreen();
    }
    else if (msg == "TEST_RED") {
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("TEST: RED LED");
      lcd.setCursor(0, 1);
      lcd.print("Pin 6 ON (5 sec)");
      digitalWrite(PIN_RED_LED, HIGH);
      delay(5000);
      digitalWrite(PIN_RED_LED, LOW);
      showIdleScreen();
    }
    else if (msg == "TEST_SERVO") {
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("TEST: SERVO ARM");
      lcd.setCursor(0, 1);
      lcd.print("Swinging...");
      sorterServo.write(ANGLE_PURE);
      delay(1500);
      sorterServo.write(ANGLE_REJECT);
      delay(1500);
      sorterServo.write(ANGLE_CENTER);
      delay(1000);
      showIdleScreen();
    }
  }
}
