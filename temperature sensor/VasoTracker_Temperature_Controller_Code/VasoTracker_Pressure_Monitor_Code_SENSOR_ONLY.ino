

/* VasoTracker Temperature Controller Code */

/*
  This code was adapted from Sébastien Parent-Charette's Wheatstone Bridge code for RobotShop.

  Find the original here: https://github.com/RobotShop/Wheatstone-Bridge-Amplifier-Shield/archive/master.zip
  
  The original code was, and indeed this adaptation is, meant to be used with the RB-Onl-38 and RB-Ite-161 from RobotShop.
  These products are available here:
          http://www.robotshop.com/en/strain-gauge-load-cell-amplifier-shield-2ch.html
          http://www.robotshop.com/en/16x2-lcd-shield-kit-arduino.html
          http://www.robotshop.com/en/arduino-r3-stackable-headers.html


  The code has been modified to perform the following functions:

  * - Read and display pressure values from 2 pressure transducers.
  * - Pass pressure values to some Python code via the serial port.
  

*/
#include "strain_gauge_shield_and_lcd_support_functions.h" // Include support function header ---> This file should be in the working folder!
//Define Variables we'll be connecting to

// Initialize the library with the numbers of the interface pins
#include <LiquidCrystal.h> // Include the LCD library
LiquidCrystal lcd(8, 9, 4, 5, 6, 7);

// which analog pin to connect
#define THERMISTORPIN A0         
// resistance at 25 degrees C
#define THERMISTORNOMINAL 10000      
// temp. for nominal resistance (almost always 25 C)
#define TEMPERATURENOMINAL 25   
// how many samples to take and average, more takes longer
// but is more 'smooth'
#define NUMSAMPLES 5
// The beta coefficient of the thermistor (usually 3000-4000)
#define BCOEFFICIENT 3950
// the value of the 'other' resistor
#define SERIESRESISTOR 10000    
#define INITIAL_SET_POINT 60
uint16_t samples[NUMSAMPLES];


// Initial calibration values
const int CST_TEMP_IN_MIN = 350;       // Raw value calibration lower point
const int CST_TEMP_IN_MAX = 650;       // Raw value calibration upper point
const int CST_PRESSURE_OUT_MIN = 0;        // Pressure calibration lower point
const int CST_PRESSURE_OUT_MAX = 140;     // Pressure calibration upper point ---> 140 is about the maximum that can be read out with the default gain on the RB-Onl-38 and 26PCDFG5G pressure transducers. Can change the gain with surface mount resistors.

const int CST_CAL_TEMP_MIN = 30;
const int CST_CAL_TEMP_MAX = 50;
const int CST_CAL_TEMP_STEP = 0.5;
const int CST_CAL_TEMP_STEP_LARGE = 1;

// initial set point
#define INITIAL_SET_POINT 30.0

// < Setup function >
void setup()
{
  // Initialize LCD screen
  lcd.begin(16, 2);
  
  // Intro screen
  displayScreen("Myograph Temp", "Controller");
  delay(3000);




  // On the first loop, get the desired temperature!

    displayScreen("Set (*C):", "Act (*C):");
    Serial.begin(9600);
    analogReference(EXTERNAL);


}




// Timing management
long display_time_step = 1000;
long display_time = 0;

//Crap counter
int count = 0;
void loop(void) {


  uint8_t i;
  float average;
 
  // take N samples in a row, with a slight delay
  for (i=0; i< NUMSAMPLES; i++) {
   samples[i] = analogRead(THERMISTORPIN);
   delay(10);
  }
 
  // average all the samples out
  average = 0;
  for (i=0; i< NUMSAMPLES; i++) {
     average += samples[i];
  }
  average /= NUMSAMPLES;
 
  //Serial.print("Average analog reading "); 
  //Serial.println(average);
 
  // convert the value to resistance
  average = 1023 / average - 1;
  average = SERIESRESISTOR / average;
  //Serial.print("Thermistor resistance "); 
  //Serial.println(average);
 
  float steinhart;
  steinhart = average / THERMISTORNOMINAL;     // (R/Ro)
  steinhart = log(steinhart);                  // ln(R/Ro)
  steinhart /= BCOEFFICIENT;                   // 1/B * ln(R/Ro)
  steinhart += 1.0 / (TEMPERATURENOMINAL + 273.15); // + (1/To)
  steinhart = 1.0 / steinhart;                 // Invert
  steinhart -= 273.15;                         // convert to C
 
  //Serial.print("Temperature "); 
  //Serial.print(steinhart);
  //Serial.println(" *C");

 
  delay(1000);



  // Display raw ADC value

  lcd.setCursor(11, 1); lcd.print(steinhart);

  if(Serial.read()>=0){

    //Serial.print("Temperature "); 
    Serial.print(steinhart);
    //Serial.print("Temperature "); 
    Serial.print(steinhart); 
    
  }
  
  count++;

  // Reset time counter
  display_time = millis();
  
}




