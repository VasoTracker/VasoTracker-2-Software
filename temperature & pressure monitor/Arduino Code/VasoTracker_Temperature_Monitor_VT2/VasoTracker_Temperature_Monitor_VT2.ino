/* VasoTracker Temperature Monitor Code: Dr. N8-Style*/
/* This code was written by Calum Wilson.*/
/* It was made significantly better with input from Nathan Tykocki.*/

//  Include the LCD, wheatstone bridge, and stepper motor support libraries and headers
#include <LiquidCrystal.h> // Include the LCD library

// Initialize the library with the numbers of the interface pins
//LiquidCrystal lcd(8, 9, 4, 5, 6, 7); 

//Shitty ebay LCD panels
// fill in aduino pin #s used to control LCD
const int rs=8; // arduino pin connect to LCD RS pin
const int en=9; // arduino pin connect to LCD Enable pin
const int d4=4;  // arduino pin connected to LCD data bit 4 pin
const int d5=5;  // arduino pin connected to LCD data bit 5 pin
const int d6=6;  // arduino pin connected to LCD data bit 6 pin
const int d7=7;  // arduino pin connected to LCD data bit 7 pin

LiquidCrystal lcd(rs, en, d4, d5, d6, d7); // constructor to initialize pins in library


#define btnRIGHT 0
#define btnUP 1
#define btnDOWN 2
#define btnLEFT 3
#define btnSELECT 4
#define btnNONE 5
#define btnUNKNOWN 6

// Initialize the thermistor values
#define thermistor_pin A2 // which analog thermistor input pin // Normally A0 or A2
#define nominal_temp 25 // nominal resistance temperature (almost always 25 C)   
#define nominal_resistance 10000 // nominal  resistance at nominal resistance temperature   
#define beta_coeff 3950 // beta coefficient of the thermistor
#define series_resistor 10000 // the value of the 'other' resistor    
#define output_pin  3

//  Setup the data recording (data to be sent out, sampling rate)
//******************************************************************************************************************
  #define       NUMSAMPLES 5               //  Change this to alter the number of samples averaged.                *
  unsigned long TIME_DELAY = 5;            //  Change this to alter the time delay for wheatstone bridge output.   *
  unsigned long LCD_TIME_DELAY = 500;      // Setting a delay to update the LCD screen

  #define initial_set_point 25.0 // initial set point
  int min_temp = 20;
  int max_temp = 40;
//******************************************************************************************************************


uint16_t      samples[NUMSAMPLES];      // Needed to determine how the samples will be summed and averaged.
unsigned long previousMillis = 0;        // Resetting the clock to determine how much time has elapsed.
unsigned long initMillis = 0;            // Resetting the clock to determine how much time has elapsed.
unsigned long elapsedMillis = 0;
bool update_flag = 0;                    // Flag to update the LCD display
double setpoint, test, cal_temp_low;


// Function for reading the keypad
int readkeypad(){
  int adc_key_in = analogRead(A0); //Normally 5 or 0
  int ret = btnUNKNOWN;
  
  if (adc_key_in < 50) ret = btnRIGHT;
  if ( (adc_key_in > 50) && (adc_key_in < 250) ) ret = btnUP;
  if ( (adc_key_in > 250) && (adc_key_in < 450) ) ret = btnDOWN;
  if ( (adc_key_in > 450) && (adc_key_in < 650) ) ret = btnLEFT;
  if ( (adc_key_in > 650) && (adc_key_in < 850) ) ret = btnSELECT;
  if (adc_key_in > 850) ret = btnNONE; 
  return ret;
 }

void displayScreen(char row1[], char row2[])
{
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print(row1);
  lcd.setCursor(0, 1); lcd.print(row2);
}

// Setup function
void setup()
{
  // Initialize LCD screen and intro
  lcd.begin(16, 2); displayScreen("VasoTracker", "Temp Monitor"); delay(3000);
  
  // Main screen
  displayScreen("", "Act (*C):  ");
  
  Serial.begin(9600);

  // Setpoint
  setpoint = initial_set_point;

  //Set the output pin
  pinMode(output_pin, OUTPUT);
}

// < Main program loop >
void temperature(bool update_flag, double setpoint) {
  uint8_t i;
  float ADC_average;
  float resistance;
  float temperature;
  unsigned long currentMillis = millis();
  
  if ((currentMillis - previousMillis) >= TIME_DELAY) {

    // Take readings
    for (i = 0; i < NUMSAMPLES; i++) {
      samples[i] = analogRead(thermistor_pin);
    }
  
    // Sum readings
    ADC_average = 0;
    for (i = 0; i < NUMSAMPLES; i++) {
      ADC_average += samples[i];
    }

    // Average the readings
    ADC_average /= NUMSAMPLES;
  
    // Convert the voltage drop over the thermistor to resistance using the Steinhart equation
    resistance = series_resistor / ((1023 / ADC_average) - 1);
    temperature = 1 / (log(resistance / nominal_resistance) / beta_coeff + (1.0 / (nominal_temp + 273.15))) - 273.15;            

    // Ensure the "VasoTracker Temp" label is always displayed
    lcd.setCursor(0, 0); 
    lcd.print("VasoTracker Temp");

    // Update temperature reading only when the flag is set
    if (update_flag == 1) {
      lcd.setCursor(11, 1); // Update the temperature on the second line
      lcd.print(temperature);
    }

    // Print to the serial port in VasoTracker format
    Serial.print("<T1"); Serial.print(":"); Serial.print(temperature); Serial.print(";");
    Serial.print("T2"); Serial.print(":"); Serial.print(setpoint); Serial.print(";"); Serial.println(">");
  }
}







//If you want to do more things it's easy to add the loops here. Like if you wanted to do only 1 transducer and add temperature controls to a single unit...
unsigned long currentMillis_LCD = millis();
void loop(){
  
  //Get the current time
  currentMillis_LCD = millis();
  elapsedMillis = currentMillis_LCD - initMillis;

  // Check if we should update display
  if (elapsedMillis > LCD_TIME_DELAY){
    initMillis = currentMillis_LCD;
    update_flag = 1;
  


    // Check if any of the LCD shield buttons have been pressed
    int lcd_button_adc = analogRead(A0);
    int buttons2 = readkeypad();
    switch(buttons2){
      
      case btnUP:
        setpoint += 1;
        if(setpoint > max_temp)
          setpoint = max_temp;
        break;
        
      case btnRIGHT:
        setpoint += 0.5;
        if(setpoint > max_temp)
          setpoint = max_temp;
        break;
  
      case btnDOWN:
        setpoint -= 1;
        if(setpoint < min_temp)
          setpoint = min_temp;
        break;
  
      case btnLEFT:
        setpoint -= 0.5;
        if(setpoint < min_temp)
          setpoint = min_temp;
        break;
    }
   }

  // Get the current temperature
  temperature(update_flag, setpoint);
  update_flag = 0;
}
