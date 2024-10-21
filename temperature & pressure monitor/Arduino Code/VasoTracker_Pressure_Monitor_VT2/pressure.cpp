// Include support function header
#include "pressure.h"
//#include "WheatstoneBridge.h"

// Return the first pressed button found
byte read_LCD_buttons()

// read the value from the analog pin connected to the LCD and determine which button is being pressed.
{
  int adc_key_in = analogRead(A5);
  if (adc_key_in > 1000) return btnNONE;
  if (adc_key_in < 50)   return btnRIGHT;
  if (adc_key_in < 250)  return btnUP;
  if (adc_key_in < 450)  return btnDOWN;
  if (adc_key_in < 650)  return btnLEFT;
  if (adc_key_in < 850)  return btnSELECT;
                         return btnNONE;
}

// Display full screen of text (2 rows x 16 characters)
void displayScreen(char row1[], char row2[])
{
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print(row1);
  lcd.setCursor(0, 1); lcd.print(row2);
}

// Displays text on screen and an ADC value at chosen position from selected pin. Waits for user input (select button by default)
int getValueADC(char row1[], char row2[], byte pos, byte pin, byte endButton)
{
  int value = 0;
  int lastValue = -1;
  byte buttons = btnNONE;

  uint8_t i;
  int avg1;
  #define       NUMSAMPLES 50 
  int      samples1[NUMSAMPLES];  

  displayScreen(row1, row2);
  delay(1000);
  while (buttons != endButton)
  {
    // Check inputs
    buttons = read_LCD_buttons();

    // Update readout from analog port
    // value = analogRead(pin);

    for (i = 0; i < NUMSAMPLES; i++){
      samples1[i] = analogRead(pin);
      delay(10);
    }

    avg1 = 0;
    for (i = 0; i < NUMSAMPLES; i++){
      avg1 += samples1[i];
    }

    avg1 /= NUMSAMPLES;

    value = avg1;

    // Update display
    if (value != lastValue)
    {
      lcd.setCursor(pos, 1); lcd.print("                ");
      lcd.setCursor(pos, 1); lcd.print(value);
      lastValue = value;
      delay(400);
    }
  }

  // Return the last measured ADC value
  return (value);
}

// Displays text on screen and a changeable value at chosen position. User input can change the value (left/right/up/down) and press select to accept it.
int getValueInRange(char row1[], char row2[], byte pos, double valueDefault, double valueMin, double valueMax, int valueStep, int valueStepLarge)
{
  int value = valueDefault;
  int lastValue = -1;
  int buttons = btnNONE;

  displayScreen(row1, row2);
  delay(500);
  while (buttons != btnSELECT)
  {
    // Check inputs
    buttons = read_LCD_buttons();
    switch (buttons)
    {
      case btnUP:
        value += valueStep;
        if (value > valueMax)
          value = valueMax;
        break;

      case btnRIGHT:
        value += valueStepLarge;
        if (value > valueMax)
          value = valueMax;
        break;

      case btnDOWN:
        value -= valueStep;
        if (value < valueMin)
          value = valueMin;
        break;

      case btnLEFT:
        value -= valueStepLarge;
        if (value < valueMin)
          value = valueMin;
        break;
    }

    // Update display
    if (value != lastValue)
    {
      lcd.setCursor(pos, 1); lcd.print("                ");
      lcd.setCursor(pos, 1); lcd.print(value);
      lastValue = value;
      delay(200);
    }
  }
  // Return the last selected value
  return (value);
}

// Choose whether or not to calibrate
int startup(char row1[], char row2[], char yes[], char no[], byte pos)
{
  int valueDef = 0;
  int lastValue = -1;
  byte choice;
  int buttons = btnNONE;

  displayScreen(row1, row2);
  while (buttons != btnSELECT)
  {
    // Check inputs
    buttons = read_LCD_buttons();
    switch (buttons)
    {
      case btnUP:
        lcd.setCursor(pos, 0); lcd.print(yes);
        choice = 1;
        break;

      case btnDOWN:
        lcd.setCursor(pos, 0); lcd.print(no);
        choice = 0;
        break;
    }

    // Update display
    /*if (valueDef != lastValue)
    {
      lcd.setCursor(pos, 0); lcd.print(choice);
      lastValue = valueDef;
      delay(200);
    }*/
  }

  // Return the last selected value
  return (choice);
}

// < Constructor >
/* Sets the proper analog pin to input. Also does calibration if given by user.
*/
WheatstoneBridge::WheatstoneBridge(byte AnalogPin, int inputMin, int inputMax, double outputMin, double outputMax)
{
  iPin = AnalogPin;
  pinMode(iPin, INPUT);
  iMin = inputMin;
  iMax = inputMax;
  oMin = outputMin;
  oMax = outputMax;
}

// < Destructor >
WheatstoneBridge::~WheatstoneBridge()
{
  // Nothing to destruct
}

// measureForce
/* Obtain the analog measurement from ADC and convert it by interpolation to a force using the latest calibration values.
*/
double WheatstoneBridge::measureForce()
{
  // Obtain ADC raw measurement
  lastForceADCRaw = analogRead(iPin);
  
  // 
  lastForce = ((oMax - oMin) / (iMax - iMin)) * (lastForceADCRaw - iMin) + oMin;
  
  // Return value
  return (lastForce);
}

// getLastForce
/* Return the last force calculation (does not perform a new reading).
*/
double WheatstoneBridge::getLastForce()
{
  return (lastForce);
}

// getLastForceRawADC
/* Return the last force raw ADC value (does not perform a new reading).
*/
int WheatstoneBridge::getLastForceRawADC()
{
  return (lastForceADCRaw);
}

// linearCalibration
/* Calibrates the Wheatstone bridge linear interpolation.
  inputMin: Minimum expected value of raw ADC input
  inputMax: Maximum expected value of raw ADC output
  outputMin:  First (lower) calibration point with a known force, usually 0.
  outputMax:  Second (higher) calibration point with a known force, usually near the maximum force measurable by the load cell used.
*/
void WheatstoneBridge::linearCalibration(int inputMin, int inputMax, double outputMin, double outputMax)
{
  iMin = inputMin;
  iMax = inputMax;
  oMin = outputMin;
  oMax = outputMax;
}
