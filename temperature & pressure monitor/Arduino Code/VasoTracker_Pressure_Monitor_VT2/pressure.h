// Include the LCD and wheatstone bridge libraries
#include <LiquidCrystal.h>
#ifndef RS_WHEATSTONE_BRIDGE_INTERFACE
#define RS_WHEATSTONE_BRIDGE_INTERFACE
  #if (ARDUINO >= 100)
    #include "Arduino.h"
  #else
    #include "WProgram.h"
  #endif
#include <LiquidCrystal.h>

// Declare external global lcd
extern LiquidCrystal lcd;
int startup(char[], char[], char[], char[], byte);
void displayScreen(char[], char[]);                                         //Make the display read 16x2 lines.
int getValueADC(char[], char[], byte, byte, byte);                          //Read intial analog output from transducer.
int getValueInRange(char[], char[], byte, double, double, double, int, int);         //Initial ranges for pressure transducers.

// Button
#define btnRIGHT  1
#define btnUP     2
#define btnDOWN   4
#define btnLEFT   8
#define btnSELECT 16
#define btnNONE   32

class WheatstoneBridge
{
  public:
     WheatstoneBridge(byte AnalogPin, int inputMin = 0, int inputMax = 1023, double outputMin = 0.00, double outputMax = 500.00);
    ~WheatstoneBridge();
    double measureForce();
    double getLastForce();
    int getLastForceRawADC();
    void linearCalibration(int inputMin = 0, int inputMax = 1023, double outputMin = 0.00, double outputMax = 500.00);
    
  private:
  // < Local attributes >
    // Hardware
    byte iPin = A1;     // Defaults to "Strain 2"
    
    // Calibration
    int iMin = 0;
    int iMax = 1023;
    double oMin = 0.00;
    double oMax = 500.00;
    
    // Measurements
    int lastForceADCRaw = 0;
    double lastForce = 0.00;
};

#endif
