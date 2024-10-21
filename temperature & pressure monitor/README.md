VasoTracker Temperature Controller
======
<img src="https://github.com/kaelome/VasoTracker/blob/master/Splash.gif" width="400" align="right">

An Arduino-based temperature controller for the VasoTracker myograph system.

## Features

* **Reads and displays temperature sensed by an 10k NTC thermistor**

* **Implements bang-bang control of heating elements mounted on the VasoTracker Vessel Chamber**

* **Writes data to serial port for acquisition and display by the VasoTracker GUI**

## Required components:

The VasoTracker Temperature Controller (with temperature sensor and two heating elements) can be built for ~£75. Here is a table of the components required:

**Parts**|**Supplier**|**Part #**|**Qty**|**£/unit**|**Total (£)**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
Arduino Uno|Arduino|A000066 |1|20.7|20.7
12V power supply|RS|903-7048|1|12.68|12.68
Arduino Proto Shield |Arduino|A000077|1|10|10
LCD Shield|iTead Studio|1602 LCD|1|3.98|3.98
10k NTC Thermistor|Adafruit|372|1|3.03|3.03
Kool-Pak 0.2Ω Thick Film (Heating) Resistor|Caddock|MP825.20.0-1%|2|6.89|13.78
NPN MOSFET|Infineon Technologies|IRL540NPBF|1|0.99|0.99
Resistor kit|Velleman-kit|RES-E12|1|4.7|4.7
Jumper wire kit|RobotShop|RB-Cix-02|1|3.77|3.77
2-pin Screw Terminal|RobotShop|RB-Dfr-465|3|0.6|1.8
Arduino R3 Stackable Headers|Sparkfun Electronics|PRT-10007|1|1.14|1.14
LEGO™ Encolsure|LEGO|-|-|-|-
 | | | |**Subtotal**|**76.57**


## Building the VasoTracker Temperature Controller:

To build the temperature controller, there is a little bit of soldering involved:

<img src=https://github.com/VasoTracker/VasoTracker/blob/master/VasoTracker_Temperature_Controller/Images/Arduino%20Temp%20Controller.jpg>
</br>
Temperature controller schematic created using the breadboard view in Fritzing (http://fritzing.org/home/)

1.	Build the temperature sensor / heating circuit on the Arduino Proto shield, as shown in image above. Note that the heating elements (resistors) are connected to the Temperature controller via jumper cables and screw terminals mounted on the Proto Shield. Similarly, the temperature sensor is connected to a screw terminal mounted on the Proto Shield.
2.	Stack the completed Proto shield on top of the Arduino Uno.
3.	Place stackable headers on top of the Proto Shield (these are required to allow clearance for the pressure transducer connections and bend the A0 pin1 on the stackable header so that it does not connect to the Arduino Uno board.
4.	Stack the LCD Shield on top of the stackable headers.
5.	Use a female 1-pin jumper cable to connect the disconnected pin on the stackable header to the A5 pin on top of the LCD shield.
6.	 Wrap one end of a wire around the Arduino reset pin and another around Vin (5V).
7.	 Stack the LCD shield on top of the Arduino.
8.	 Connect a power supply to the ARDUINO device to switch the unit on. To supply sufficient power, it is recommended to use a 12V wall wort power supply, and to use the USB cable for data transmission.
9.	 Optionally, the device may be placed in an enclosure. We chose to build one from Lego® blocks because of their versatility .

**Notes:**
1. The A000077 Proto Shield is being discontinued. Many other suppliers sell equivalent prototyping boards (e.g. Adafruit 2077 Proto Shield).
2. For pictures of Step 3 onwards, check the similar instructions for the pressure controller here: https://github.com/VasoTracker/VasoTracker/tree/master/VasoTracker_Pressure_Monitor)**
2. The LCD shield uses the A0 pin for buttons. However, we (stupidly) connected our thermistor to A0. By bending the A0 pin on the stackable header and connecting it to A5 on the LCD shield, we are rerouting A0 on the LCD shield to A5 (thus enabling the thermistor to use A0). Obviously, by connecting the thermistor to A5, and altering the Arduino code, this step could be avoided!
3. Should prevent accidental resetting of the Arduino (which would be annoying if it occurred during an experiment)... connect reset pin to 5V via 120 ohm resistor
4. As an alternative to soldering the components onto the Arduino Proto Shield, an Arduino Proto Shield with Mini Breadboard could be used.




## Programming the VasoTracker Temperature Controller:

   * Download VasoTracker as a zip file and extract all files (if you don't have it already) from here: https://github.com/kaelome/VasoTracker
   * Download and install the Aruino IDE (Python 3.6) <https://www.arduino.cc/en/Guide/Windows#toc1>
   * Open the following Arduino file in the Arduino IDE: VasoTracker/VasoTracker_Temperature_Controller/VasoTracker_Temperature_Controller_Code/VasoTracker_Temperature_Controller_Code.ino
   *	Upload code to the Arduino by clicking the upload button <img src="https://github.com/kaelome/VasoTracker/blob/master/VasoTracker_Pressure_Monitor/Images/Arduino%20Upload%20Button.png" width="400" align="center">





### Setting up the VasoTracker Temperature Controller:

  * Ensure heating elements are secured to the Vessel Chamber
  * Place temperature sensor in the Vessel Chamber (it is held in place by a magnetic clamp)
  * Connect the heating elements to the Temperature Controller.
  * Connect Temperature Controller to data acquisition computer via Arduino USB cable.
  * Connect 12V power supply to the Arduino
  * Adjust setpoint to desired temperature using LCD buttons
