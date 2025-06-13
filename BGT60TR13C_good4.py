import spidev
import time
import numpy as np
import logging
import OPi.GPIO as GPIO

class BGT60TR13C:
    def __init__(self, spi_bus=1, spi_device=0, 
                 cs_pin=8, miso_pin=9, mosi_pin=10, sclk_pin=11,
                 rst_pin=7, irq_pin=25):
        
        self.spi = spidev.SpiDev()
        self.spi_bus = spi_bus
        self.spi_device = spi_device
        self.cs_pin = cs_pin
        self.miso_pin = miso_pin
        self.mosi_pin = mosi_pin
        self.sclk_pin = sclk_pin
        self.rst_pin = rst_pin
        self.irq_pin = irq_pin

        # Register Addresses
        self.REG_MAIN = 0x00
        self.REG_ADC0 = 0x01
        self.REG_CHIP_ID = 0x02
        self.REG_STAT1 = 0x03
        self.REG_PACR1 = 0x04
        self.REG_PACR2 = 0x05
        self.REG_SFCTL = 0x06
        self.REG_SADC = 0x07
        self.REG_CSI0 = 0x08
        self.REG_CSI1 = 0x09
        self.REG_CSI2 = 0x0A
        self.REG_CSIC = 0x0B
        self.REG_CSDS_0 = 0x0C
        self.REG_CSDS_1 = 0x0D
        self.REG_CSDS_2 = 0x0E
        self.REG_CSDSC = 0x0F
        self.REG_CSU1_0 = 0x10
        self.REG_CSU1_2 = 0x11
        self.REG_CSU1_3 = 0x12
        self.REG_CSD1_0 = 0x13
        self.REG_CSD1_1 = 0x14
        self.REG_CSD1_2 = 0x15
        self.REG_CSC1 = 0x16
        self.REG_CSU2_0 = 0x17
        self.REG_CSU2_1 = 0x18
        self.REG_CSU2_2 = 0x19
        self.REG_CSD2_0 = 0x1A
        self.REG_CSD2_1 = 0x1B
        self.REG_CSD2_2 = 0x1C
        self.REG_CSC2 = 0x1D
        self.REG_CSU3_0 = 0x1E
        self.REG_CSU3_1 = 0x1F
        self.REG_CSU3_2 = 0x20
        self.REG_CSD3_0 = 0x21
        self.REG_CSD3_1 = 0x22
        self.REG_CSD3_2 = 0x23
        self.REG_CSC3 = 0x24
        self.REG_CSU4_0 = 0x25
        self.REG_CSU4_1 = 0x26
        self.REG_CSU4_2 = 0x27
        self.REG_CSD4_0 = 0x28
        self.REG_CSD4_1 = 0x29
        self.REG_CSD4_2 = 0x2A
        self.REG_CSC4 = 0x2B
        self.REG_CCR0 = 0x2C
        self.REG_CCR1 = 0x2D
        self.REG_CCR2 = 0x2E
        self.REG_CCR3 = 0x2F
        self.REG_PLL1_0 = 0x30  # FSU - Frequency Start Unit
        self.REG_PLL1_1 = 0x31  # RSU - Ramp Step Unit
        self.REG_PLL1_2 = 0x32  # RTU - Ramp Time Unit
        self.REG_PLL1_3 = 0x33  # Additional PLL config
        self.REG_PLL14 = 0x34
        self.REG_PLL15 = 0x35
        self.REG_PLL16 = 0x36
        self.REG_PLL17 = 0x37
        self.REG_PLL20 = 0x38
        self.REG_PLL21 = 0x39
        self.REG_PLL22 = 0x3A
        self.REG_PLL23 = 0x3B
        self.REG_PLL24 = 0x3C
        self.REG_PLL25 = 0x3D
        self.REG_PLL26 = 0x3E
        self.REG_PLL27 = 0x3F
        self.REG_PLL30 = 0x40
        self.REG_PLL31 = 0x41
        self.REG_PLL32 = 0x42
        self.REG_PLL33 = 0x43
        self.REG_PLL34 = 0x44
        self.REG_PLL35 = 0x45
        self.REG_PLL36 = 0x46
        self.REG_PLL37 = 0x47
        self.REG_PLL40 = 0x48
        self.REG_PLL41 = 0x49
        self.REG_PLL42 = 0x4A
        self.REG_PLL43 = 0x4B
        self.REG_PLL44 = 0x4C
        self.REG_PLL45 = 0x4D
        self.REG_PLL46 = 0x4E
        self.REG_PLL47 = 0x4F
        self.REG_RFT0 = 0x55
        self.REG_RFT1 = 0x56
        self.REG_RLL_DFT0 = 0x59
        self.REG_STAT0 = 0x5D
        self.REG_FIFO_ACS = 0x60

        # Chirp calculation constants
        self.FREQ_REF = 80e6      # Reference frequency: 80 MHz
        self.FREQ_OSC = 80e6      # Oscillator frequency: 80 MHz
        self.LIGHT_SPEED = 3e8

        # Default chirp configuration
        self.chirp_config = {
            'start_freq_ghz': 61.0,
            'bandwidth_mhz': 1000,
            'ramp_time_us': 100,
            'sample_rate_khz': 2000
        }

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Processing parameters
        self.sample_per_chirp = 64
        self.motion_threshold = 100
        self.breath_filter = 0.1
        self.fft_threshold_db = 20

        # State variables
        self.last_samples = []
        self.motion_detected = False
        self.breathing_rate = 0
        self.distance = 0
        self.motion_history = []
        self.spectrum_data = []

        # Initialize chirp configuration
        self.update_chirp_configuration()

    def calculate_FSU(self, start_freq):
        """Calculate Frequency Start Unit"""
        start_freq_hz = start_freq * 1e9
        f_base = 60e9
        fsu = int((start_freq_hz - f_base) * (2**24) / self.FREQ_REF)
        return fsu & 0xFFFFFF

    def calculate_RSU(self, bandwidth, ramp_time):
        """Calculate Ramp Step Unit"""
        bandwidth_hz = bandwidth * 1e6
        ramp_time_s = ramp_time * 1e-6
        freq_step = bandwidth_hz / (ramp_time_s * self.FREQ_OSC)
        rsu = int(freq_step * (2**24) / self.FREQ_REF)
        return rsu & 0xFFFFFF
    
    def calculate_RTU(self, ramp_time, adc_samprate=2000):
        """Calculate Ramp Time Unit"""
        ramp_time_s = ramp_time * 1e-6
        adc_rate = adc_samprate * 1e3
        num_samples = int(ramp_time_s * adc_rate)
        rtu = num_samples // 8 + 4
        return rtu & 0xFFFFFF

    def calculate_range_resolution(self):
        """Calculate range resolution"""
        bandwidth = self.chirp_config['bandwidth_mhz'] * 1e6
        return self.LIGHT_SPEED / (2 * bandwidth)

    def calculate_max_range(self):
        """Calculate maximum range"""
        ramp_time = self.chirp_config['ramp_time_us'] * 1e-6
        return (self.LIGHT_SPEED * ramp_time) / 2

    def update_chirp_configuration(self):
        """Update register values based on current chirp configuration"""
        self.PLL1_0_CONFIG = self.calculate_FSU(self.chirp_config['start_freq_ghz'])
        self.PLL1_1_CONFIG = self.calculate_RSU(
            self.chirp_config['bandwidth_mhz'], 
            self.chirp_config['ramp_time_us']
        )
        self.PLL1_2_CONFIG = self.calculate_RTU(
            self.chirp_config['ramp_time_us'],
            self.chirp_config['sample_rate_khz']
        )
        
        # Default configurations as 32-bit values
        self.PLL1_3_CONFIG = 0x00000080  # Default from datasheet
        self.PACR1_CONFIG = 0x00000001
        self.PACR2_CONFIG = 0x00000007
        self.SFCTL_CONFIG = 0x00000001

        self.logger.info(f"Updated chirp configuration:")
        self.logger.info(f" - Start frequency: {self.chirp_config['start_freq_ghz']} GHz")
        self.logger.info(f" - Bandwidth: {self.chirp_config['bandwidth_mhz']} MHz")
        self.logger.info(f" - Ramp time: {self.chirp_config['ramp_time_us']} μs")
        self.logger.info(f" - Range resolution: {self.calculate_range_resolution():.3f} m")
        self.logger.info(f" - Max range: {self.calculate_max_range():.3f} m")
        self.logger.info(f" - PLL1_0 (FSU): 0x{self.PLL1_0_CONFIG:06X}")
        self.logger.info(f" - PLL1_1 (RSU): 0x{self.PLL1_1_CONFIG:06X}")
        self.logger.info(f" - PLL1_2 (RTU): 0x{self.PLL1_2_CONFIG:06X}")

    def set_custom_chirp(self, start_freq_ghz, bandwidth_mhz, ramp_time_us, sample_rate_khz=2000):
        """Set custom chirp parameters"""
        # Validate input ranges
        if not (59.0 <= start_freq_ghz <= 64.0):
            raise ValueError("Start frequency must be between 59-64 GHz")
        
        if not (50 <= bandwidth_mhz <= 4000):
            raise ValueError("Bandwidth must be between 50-4000 MHz")
        
        if not (10 <= ramp_time_us <= 1000):
            raise ValueError("Ramp time must be between 10-1000 μs")
        
        self.chirp_config = {
            'start_freq_ghz': start_freq_ghz,
            'bandwidth_mhz': bandwidth_mhz,
            'ramp_time_us': ramp_time_us,
            'sample_rate_khz': sample_rate_khz
        }
        
        self.update_chirp_configuration()

    def write_register(self, address: int, data: int) -> None:
        """Write 32-bit register"""
        try: 
            cmd = [0] * 4

            GPIO.output(self.cs_pin, GPIO.LOW)    # Enable chip access
            time.sleep(0.001)

            # Address byte: [7:1] Addr, [0] = 1 for write
            cmd[0] = ((address & 0x7F) << 1) | 0x01 

            # Data bytes (MSB first)
            cmd[1] = (data >> 16) & 0xFF 
            cmd[2] = (data >> 8) & 0xFF
            cmd[3] = data & 0xFF

            self.spi.xfer2(cmd)
            self.logger.debug(f'Successful Write Register 0x{address:02X}: 0x{data:08X}')
            
            time.sleep(0.001)
            GPIO.output(self.cs_pin, GPIO.HIGH)   # Disable chip access

        except Exception as e:
            GPIO.output(self.cs_pin, GPIO.HIGH)
            self.logger.error(f'Error Writing Register 0x{address:02X}: {e}')

    def read_register(self, address: int) -> int:
        """Read 32-bit register"""
        try: 
            # Address byte: [7:1] Addr, [0] = 0 for read
            addr_byte = (address & 0x7F) << 1
            cmd = [addr_byte, 0x00, 0x00, 0x00]
            
            GPIO.output(self.cs_pin, GPIO.LOW)
            time.sleep(0.001)
            
            response = self.spi.xfer2(cmd)
            
            time.sleep(0.001)
            GPIO.output(self.cs_pin, GPIO.HIGH)
            
            # Combine response bytes into 32-bit value
            result = (response[1] << 16) | (response[2] << 8) | response[3]
            self.logger.debug(f'Successful Read Register 0x{address:02X}: 0x{result:08X}')
            
            return result

        except Exception as e:
            GPIO.output(self.cs_pin, GPIO.HIGH)
            self.logger.error(f'Error Reading Register 0x{address:02X}: {e}')
            return 0
        
    def begin(self) -> bool:
        """Initialize the BGT60TR13C sensor"""
        try:
            # Setup GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.cs_pin, GPIO.OUT, initial=GPIO.HIGH)
            
            # Reset pin setup
            if self.rst_pin is not None:
                GPIO.setup(self.rst_pin, GPIO.OUT, initial=GPIO.HIGH)
                self.logger.info("Resetting BGT60TR13C...")
                GPIO.output(self.rst_pin, GPIO.LOW)
                time.sleep(0.01)
                GPIO.output(self.rst_pin, GPIO.HIGH)
                time.sleep(0.1)
            
            # IRQ pin setup
            if self.irq_pin is not None:
                GPIO.setup(self.irq_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

            # Setup SPI
            self.spi.open(self.spi_bus, self.spi_device)
            self.spi.max_speed_hz = 1_000_000
            self.spi.mode = 0
            self.spi.bits_per_word = 8

            # Configure registers with calculated values
            self.logger.info("Configuring BGT60TR13C registers...")
            
            # Write PLL configuration
            self.write_register(self.REG_PLL1_0, self.PLL1_0_CONFIG)
            time.sleep(0.01)
            self.write_register(self.REG_PLL1_1, self.PLL1_1_CONFIG)
            time.sleep(0.01)
            self.write_register(self.REG_PLL1_2, self.PLL1_2_CONFIG)
            time.sleep(0.01)
            self.write_register(self.REG_PLL1_3, self.PLL1_3_CONFIG)
            time.sleep(0.01)
            
            # Write other configuration registers
            self.write_register(self.REG_PACR1, self.PACR1_CONFIG)
            time.sleep(0.01)
            self.write_register(self.REG_PACR2, self.PACR2_CONFIG)
            time.sleep(0.01)
            self.write_register(self.REG_SFCTL, self.SFCTL_CONFIG)
            time.sleep(0.01)

            # Verify configuration
            chip_id = self.read_register(self.REG_CHIP_ID)
            self.logger.info(f"Chip ID: 0x{chip_id:08X}")

            if chip_id != 0:  # Any non-zero chip ID indicates communication
                self.logger.info("BGT60TR13C initialized successfully")
                return True
            else:
                self.logger.error("Failed to communicate with sensor")
                return False

        except Exception as e:
            self.logger.error(f"Error initializing BGT60TR13C: {e}")
            return False

    def start_chirp(self) -> None:
        """Start chirp sequence"""
        self.write_register(self.REG_MAIN, 0x01)

    def wait_fifo_ready(self, timeout: float = 1.0) -> bool:
        """Wait for FIFO to be ready"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            stat0 = self.read_register(self.REG_STAT0)
            if stat0 & 0x01:
                return True
            time.sleep(0.001)

        return False

    def read_fifo_data(self, num_samples=None) -> list:
        """Read FIFO data"""
        if num_samples is None:
            num_samples = self.sample_per_chirp

        try:
            samples = []
            for _ in range(num_samples):
                sample = self.read_register(self.REG_FIFO_ACS)
                samples.append(sample)
            return samples
        
        except Exception as e:
            self.logger.error(f'Error Reading FIFO data: {e}')
            return []

    def fft_analyze(self, samples) -> dict:
        """Perform FFT analysis on samples"""
        if len(samples) < 2:
            return None
        
        try:
            data = np.array(samples, dtype=np.float32)
            windowed_data = data * np.blackman(len(data))

            fft_result = np.fft.fft(windowed_data)
            magnitude = np.abs(fft_result)

            magnitude_db = 20 * np.log10(magnitude + 1e-10)

            sample_rate = self.chirp_config['sample_rate_khz'] * 1000
            freqs = np.fft.fftfreq(len(samples), 1/sample_rate)

            self.spectrum_data = {
                'frequencies': freqs[:len(freqs)//2],
                'magnitude_db': magnitude_db[:len(magnitude_db)//2],
                'magnitude': magnitude[:len(magnitude)//2]
            }

            return self.spectrum_data
        
        except Exception as e:
            self.logger.error(f'Error in FFT analysis: {e}')
            return None

    def detect_motion(self, samples) -> bool:
        """Detect motion from samples"""
        if len(samples) < 2:
            return False
            
        try:
            # Calculate RMS of current samples
            current_rms = np.sqrt(np.mean(np.array(samples)**2))
            
            if len(self.last_samples) > 0:
                # Calculate RMS of previous samples
                previous_rms = np.sqrt(np.mean(np.array(self.last_samples)**2))
                
                # Calculate difference
                rms_diff = abs(current_rms - previous_rms)
                
                # Motion detection threshold
                motion_detected = rms_diff > self.motion_threshold
                
                # Update motion history
                self.motion_history.append(motion_detected)
                if len(self.motion_history) > 10:
                    self.motion_history.pop(0)
                
                # Consider motion detected if recent history shows activity
                self.motion_detected = sum(self.motion_history) > 2
                
            self.last_samples = samples.copy()
            return self.motion_detected
            
        except Exception as e:
            self.logger.error(f"Error in motion detection: {e}")
            return False

    def estimate_distance(self, samples) -> float:
        """Estimate distance from samples"""
        try:
            spectrum = self.fft_analyze(samples)
            if spectrum is None:
                return 0
                
            # Find peak frequency
            magnitude = spectrum['magnitude']
            peak_idx = np.argmax(magnitude)
            peak_freq = abs(spectrum['frequencies'][peak_idx])
            
            # Convert frequency to distance
            # Distance = (c * f * T) / (2 * B)
            # where T is ramp time, B is bandwidth
            ramp_time = self.chirp_config['ramp_time_us'] * 1e-6
            bandwidth = self.chirp_config['bandwidth_mhz'] * 1e6
            
            distance = (self.LIGHT_SPEED * peak_freq * ramp_time) / (2 * bandwidth)
            
            self.distance = distance
            return distance
            
        except Exception as e:
            self.logger.error(f"Error in distance estimation: {e}")
            return 0

    def estimate_breathing_rate(self, samples) -> float:
        """Estimate breathing rate from samples"""
        try:
            # Apply bandpass filter for breathing frequency (0.1-0.5 Hz)
            if len(samples) < 64:
                return 0
                
            # Simple breathing rate estimation
            # This is a placeholder - implement proper signal processing
            return self.breathing_rate
            
        except Exception as e:
            self.logger.error(f"Error in breathing rate estimation: {e}")
            return 0

    def get_sensor_data(self) -> dict:
        """Get comprehensive sensor data"""
        try:
            # Start chirp and wait for data
            self.start_chirp()
            
            if self.wait_fifo_ready():
                samples = self.read_fifo_data()
                
                if samples:
                    # Process all measurements
                    motion = self.detect_motion(samples)
                    distance = self.estimate_distance(samples)
                    breathing = self.estimate_breathing_rate(samples)
                    spectrum = self.fft_analyze(samples)
                    
                    return {
                        'motion_detected': motion,
                        'distance_m': distance,
                        'breathing_rate_bpm': breathing,
                        'raw_samples': samples,
                        'spectrum': spectrum,
                        'timestamp': time.time()
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting sensor data: {e}")
            return None

    def calibrate_sensor(self, num_samples: int = 50) -> None:
        """Calibrate sensor baseline"""
        self.logger.info("Calibrating sensor...")
        
        baseline_samples = []
        for i in range(num_samples):
            self.start_chirp()
            if self.wait_fifo_ready():
                samples = self.read_fifo_data()
                if samples:
                    baseline_samples.extend(samples)
            time.sleep(0.01)
        
        if baseline_samples:
            baseline_rms = np.sqrt(np.mean(np.array(baseline_samples)**2))
            self.motion_threshold = baseline_rms * 2  # Set threshold as 2x baseline
            self.logger.info(f"Calibration complete. Motion threshold: {self.motion_threshold:.2f}")
        else:
            self.logger.warning("Calibration failed - no data received")

    def chirp_info(self) -> dict:
        """Get chirp configuration information"""
        return {
            'configuration': self.chirp_config,
            'range_resolution_m': self.calculate_range_resolution(),
            'max_range_m': self.calculate_max_range(),
            'register_values': {
                'FSU': f"0x{self.PLL1_0_CONFIG:06X}",
                'RSU': f"0x{self.PLL1_1_CONFIG:06X}",
                'RTU': f"0x{self.PLL1_2_CONFIG:06X}"
            }
        }

    def close(self) -> None:
        """Cleanup resources"""
        try:
            if self.spi:
                self.spi.close()
            GPIO.cleanup()
            self.logger.info("BGT60TR13C cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    
if __name__ == "__main__":
    sensor = BGT60TR13C(spi_bus=1, spi_device=0)

    try:
        if sensor.begin():
            print('Radar initialized successfully!')

            sensor.calibrate_sensor()

            info = sensor.chirp_info()
            print('\nChirp Configuration:')

            for key, value in info.items():
                print(f'{key}: {value}')

            print("\nStarting continuous measurement...")
            print("Press Ctrl+C to stop")

            while True:
                data = sensor.get_sensor_data()
                
                if data:
                    print(f"\rMotion: {'YES' if data['motion_detected'] else 'NO'} | "
                          f"Distance: {data['distance_m']:.2f}m | "
                          f"Breathing: {data['breathing_rate_bpm']:.1f} BPM", end='')
                
                time.sleep(0.1)
        else:
            print("Failed to initialize sensor!")
            
    except KeyboardInterrupt:
        print("\nStopping measurement...")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        sensor.close()
        print("Sensor closed.")