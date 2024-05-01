import math
import threading
import time
class RadarSystem:
    """A class control all radars to make them detect satellite positions periodically
     Predictor can get the informations by visiting radars list"""
    current_time = 0

    def __init__(self, radars, earth):
        self.radars = radars
        self.earth = earth
        self.timer = None

    def check_los(self, sat_pos):
        print('check',self.current_time)
        self.current_time += 1
        for radar in self.radars:
            radar.line_of_sight(sat_pos, self.earth.ellipse_equation,self.current_time)

    def start_timer(self, interval=5):
        """Start a periodic timer that checks LOS every 'interval' seconds."""
        self.timer = threading.Timer(interval, self.run_timer)
        self.timer.start()

    def run_timer(self):
        """Handle the timer's event: check LOS and restart the timer."""
        """ TODO :
        True satellite positions need to be given, once the simulator finished """
        self.check_los(np.array([[1000000], [math.pi], [math.pi]]))
        self.start_timer()  # Restart the timer for the next check

    def stop_timer(self):
        """Stop the periodic timer."""
        if self.timer:
            self.timer.cancel()


""" TODO :
Init the positions of the radors, they should be on the surface of the earth.
"""
radars = [Radar(np.array([[0], [0], [0]])), Radar(np.array([[0], [0], [0]]))]
radar_system = RadarSystem(radars, Earth())
""" TODO :
change to real time interval
"""
radar_system.start_timer(1)

# Keep the program running to allow timer to trigger
# try:
#     while True:
#         time.sleep(1)
# except KeyboardInterrupt:
#     print("Stopping radar system.")
#     radar_system.stop_timer()