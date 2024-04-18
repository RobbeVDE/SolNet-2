import time
import os
class Timer:
    def __init__(self) -> None:
        self.start_time = time.time()
    
    def stop(self):
        self.end_time = time.time()

    def elapsed_time(self):
        
        return self.end_time-self.start_time
    
    def save_time(self, 
                  case):
        
        try:
            timers = pd.read_pickle("timers.pkl")
        except:
            timers = pd.DataFrame()
        
        timers.loc[:,case] = self.end_time-self.start_time

        timers.to_pickle("timers.pkl")
    