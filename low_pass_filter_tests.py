import numpy as np
import matplotlib.pyplot as plt
import random 
from scipy.signal import savgol_filter
import hampel
import copy


"""
First we generate a random Sin Wave that we can run our tests on
"""
num_samples = 750
t = np.linspace(0, 5, num_samples)
x1 = random.uniform(0, 2 * np.pi) * 5
x2 = random.uniform(0, 2* np.pi) * 3
original_graph = np.sin(x1 * t)+np.cos(x2*t)
sin_noise = random.random()*np.sin(40 * t)
#(np.e**(t))*
white_noise = np.random.normal(0, 0.1, 750) + np.sin(20*x1*t)

#function for the sine wave
def y(t): return(original_graph + white_noise)

def moving_average(data, window_size):
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='valid')

y_ma = moving_average(y(t), 25)

class EmaFilterWithPriming:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.last_st = 0.0
        self.last_bt = 0.0
        self.priming_run = 1

    def run(self, input_value):
        if self.priming_run<3:
            self.priming_run += 1
            old_val = copy.copy(self.last_st)
            self.last_st = input_value  # Prime with input value
            if self.priming_run >= 2: self.last_bt = self.last_st - old_val
        else:
            old_st = copy.copy(self.last_st)
            self.last_st = self.alpha * input_value + (1 - self.alpha) * (self.last_st + self.last_bt)
            self.last_bt = self.beta * (self.last_st-old_st) + (1 - self.beta) * self.last_bt
        return self.last_st+0*self.last_bt
    

ema_filter = EmaFilterWithPriming(0.03, 0.09)

output_value = []
for i in y(t):
    input_value = i
    output_value.append(ema_filter.run(input_value))

class CasualDigitalLowPass:
    
    def __init__(self):
        self.time_past = 0
        self.signal_past = 0
        self.signal_2past = 0
        self.filtered_past = 0
        self.filtered_2past = 0
        self.run = 1


    def run_adjusted_butterworth(self, cutoff_freq, time, unfiltered_signal):

        if self.run <= 1:
            self.signal_2past = self.signal_past
            self.signal_past = unfiltered_signal    
            self.run = self.run+1      
            return 0

        time_interval = time-self.time_past
        sampling_rate = 1 / time_interval  # conversion to frequency hertz
        self.time_past = time

        
        #return 1
        
        corrected_cutoff_freq = np.tan(np.pi * cutoff_freq / sampling_rate)# radians for butterworth
        
        K1 = np.sqrt(2) * corrected_cutoff_freq
        K2 = corrected_cutoff_freq**2
        
        a0 = K2 / (1 + K1 + K2)
        a1 = 2 * a0
        a2 = a0
        
        K3 = a1 / K2
        
        b1 = -a1 + K3
        b2 = 1 - a1 - K3

        filtered_signal = a0 * unfiltered_signal + a1 * self.signal_past + a2 * self.signal_2past + b1 * self.filtered_past + b2 * self.filtered_2past
        #variable resets
        self.signal_2past = self.signal_past
        self.signal_past = unfiltered_signal
        self.filtered_2past = self.filtered_past
        self.filtered_past = filtered_signal
        
        return filtered_signal
    
# winter_low_filtered_sig = np.zeros_like(t)
# for i in range(2, len(t)):  # need to start at 3rd sample
#     winter_low_filtered_sig[i] = winter_low(x1, # where x1 is the cutoff frequency
#                                             t[i] - t[i-1],
#                                             y(t)[i], y(t)[i-1], y(t)[i-2],
#                                             winter_low_filtered_sig[i-1], winter_low_filtered_sig[i-2])

butt_filter = CasualDigitalLowPass()
winter_low_filtered_sig = []
print(type(t))
for i in range(len(t)):
    winter_low_filtered_sig.append(butt_filter.run_adjusted_butterworth(cutoff_freq = x1, time = t[i], unfiltered_signal=(y(t)[i])))

hampel_result = hampel.hampel(y(t))
y_sav = savgol_filter(y(t), window_length = 50, polyorder=5, mode = 'wrap')
# y_hat = (y_ma+y_emv[12:-12:])/2
#y_hat = savgol_filter(hampel_result.filtered_data, window_length = 700, polyorder=5)
#plot the waves:
plt.plot(t, y(t), color='gray', label='noisy_graph', alpha = 0.5)
plt.plot(t, original_graph, color='g', label='original')
plt.plot(t, y_sav, 'r', label = 'filtered')
plt.plot(t[:-24:], y_ma, 'orange')
plt.plot(t, output_value, 'yellow')
plt.plot(t, winter_low_filtered_sig, 'pink')

plt.xlabel('Time ')
plt.ylabel('Amplitude (temp or voltage)')
plt.title('Messy trig wave')
plt.grid(True)
plt.show()

