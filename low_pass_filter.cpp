#include <cmath>  // For sqrt and tan

class CasualDigitalLowPass {
private:
    double time_past;
    double signal_past;
    double signal_2past;
    double filtered_past;
    double filtered_2past;
    int run;

public:
    CasualDigitalLowPass() {
        time_past = 0;
        signal_past = 0;
        signal_2past = 0;
        filtered_past = 0;
        filtered_2past = 0;
        run = 1;
    }

    double run_adjusted_butterworth(double cutoff_freq, double time, double unfiltered_signal) {
        // Initialize on first runs
        if (run <= 1) {
            signal_2past = signal_past;
            signal_past = unfiltered_signal;
            run++;
            return 0;
        }

        // Calculate time interval and sampling rate
        double time_interval = time - time_past;
        double sampling_rate = 1.0 / time_interval;
        time_past = time;

        // Corrected cutoff frequency
        double corrected_cutoff_freq = std::tan(M_PI * cutoff_freq / sampling_rate);

        // Coefficients calculation
        double K1 = std::sqrt(2) * corrected_cutoff_freq;
        double K2 = corrected_cutoff_freq * corrected_cutoff_freq;

        double a0 = K2 / (1 + K1 + K2);
        double a1 = 2 * a0;
        double a2 = a0;

        double K3 = a1 / K2;

        double b1 = -a1 + K3;
        double b2 = 1 - a1 - K3;

        // Calculate filtered signal
        double filtered_signal = a0 * unfiltered_signal + a1 * signal_past + a2 * signal_2past
                                 + b1 * filtered_past + b2 * filtered_2past;

        // Update past values
        signal_2past = signal_past;
        signal_past = unfiltered_signal;
        filtered_2past = filtered_past;
        filtered_past = filtered_signal;

        return filtered_signal;
    }
};

int main() {
    CasualDigitalLowPass filter;
    return 0;
};