import numpy as np
import pylab
from calibration_on_the_spot.calibration_on_the_spot import fit_cots

#Simulation run parameters
frame_count = 20
random_seed = 42123


# COTS parameters 
initpix = (16,16)  # Tuple of initial pixels specifying the spot center (y,x) (Int,Int)
fitframes=range(0,frame_count-1) # Frames to be included in analysis (Ints)
pw = 1.0 # Width of square pixels [nm] (Float)
deltapix = 10 # Half width of array passed to fit (Int)
mux_init = 0.1 # Center x coordinate [nm] (Float)
muy_init = 0.1 # Center y coordinate [nm] (Float)
b_init = 105.0 # Background signal per pixel (Float)
N_init = 1e5 # Total signal value (Float)
sigma_init = 3.0 # PSF width [nm] (Float)
initvals = pylab.array([mux_init, muy_init, sigma_init, N_init, b_init])


gain_init = 25.0 # Gain (Float)
S0_init = 200.0 # Signal offset (Float)

estimation_tolerance = 0.15

def test_fit_cots():
    
    # Test COTs estimation algorithm for simulated EMCCD output data at three
    # different gain levels with a Gaussian source distribution

    test_gains=[20, 35, 50]

    for test_gain in test_gains:
    
        emccd_output_data = np.load("./test/data/32x32gaussian_2.4sigma_" \
                                    + str(test_gain) + "gain_20frames.npy")
    
        gain_estimates, S0_estimates, location_estimates, r_squared_values, variance = \
            fit_cots(fitframes, pw, initvals, initpix, deltapix, emccd_output_data,
            gain_init, S0_init)

        average_gain_estimate = np.mean(gain_estimates)

        print("Gain in test data: "+str(average_gain_estimate)+\
            'Estimated gain using COTs library: '+str(test_gain))


        assert average_gain_estimate > test_gain - (test_gain * estimation_tolerance)
        assert average_gain_estimate < test_gain + (test_gain * estimation_tolerance)

        print("Error at gain: "+str(test_gain)+' is: '+str(((average_gain_estimate-test_gain)/test_gain)))
