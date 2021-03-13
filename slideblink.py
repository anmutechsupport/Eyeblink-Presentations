"""
Adapted from https://github.com/NeuroTechX/bci-workshop
"""
import pyautogui
from time import sleep
from os import system as sys
from datetime import datetime
import numpy as np  # Module that simplifies computations on matrices
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data


class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3

""" EXPERIMENTAL PARAMETERS """
# Modify these to change aspects of the signal processing

# Length of the EEG data buffer (in seconds)
# This buffer will hold last n seconds of data and be used for calculations
BUFFER_LENGTH = 5

# Length of the epochs used to compute the FFT (in seconds)
EPOCH_LENGTH = 1

# Amount of overlap between two consecutive epochs (in seconds)
OVERLAP_LENGTH = 0.5

# Amount to 'shift' the start of each next consecutive epoch
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

# Index of the channel(s) (electrodes) to be used
# 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
INDEX_CHANNEL_BLINK = [1]
INDEX_CHANNEL_JAW = [2]
INDEX_CHANNELS = [INDEX_CHANNEL_BLINK, INDEX_CHANNEL_JAW]

if __name__ == "__main__":

    """ 1. CONNECT TO EEG STREAM """

    # Search for active LSL streams
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)

    # Get the stream info
    info = inlet.info()

    # Get the sampling frequency
    # This is an important value that represents how many EEG data points are
    # collected in a second. This influences our frequency band calculation.
    # for the Muse 2016, this should always be 256
    fs = int(info.nominal_srate())

    """ 2. INITIALIZE BUFFERS """

    # Initialize raw EEG data buffer
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state = None  # for use with the notch filter

    # Compute the number of epochs in "buffer_length"
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                              SHIFT_LENGTH + 1))

    # Initialize the band power buffer (for plotting)
    # bands will be ordered: [delta, theta, alpha, beta]
    band_buffer = np.zeros((n_win_test, 4))

    #list of buffers for iteration
    buffers = [[eeg_buffer, eeg_buffer], [band_buffer, band_buffer]]

    """ 3. GET DATA """

    # The try/except structure allows to quit the while loop by aborting the
    # script with <Ctrl-C>
    print('Press Ctrl-C in the console to break the while loop.')



    try:
        # The following loop acquires data, computes band powers, and calculates neurofeedback metrics based on those band powers
        while True:

            for index in range(len(INDEX_CHANNELS)):

                """ 3.1 ACQUIRE DATA """
                # Obtain EEG data from the LSL stream
                eeg_data, timestamp = inlet.pull_chunk(
                    timeout=1, max_samples=int(SHIFT_LENGTH * fs))

                # Only keep the channel we're interested in
                ch_data = np.array(eeg_data)[:, INDEX_CHANNELS[index]]

                # Update EEG buffer with the new data
                buffers[0][index], filter_state = utils.update_buffer(
                    buffers[0][index], ch_data, notch=True,
                    filter_state=filter_state)

                """ 3.2 COMPUTE BAND POWERS """
                # Get newest samples from the buffer
                data_epoch = utils.get_last_data(buffers[0][int(index)],
                                        EPOCH_LENGTH * fs)

                # Compute band powers
                band_powers = utils.compute_feature_vector(data_epoch, fs)
                buffers[1][index], _ = utils.update_buffer(buffers[1][index],
                                           np.asarray([band_powers]))

            print('Delta: ', "l'"+str(buffers[1][0][-1][Band.Delta])+" Delta: "+"r'"+str(buffers[1][1][-1][Band.Delta]))
            # head shakes are most responsive for forehead electrode delta(PSD>2.1)
            # eye blinks alpha(>0.9, <1.1)


            """ 3.3 COMPUTE NEUROFEEDBACK METRICS """

            # + buffers[1][1][-2][Band.Delta] > 4.4:
            if  buffers[1][1][-1][Band.Delta] > 2.2:
                print("""

                right

                """)
                pyautogui.press('right')
                # buffers[1][1][:-1, :]

            #  + buffers[1][0][-2][Band.Delta] > 4.4:
            elif buffers[1][0][-1][Band.Delta] > 2:
                print("""

                left

                """)
                pyautogui.press('left')
                # buffers[1][0][:-1, :]

    except KeyboardInterrupt:
        print('Closing!')
