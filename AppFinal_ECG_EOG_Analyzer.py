import sys  # Importing necessary libraries and modules
import numpy as np  # Numerical operations library
import scipy.signal as signal  # Signal processing library
import matplotlib.pyplot as plt  # Plotting library
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QWidget, QFileDialog, QComboBox  # Importing PyQt5 widgets for GUI
from PyQt5.QtCore import Qt  # Importing Qt core module
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # Matplotlib backend for Qt5
import wfdb  # Importing WFDB for ECG file handling
import mne  # Importing MNE for EEG file handling and processing

class MainWindow(QMainWindow):
    
    # Define the channel names as a class attribute
    channel_names = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz',
        'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2'
    ]
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('ECG and EOG Analyzer')  # Setting window title
        self.setGeometry(100, 100, 1200, 800)  # Setting window size and position

        central_widget = QWidget()  # Creating central widget
        self.setCentralWidget(central_widget)  # Setting central widget for the main window

        main_layout = QHBoxLayout()  # Creating horizontal layout for main window

        control_layout = QVBoxLayout()  # Creating vertical layout for control widgets
        plot_layout = QVBoxLayout()  # Creating vertical layout for plot widgets

        # Buttons
        self.ecg_button = QPushButton('ECG')  # Creating ECG button
        self.ecg_button.clicked.connect(self.load_ecg_file)  # Connecting ECG button click event to load ECG file method
        control_layout.addWidget(self.ecg_button)  # Adding ECG button to control layout

        self.eeg_button = QPushButton('EEG (Remove EOG)')  # Creating EEG button
        self.eeg_button.clicked.connect(self.load_eeg_file)  # Connecting EEG button click event to load EEG file method
        control_layout.addWidget(self.eeg_button)  # Adding EEG button to control layout

        self.analyze_button = QPushButton('Analysis')  # Creating Analysis button
        self.analyze_button.clicked.connect(self.analyze_signals)  # Connecting Analysis button click event to analysis method
        control_layout.addWidget(self.analyze_button)  # Adding Analysis button to control layout

        self.clear_button = QPushButton('Clear')  # Creating Clear button
        self.clear_button.clicked.connect(self.clear_plots)  # Connecting Clear button click event to clear plots method
        control_layout.addWidget(self.clear_button)  # Adding Clear button to control layout

        # Dropdown for method selection
        self.method_dropdown = QComboBox()  # Creating dropdown for method selection
        self.method_dropdown.addItem("Pan-Tompkin")  # Adding Pan-Tompkin method option
        self.method_dropdown.addItem("Correlation")  # Adding Correlation method option
        self.method_dropdown.addItem("Detrending")  # Adding Detrending method option
        control_layout.addWidget(self.method_dropdown)  # Adding dropdown to control layout

        # Time Sliders
        self.start_slider = QSlider(Qt.Horizontal)  # Creating horizontal slider for start time
        self.end_slider = QSlider(Qt.Horizontal)  # Creating horizontal slider for end time

        self.start_slider.setRange(0, 1000)  # Setting range for start slider
        self.end_slider.setRange(0, 1000)  # Setting range for end slider

        self.start_slider.setValue(0)  # Setting initial value for start slider
        self.end_slider.setValue(1000)  # Setting initial value for end slider

        self.start_slider.setTickPosition(QSlider.TicksBelow)  # Setting tick position for start slider
        self.end_slider.setTickPosition(QSlider.TicksBelow)  # Setting tick position for end slider

        self.start_slider.setTickInterval(50)  # Setting tick interval for start slider
        self.end_slider.setTickInterval(50)  # Setting tick interval for end slider

        self.start_slider.valueChanged.connect(self.update_start_label)  # Connecting start slider value change event to update start label method
        self.end_slider.valueChanged.connect(self.update_end_label)  # Connecting end slider value change event to update end label method

        self.start_label = QLabel('Start: 0')  # Creating label for start time display
        self.end_label = QLabel('End: 1000')  # Creating label for end time display

        control_layout.addWidget(self.start_label)  # Adding start label to control layout
        control_layout.addWidget(self.start_slider)  # Adding start slider to control layout
        control_layout.addWidget(self.end_label)  # Adding end label to control layout
        control_layout.addWidget(self.end_slider)  # Adding end slider to control layout

        # Labels and Result Boxes
        self.ecg_label = QLabel('Result for ECG peak detection')  # Creating label for ECG result
        self.ecg_result = QLabel('')  # Creating label for ECG peak result
        control_layout.addWidget(self.ecg_label)  # Adding ECG label to control layout
        control_layout.addWidget(self.ecg_result)  # Adding ECG result label to control layout

        self.eog_label = QLabel('EOG info')  # Creating label for EOG info
        self.eog_result = QLabel('')  # Creating label for EOG result
        control_layout.addWidget(self.eog_label)  # Adding EOG label to control layout
        control_layout.addWidget(self.eog_result)  # Adding EOG result label to control layout

        main_layout.addLayout(control_layout)  # Adding control layout to main layout

        # Plot Widgets
        self.raw_ecg_canvas = FigureCanvas(plt.Figure())  # Creating canvas for raw ECG plot
        self.analysed_ecg_canvas = FigureCanvas(plt.Figure())  # Creating canvas for analysed ECG plot
        self.raw_eeg_canvas = FigureCanvas(plt.Figure())  # Creating canvas for raw EEG plot
        self.proc_eeg_canvas = FigureCanvas(plt.Figure())  # Creating canvas for processed EEG plot

        plot_layout.addWidget(self.raw_ecg_canvas)  # Adding raw ECG canvas to plot layout
        plot_layout.addWidget(self.analysed_ecg_canvas)  # Adding analysed ECG canvas to plot layout
        plot_layout.addWidget(self.raw_eeg_canvas)  # Adding raw EEG canvas to plot layout
        plot_layout.addWidget(self.proc_eeg_canvas)  # Adding processed EEG canvas to plot layout

        main_layout.addLayout(plot_layout)  # Adding plot layout to main layout

        central_widget.setLayout(main_layout)  # Setting main layout for central widget

        # Apply style sheet
        self.setStyleSheet("""
            QPushButton#ecgButton {
                background-color: #4CAF50; /* Green */
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
            }
            QPushButton#ecgButton:hover {
                background-color: #45a049;
            }
            QPushButton#eegButton {
                background-color: #008CBA; /* Blue */
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
            }
            QPushButton#eegButton:hover {
                background-color: #005f6b;
            }
            QPushButton#analyzeButton {
                background-color: #f44336; /* Red */
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
            }
            QPushButton#analyzeButton:hover {
                background-color: #da190b;
            }
            QPushButton#clearButton {
                background-color: #555555; /* Dark Gray */
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
            }
            QPushButton#clearButton:hover {
                background-color: #333333;
            }
            QSlider#startSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #dddddd;
                margin: 2px 0;
            }
            QSlider#startSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #999999;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
            QSlider#endSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #dddddd;
                margin: 2px 0;
            }
            QSlider#endSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #999999;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
            QLabel {
                font-size: 14px;
                color: #333333;
            }
            QComboBox#methodDropdown {
                padding: 5px;
                font-size: 14px;
                background-color: white;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            QLabel#ecgResultLabel,
            QLabel#eogResultLabel {
                font-size: 12px;
                color: #777777;
            }
        """)

    def load_ecg_file(self):
        options = QFileDialog.Options()  # Creating options for file dialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Open ECG File", "", "ECG Files (*.dat *.hea)", options=options)  # Opening file dialog to get ECG file
        if fileName:  # If file is selected
            record = wfdb.rdrecord(fileName.replace('.dat', '').replace('.hea', ''))  # Reading ECG record
            self.ecg_data = record.p_signal[:, 0]  # Extracting ECG data from record
            self.plot_signal(self.raw_ecg_canvas, self.ecg_data, 'Raw ECG Signal', 'blue')  # Plotting raw ECG signal

    def load_eeg_file(self):
        options = QFileDialog.Options()  # Creating options for file dialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Open EEG File", "", "EEG Files (*.edf)", options=options)  # Opening file dialog to get EEG file
        if fileName:  # If file is selected
            raw = mne.io.read_raw_edf(fileName, preload=True)  # Reading raw EEG data
            eog_channels = mne.pick_types(raw.info, eeg=False, eog=True)  # Picking EOG channels from EEG data
            
            if len(eog_channels) > 0:  # If EOG channels are found
                eog_data = raw.get_data(picks=eog_channels)  # Getting EOG data
                self.eeg_data = raw.get_data()  # Getting EEG data
                self.eeg_data -= np.mean(eog_data, axis=0)  # Removing EOG artifacts from EEG data
                self.plot_signal(self.raw_eeg_canvas, self.eeg_data[0], 'Raw EEG Signal', 'green')  # Plotting raw EEG signal
                self.eog_result.setText("")  # Clearing EOG result text
            else:  # If no EOG channels are found
                self.eeg_data = raw.get_data()  # Getting EEG data
                self.plot_signal(self.raw_eeg_canvas, self.eeg_data[0], 'Raw EEG Signal', 'green')  # Plotting raw EEG signal
                self.eog_result.setText("No EOG channels found in the EEG file.")  # Setting EOG result text

    def plot_signal(self, canvas, data, title, color):
        start_index = self.start_slider.value()  # Getting start index from slider
        end_index = self.end_slider.value()  # Getting end index from slider
        ax = canvas.figure.add_subplot(111)  # Adding subplot to canvas
        ax.clear()  # Clearing subplot
        ax.plot(data[start_index:end_index], color=color)  # Plotting data on subplot
        ax.set_title(title)  # Setting subplot title
        canvas.draw()  # Redrawing canvas
    
    def analyze_signals(self):
        if hasattr(self, 'ecg_data'):  # If ECG data is loaded
            start_index = self.start_slider.value()  # Getting start index from slider
            end_index = self.end_slider.value()  # Getting end index from slider
            ecg_segment = self.ecg_data[start_index:end_index]  # Extracting segment of ECG data
            processed_ecg = self.preprocess_signal(ecg_segment)  # Preprocessing ECG segment

            # Select the detection method
            method = self.method_dropdown.currentText()  # Getting selected method from dropdown
            if method == "Pan-Tompkin":
                qrs_peaks = self.pan_tompkins(processed_ecg)  # Applying Pan-Tompkin method
            elif method == "Correlation":
                qrs_peaks = self.detect_r_peaks_by_correlation(processed_ecg, 500)  # Applying Correlation method
            elif method == "Detrending":
                qrs_peaks = self.detect_r_peaks_by_detrending(processed_ecg, 500)  # Applying Detrending method

            self.plot_signal_with_qrs(self.analysed_ecg_canvas, processed_ecg, qrs_peaks, 'Processed ECG Signal with QRS Detection', 'red')  # Plotting processed ECG with QRS detection
            self.ecg_result.setText(str(qrs_peaks))  # Setting ECG result text
        
        if hasattr(self, 'eeg_data'):  # If EEG data is loaded
            start_index = self.start_slider.value()  # Getting start index from slider
            end_index = self.end_slider.value()  # Getting end index from slider

            # Extracting segment of EEG data
            eeg_segment = self.eeg_data[:, start_index:end_index]

            # Finding EOG channels
            eog_indices = self.find_eog_channels(self.channel_names)

            # Removing EOG channels from EEG data
            clean_eeg_segment = self.remove_eog_channels(eeg_segment, eog_indices)

            # Preprocessing EEG segment without EOG channels
            processed_eeg = self.preprocess_signal(clean_eeg_segment)

            # Plotting processed EEG signal
            self.plot_signal(self.proc_eeg_canvas, processed_eeg[0], 'Processed EEG Signal', 'purple')

            # Detecting EOG peaks (simulated for demonstration)
            eog_peaks = self.detect_eog_peaks(processed_eeg[0])

            # Plotting EOG peaks as red dots on EEG plot
            self.plot_eog_peaks(self.proc_eeg_canvas, processed_eeg[0], eog_peaks)

            # Clearing EOG result text
            self.eog_result.setText("")

        #if hasattr(self, 'eeg_data'):  # If EEG data is loaded
        #    eog_indices = self.find_eog_channels(self.eeg_data)  # Finding indices of EOG channels
        #    eog_data = self.eeg_data[eog_indices]  # Extracting EOG data
        #    eog_peaks = self.detect_eog_peaks(eog_data)  # Detecting peaks in EOG data
        #    self.plot_eog_peaks(self.proc_eeg_canvas, eog_data, eog_peaks)  # Plotting EOG peaks
        #    self.eog_result.setText(f'EOG Result: Detected {len(eog_peaks)} EOG artifacts')  # Displaying result of EOG detection

    def preprocess_signal(self, signal_data):
        # Simple preprocessing: detrending and filtering
        detrended_signal = signal.detrend(signal_data)  # Detrending signal
        b, a = signal.butter(3, 0.1, btype='low')  # Butterworth low-pass filter coefficients
        filtered_signal = signal.filtfilt(b, a, detrended_signal)  # Applying filter to detrended signal
        return filtered_signal  # Returning filtered signal
    
    def detect_qrs(self, ecg_signal):
        # Simple QRS detection logic for demonstration purposes
        peaks, _ = signal.find_peaks(ecg_signal, distance=150, height=np.mean(ecg_signal) + 0.5 * np.std(ecg_signal))  # Finding peaks in ECG signal
        return peaks  # Returning detected peaks   
     
    def plot_signal_with_qrs(self, canvas, data, qrs_peaks, title, color):
        ax = canvas.figure.add_subplot(111)  # Adding subplot to canvas
        ax.clear()  # Clearing subplot
        ax.plot(data, color=color, label='ECG Signal')  # Plotting ECG signal
        if qrs_peaks.size > 0:  # If QRS peaks are detected
            ax.plot(qrs_peaks, data[qrs_peaks], 'ro', label='QRS Peaks')  # Plotting QRS peaks
        ax.set_title(title)  # Setting subplot title
        ax.legend()  # Adding legend to subplot
        canvas.draw()  # Redrawing canvas

    def clear_plots(self):
        self.analysed_ecg_canvas.figure.clear()  # Clearing analysed ECG canvas
        self.raw_ecg_canvas.figure.clear()  # Clearing raw ECG canvas
        self.raw_eeg_canvas.figure.clear()  # Clearing raw EEG canvas
        self.proc_eeg_canvas.figure.clear()  # Clearing processed EEG canvas
        self.analysed_ecg_canvas.draw()  # Redrawing analysed ECG canvas
        self.raw_ecg_canvas.draw()  # Redrawing raw ECG canvas
        self.raw_eeg_canvas.draw()  # Redrawing raw EEG canvas
        self.proc_eeg_canvas.draw()  # Redrawing processed EEG canvas
        self.ecg_result.setText("")  # Clearing ECG result text
        self.eog_result.setText("")  # Clearing EOG result text

    def update_start_label(self, value):
        self.start_label.setText(f'Start: {value}')  # Updating start label text
        if value >= self.end_slider.value():
            self.end_slider.setValue(value + 1)  # Adjusting end slider value

    def update_end_label(self, value):
        self.end_label.setText(f'End: {value}')  # Updating end label text
        if value <= self.start_slider.value():
            self.start_slider.setValue(value - 1)  # Adjusting start slider value


    def pan_tompkins(self, ECGdata):
        fs = 500  # Assuming a sampling rate of 500Hz

        # Step 1: Low pass filtering (Butterworth filter, 5th order)
        low_pass_cutoff = 40
        b_lp, a_lp = signal.butter(5, low_pass_cutoff / (fs / 2), 'low')  # Calculating Butterworth filter coefficients
        ECG_lp = signal.filtfilt(b_lp, a_lp, ECGdata)  # Applying low-pass filter to ECG data

        # Step 2: High pass filtering (Butterworth filter, 5th order)
        high_pass_cutoff = 5
        b_hp, a_hp = signal.butter(5, high_pass_cutoff / (fs / 2), 'high')  # Calculating Butterworth filter coefficients
        ECG_hp = signal.filtfilt(b_hp, a_hp, ECG_lp)  # Applying high-pass filter to ECG data

        # Step 3: Derivative filter
        derivative_filter = np.array([1, 2, 0, -2, -1]) * (1 / 8)  # Defining derivative filter coefficients
        ECG_der = np.convolve(ECG_hp, derivative_filter, mode='same')  # Applying derivative filter to ECG data

        # Step 4: Squaring
        ECG_squared = ECG_der ** 2  # Squaring the signal

        # Step 5: Moving average window integration (150 ms window)
        window_size = int(0.150 * fs)  # Calculating window size
        ECG_ma = np.convolve(ECG_squared, np.ones(window_size) / window_size, mode='same')  # Applying moving average filter

        # Step 6: Peak detection
        threshold = np.max(ECG_ma) * 0.6  # Calculating threshold for peak detection
        peaks, _ = signal.find_peaks(ECG_ma, height=threshold, distance=int(0.2 * fs))  # Finding peaks in ECG signal

        # Adjust peak locations to align with R-peaks
        adjusted_peaks = []  # Initializing list for adjusted peaks
        search_window = int(0.1 * fs)  # Calculating search window size
        for peak in peaks:
            left = max(0, peak - search_window)  # Setting left boundary of search window
            right = min(len(ECGdata), peak + search_window)  # Setting right boundary of search window
            if left < right:
                r_peak = np.argmax(ECGdata[left:right]) + left  # Adjusting peak location
                adjusted_peaks.append(r_peak)  # Adding adjusted peak to list

        return np.array(adjusted_peaks)  # Returning adjusted peaks

    def detect_r_peaks_by_correlation(self, ECGdata, fs):
        QRS_template = np.array([0, 1, 0])  # Dummy template for demonstration purposes
        ECGdata = (ECGdata - np.mean(ECGdata)) / np.max(np.abs(ECGdata))  # Normalizing ECG data
        QRS_template = (QRS_template - np.mean(QRS_template)) / np.max(np.abs(QRS_template))  # Normalizing QRS template

        correlation = np.convolve(ECGdata, np.flip(QRS_template), mode='same')  # Calculating cross-correlation

        threshold = max(correlation) * 0.6  # Calculating threshold for peak detection
        peaks, _ = signal.find_peaks(correlation, height=threshold, distance=int(0.2 * fs))  # Finding peaks in cross-correlation

        return peaks  # Returning detected peaks

    def detect_r_peaks_by_detrending(self, ECGdata, fs):
        # Step 1: Detrend the ECG signal to remove the baseline wander
        ECG_detrended = signal.detrend(ECGdata)  # Detrending ECG data

        # Step 2: Low-pass filter with cutoff frequency of 40Hz (Butterworth filter, 3rd order)
        low_pass_cutoff = 40  # Setting low-pass cutoff frequency
        b_lp, a_lp = signal.butter(3, low_pass_cutoff / (fs / 2), 'low')  # Calculating Butterworth filter coefficients
        ECG_filtered = signal.filtfilt(b_lp, a_lp, ECG_detrended)  # Applying low-pass filter to detrended ECG data

        # Step 3: High-pass filter with cutoff frequency of 0.5Hz (Butterworth filter, 3rd order)
        high_pass_cutoff = 0.5  # Setting high-pass cutoff frequency
        b_hp, a_hp = signal.butter(3, high_pass_cutoff / (fs / 2), 'high')  # Calculating Butterworth filter coefficients
        ECG_filtered = signal.filtfilt(b_hp, a_hp, ECG_filtered)  # Applying high-pass filter to filtered ECG data

        # Step 4: Derivative filter
        derivative_filter = np.array([1, 2, 0, -2, -1]) * (1 / 8)  # Defining derivative filter coefficients
        ECG_derivative = np.convolve(ECG_filtered, derivative_filter, mode='same')  # Applying derivative filter to filtered ECG data

        # Step 5: Squaring the signal to emphasize larger differences
        ECG_squared = ECG_derivative ** 2  # Squaring the signal

        # Step 6: Moving average filter with window size of 150ms
        window_size = int(0.150 * fs)  # Calculating window size
        ECG_ma = np.convolve(ECG_squared, np.ones(window_size) / window_size, mode='same')  # Applying moving average filter

        # Step 7: Peak detection using a dynamic threshold
        threshold = np.mean(ECG_ma) + 0.5 * np.std(ECG_ma)  # Calculating threshold for peak detection
        peaks, _ = signal.find_peaks(ECG_ma, height=threshold, distance=int(0.2 * fs))  # Finding peaks in moving average

        # Step 8: Adjust peak locations to align with R-peaks
        adjusted_peaks = []  # Initializing list for adjusted peaks
        search_window = int(0.1 * fs)  # Calculating search window size
        for peak in peaks:
            left = max(0, peak - search_window)  # Setting left boundary of search window
            right = min(len(ECGdata), peak + search_window)  # Setting right boundary of search window
            if left < right:
                r_peak = np.argmax(ECGdata[left:right]) + left  # Adjusting peak location
                adjusted_peaks.append(r_peak)  # Adding adjusted peak to list

        return np.array(adjusted_peaks)  # Returning adjusted peaks
    
    def find_eog_channels(self, channel_names):
        """
        Finds the indices of EOG channels based on their positions or characteristics.
        
        Parameters:
        channel_names (list of str): List of channel names.
        
        Returns:
        eog_indices (list of int): List of indices of EOG channels.
        """
        eog_indices = []
        for idx, name in enumerate(channel_names):
            # Example: Check for positions near the outer canthi of the eyes
            if name in ['Fp1', 'Fp2']:  # Modify this list based on your EEG system's electrode positions
                eog_indices.append(idx)
        return eog_indices

    def remove_eog_channels(self, eeg_data, eog_indices):
        """
        Removes EOG channels from the EEG data based on predefined channel names.
        Parameters:
        eeg_data (ndarray): EEG data array of shape (channels, samples).
        eog_indices (list of int): List of indices of EOG channels.

        Returns:
        clean_eeg (ndarray): EEG data with EOG channels removed.
        """
        clean_eeg_data = np.delete(eeg_data, eog_indices, axis=0)  # Removing EOG channels from EEG data
        return clean_eeg_data

    def detect_eog_peaks(self, eog_data):
        # Simple peak detection for demonstration purposes
        peaks, _ = signal.find_peaks(eog_data, distance=150, height=np.mean(eog_data) + 0.5 * np.std(eog_data))
        return peaks

    def plot_eog_peaks(self, canvas, data, eog_peaks):
        ax = canvas.figure.get_axes()[0]  # Getting the first (and only) axis from the canvas
        ax.plot(eog_peaks, data[eog_peaks], 'ro', label='EOG Peaks')  # Plotting EOG peaks
        ax.legend()  # Adding legend to the plot
        canvas.draw()  # Redrawing canvas

if __name__ == '__main__':
    app = QApplication(sys.argv)  # Creating Qt application instance
    app.setStyle('Fusion')  # Setting application style to 'Fusion' for consistent look
    main_win = MainWindow()  # Creating main window instance
    main_win.show()  # Displaying main window
    sys.exit(app.exec_())  # Executing application

