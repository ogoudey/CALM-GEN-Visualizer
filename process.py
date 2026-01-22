import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
def get_values_from_data(csv_path, header="average_pupil_diameter_mm"):
    """
    Returns `header` data column from the CSV.
    """
	
    df = pd.read_csv(csv_path)
    return df[header].values

def fit_increments(inc):
    """Returns a pupil size estimator given the sampled pupil sizes."""
    x = np.linspace(0, 1, len(inc), endpoint=True)
    print(len(inc))
    y = inc
    #coeffs = np.polyfit(x, y, 3)
    #funk = np.poly1d(coeffs)
    p = Polynomial.fit(x, y, deg=min(3,len(inc)-1))
    funk = p
    return funk

def clean_pupil_sizes(pupil_sizes):
    """Interpolates missing data"""
    arr = np.array(pupil_sizes, dtype=float)
    arr[(arr < 0.8) | (arr > 10)] = np.nan
    valid = ~np.isnan(arr)
    x = np.arange(len(arr))
    cleaned = np.interp(x, x[valid], arr[valid])

    return cleaned

def calculate(pupil_sizes, brightnesses, predict_function, N=2, window_size=2):
    """
    Calculate the "average power".

    Does `pupil_sizes` - `predicted_pupil_sizes`

    Args:
        pupil_sizes: The actual recorded pupil sizes
        brightnesses: The recorded brightness of the screen (previously called "luminance")
        predict_function: The "calibrated" pupil_size estimator

    """
    cleaned_pupil_sizes = clean_pupil_sizes(pupil_sizes)
    predicted_pupil_sizes = np.array( [predict_function(l) for l in brightnesses] )
    print(f"cleaned_pupil_sizes: {cleaned_pupil_sizes}")
    print(f"predicted_pupil_sizes: {predicted_pupil_sizes}")
    
    difference = cleaned_pupil_sizes - predicted_pupil_sizes
    print(f"difference: {difference}")

    y_padded = np.pad(difference, (N//2, N-1-N//2), mode='edge')
    predicted_pupil_size_moving_avg = np.convolve(y_padded, np.ones((N,))/N, mode='valid')
    print(f"predicted_pupil_size_moving_avg: {predicted_pupil_size_moving_avg}")
    power = predicted_pupil_size_moving_avg ** 2

    moving_avg_power = np.convolve(power, np.ones(window_size)/window_size, mode='valid')
    return predicted_pupil_size_moving_avg, moving_avg_power

def analyze_world_interaction(calibration_path_str: str, data_path_str: str):
    increments = get_values_from_data(calibration_path_str) # a df[header].values object (a column?)
    cleaned_increments = clean_pupil_sizes(increments)
    predict_function = fit_increments(cleaned_increments)
    brightnesses = get_values_from_data(calibration_path_str, "screen_brightness")
    predictions = np.array( [predict_function(l) for l in brightnesses] )
    save_calibration_plot(increments, brightnesses, predictions)

    timestamps = get_values_from_data(data_path_str, "timestamp") 
    pupil_sizes = clean_pupil_sizes(get_values_from_data(data_path_str))
    brightnesses = get_values_from_data(data_path_str, "screen_brightness")
    predicted_pupil_sizes = np.array( [predict_function(l) for l in brightnesses] )
    difference, arousal = calculate(pupil_sizes, brightnesses, predict_function) # two np.array-s
    return timestamps, pupil_sizes, brightnesses, predicted_pupil_sizes, difference, arousal

def save_calibration_plot(increments, brightnesses, predictions):
    plt.figure()
    plt.plot(increments, label="Pupil Size (Increments)", linewidth=2)
    plt.plot(brightnesses, label="Brightness", linewidth=2)
    plt.plot(predictions, label="Prediction function", linewidth=2)
    plt.title("Calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig("calibration_validation.png")






def plot_full_data_for_one_csv():
    csv_path="data/Calibration_log_11251606_001.csv"

    # --- Calibrate ---
    increments = get_values_from_data(csv_path)[:17]
    predict_function = fit_increments(increments)
    brightnesses = get_values_from_data(csv_path, "screen_brightness")[:17]

    csv_path = "data/Pupil_log_Standard_11251608_001.csv"
    # --- Calculate ---
    pupil_sizes = get_values_from_data(csv_path)
    brightnesses = get_values_from_data(csv_path, "screen_brightness")
    predicted_pupil_sizes = np.array( [predict_function(l) for l in brightnesses] )
    difference, arousal = calculate(pupil_sizes, brightnesses, predict_function)

    print("len(increments):", len(increments))
    print("len(predicted_pupil_sizes):", len(predicted_pupil_sizes))
    print("len(pupil_sizes):", len(pupil_sizes))
    print("len(brightnesses):", len(brightnesses))
    print("len(arousal):", len(arousal))

    # --- Plot calculation ---
    plt.figure()
    plt.plot(brightnesses, label="Brightness", linewidth=2)
    plt.plot(clean_pupil_sizes(pupil_sizes), label="Actual Pupil Size", linewidth=2)
    plt.plot(difference, label="Difference", linewidth=2)
    plt.plot(np.array( [predict_function(l) for l in brightnesses] ), label="Predicted Pupil Size", linewidth=2)
    plt.plot(arousal, label="Arousal", linewidth=2)
    plt.title("Brightness vs Pupil Size vs Predicted Pupil Size vs Arousal")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main(csv_path="data/Calibration_log_11251606_001.csv"):
    # Splits are within data (assumed)

    # --- Calibrate ---
    increments = get_values_from_data(csv_path)[:17]
    predict_function = fit_increments(increments)
    brightnesses = get_values_from_data(csv_path, "screen_brightness")[:17]

    # --- Plot calibration ---
    plt.figure()
    plt.plot(increments, label="Pupil Size (Increments)", linewidth=2)
    plt.plot(brightnesses, label="Brightness", linewidth=2)
    plt.plot(np.array( [predict_function(l) for l in brightnesses] ), label="Prediction function", linewidth=2)
    plt.title("Calibration")
    plt.legend()
    plt.tight_layout()
    #plt.show()

    plt.figure()
    csvs = {"data/Pupil_log_Standard_11251608_001.csv": "Standard", "data/Pupil_log_AI_11251615_001.csv": "AI"}
    for csv_path, title in csvs.items():    
        # --- Calculate ---
        pupil_sizes = get_values_from_data(csv_path)
        brightnesses = get_values_from_data(csv_path, "screen_brightness")
        predicted_pupil_sizes = np.array( [predict_function(l) for l in brightnesses] )
        difference, arousal = calculate(pupil_sizes, brightnesses, predict_function, N=10, window_size=1)

        print("len(increments):", len(increments))
        print("len(predicted_pupil_sizes):", len(predicted_pupil_sizes))
        print("len(pupil_sizes):", len(pupil_sizes))
        print("len(brightnesses):", len(brightnesses))
        print("len(arousal):", len(arousal))

        # --- Plot calculation ---
        plt.plot(difference, label=f"Difference {title}", linewidth=2)
    plt.title("Difference between Actual and Predicted Pupil Size")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_full_data_for_one_csv()
    main()