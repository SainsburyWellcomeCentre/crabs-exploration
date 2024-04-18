import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_metrics_from_csv(filename):
    """
    Read the metrics from a CSV file.

    Parameters
    ----------
    filename : str
        Name of the CSV file to read.

    Returns
    -------
    tuple:
        Tuple containing lists of true positives, missed detections, false positives, number of switches, and total ground truth for each frame.
    """
    true_positives_list = []
    missed_detections_list = []
    false_positives_list = []
    num_switches_list = []
    total_ground_truth_list = []
    mota_value_list = []

    with open(filename, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            true_positives_list.append(int(row["True Positives"]))
            missed_detections_list.append(int(row["Missed Detections"]))
            false_positives_list.append(int(row["False Positives"]))
            num_switches_list.append(int(row["Number of Switches"]))
            total_ground_truth_list.append(int(row["Total Ground Truth"]))
            mota_value_list.append(float(row["Mota"]))

    return (
        true_positives_list,
        missed_detections_list,
        false_positives_list,
        num_switches_list,
        total_ground_truth_list,
        mota_value_list,
    )


def plot_histogram(filename):
    """
    Plot metrics along with the total ground truth for each frame.

    Parameters
    ----------
    true_positives_list : list[int]
        List of counts of true positives for each frame.
    missed_detections_list : list[int]
        List of counts of missed detections for each frame.
    false_positives_list : list[int]
        List of counts of false positives for each frame.
    num_switches_list : list[int]
        List of counts of identity switches for each frame.
    total_ground_truth_list : list[int]
        List of total ground truth objects for each frame.
    """
    (
        true_positives_list,
        missed_detections_list,
        false_positives_list,
        num_switches_list,
        total_ground_truth_list,
        mota_value_list,
    ) = read_metrics_from_csv(filename)
    filepath = Path(filename)
    plot_name = filepath.name

    num_frames = len(true_positives_list)
    frames = range(1, num_frames + 1)

    plt.figure(figsize=(10, 6))

    overall_mota = sum(mota_value_list) / len(mota_value_list)

    # Calculate percentages
    true_positives_percentage = [
        tp / gt * 100 if gt > 0 else 0
        for tp, gt in zip(true_positives_list, total_ground_truth_list)
    ]
    missed_detections_percentage = [
        md / gt * 100 if gt > 0 else 0
        for md, gt in zip(missed_detections_list, total_ground_truth_list)
    ]
    false_positives_percentage = [
        fp / gt * 100 if gt > 0 else 0
        for fp, gt in zip(false_positives_list, total_ground_truth_list)
    ]
    num_switches_percentage = [
        ns / gt * 100 if gt > 0 else 0
        for ns, gt in zip(num_switches_list, total_ground_truth_list)
    ]

    # Plot metrics
    plt.plot(
        frames,
        true_positives_percentage,
        label=f"True Positives ({sum(true_positives_list)})",
        color="g",
    )
    plt.plot(
        frames,
        missed_detections_percentage,
        label=f"Missed Detections ({sum(missed_detections_list)})",
        color="r",
    )
    plt.plot(
        frames,
        false_positives_percentage,
        label=f"False Positives ({sum(false_positives_list)})",
        color="b",
    )
    plt.plot(
        frames,
        num_switches_percentage,
        label=f"Number of Switches ({sum(num_switches_list)})",
        color="y",
    )

    plt.xlabel("Frame Number")
    plt.ylabel("Percentage of Total Ground Truth (%)")
    plt.title(f"{plot_name}_mota:{overall_mota:.2f}")
    # plt.text(0.5, 0.95, f'mAP: {map_value:.2f}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.legend()
    plt.savefig(f"{plot_name}.pdf")
    plt.show()


filename = "/Users/nikkhadijahnikaznan/Git/crabs-exploration/05.09.2023-05-Left_test_last_tracking_output"
plot_histogram(filename)
