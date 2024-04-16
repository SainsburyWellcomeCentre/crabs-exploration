import csv
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

    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            true_positives_list.append(int(row['True Positives']))
            missed_detections_list.append(int(row['Missed Detections']))
            false_positives_list.append(int(row['False Positives']))
            num_switches_list.append(int(row['Number of Switches']))
            total_ground_truth_list.append(int(row['Total Ground Truth']))

    return true_positives_list, missed_detections_list, false_positives_list, num_switches_list, total_ground_truth_list


def plot_histogram(true_positives_list, missed_detections_list, false_positives_list, num_switches_list, total_ground_truth_list):
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
    num_frames = len(true_positives_list)
    frames = range(1, num_frames + 1)

    plt.figure(figsize=(10, 6))

    # Plot metrics
    plt.plot(frames, true_positives_list, label='True Positives', color='g')
    plt.plot(frames, missed_detections_list, label='Missed Detections', color='r')
    plt.plot(frames, false_positives_list, label='False Positives', color='b')
    plt.plot(frames, num_switches_list, label='Number of Switches', color='y')

    # Plot total ground truth
    plt.plot(frames, total_ground_truth_list, label='Total Ground Truth', linestyle='--', color='k')

    plt.xlabel('Frame Number')
    plt.ylabel('Count')
    plt.title('Tracking Performance Metrics')
    plt.legend()
    plt.savefig("04.09.2023-04-Right_RE_test_model_20240413_081601_tracking_output.pdf")
    plt.show()


filename = '/Users/nikkhadijahnikaznan/Git/crabs-exploration/04.09.2023-04-Right_RE_test_model_20240413_081601_tracking_output'
true_positives_list, missed_detections_list, false_positives_list, num_switches_list, total_ground_truth_list = read_metrics_from_csv(filename)
plot_histogram(true_positives_list, missed_detections_list, false_positives_list, num_switches_list, total_ground_truth_list)
