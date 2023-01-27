from matplotlib import pyplot as plt

from misc import DataReader


def main():
    data_folder = "data"
    subjects = ("sujet1",)

    for subject in subjects:
        folder = f"{data_folder}/{subject}"
        filenames = DataReader.fetch_trial_names(folder)
        for filename in filenames:
            cycl_data = DataReader.read_cycl_data(f"{data_folder}/{subject}/{filename}_CYCL.csv")
            # cycl_data = DataReader.read_cycl_data(f"{data_folder}/{subject}/{filename}_CYCL.csv")

            plt.figure()
            plt.plot(cycl_data[:, 1], cycl_data[:, 2])
            plt.axis('equal')
            plt.show()


if __name__ == "__main__":
    main()
