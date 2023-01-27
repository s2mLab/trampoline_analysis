from misc import DataReader, DataPlotter


def main():
    data_folder = "data"
    subjects = ("sujet1",)
    show_cop = False
    show_sensors = False
    load_huge_files = False

    for subject in subjects:
        folder = f"{data_folder}/{subject}"
        filenames = DataReader.fetch_trial_names(folder)
        for filename in filenames:
            # Load data
            cycl_data = DataReader.read_cycl_data(f"{data_folder}/{subject}/{filename}_CYCL.CSV")
            right_gl_data = DataReader.read_gl_data(f"{data_folder}/{subject}/{filename}_GL_R.CSV")
            left_gl_data = DataReader.read_gl_data(f"{data_folder}/{subject}/{filename}_GL_L.CSV")
            right_sensor_data = DataReader.read_sensor_data(f"{data_folder}/{subject}/{filename}_R.CSV") if load_huge_files else None
            left_sensor_data = DataReader.read_sensor_data(f"{data_folder}/{subject}/{filename}_L.CSV") if load_huge_files else None

            # Get some computed measures
            flight_times = cycl_data.flight_times()
            print(flight_times)

            # Print if required
            if show_cop:
                fig_name = f"COP ({filename})"
                DataPlotter.plot_xy(
                    cycl_data, figure_name=fig_name,
                    title="CoP in meters",
                    x_label="X-coordinates (m)",
                    y_label="Y-coordinates (m)",
                    color='blue',
                )
                DataPlotter.plot_xy(right_gl_data, figure_name=fig_name, color='orange')
                DataPlotter.plot_xy(left_gl_data, figure_name=fig_name, color='orange')
            if show_sensors:
                fig_name = f"Sensors ({filename})"
                DataPlotter.plot_t(
                    right_sensor_data,
                    figure_name=fig_name,
                    title="Sensors",
                    x_label="Time (s)",
                    y_label="Sensors",
                )
                DataPlotter.plot_t(left_sensor_data, figure_name=fig_name)

    if show_cop or show_sensors:
        DataPlotter.show()


if __name__ == "__main__":
    main()
