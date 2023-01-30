from misc import Data, DataReader
from scipy.stats import pearsonr


def main():
    # ---- OPTIONS ---- #
    data_folder = "data"
    subjects = ("sujet1",)
    show_cop = False
    show_cop_displacement = True
    show_cop_velocity = True
    show_cop_acceleration = True
    show_sensors = False
    skip_huge_files = True
    # ----------------- #

    if show_sensors and skip_huge_files:
        raise ValueError("It is not possible to 'show_sensors' if 'skip_huge_files'")

    for subject in subjects:
        folder = f"{data_folder}/{subject}"
        filenames = DataReader.fetch_trial_names(folder)

        cycl_data = []
        right_gl_data = []
        left_gl_data = []
        right_sensor_data = []
        left_sensor_data = []
        for filename in filenames:
            # Load data
            cycl_data.append(DataReader.read_cycl_data(f"{data_folder}/{subject}/{filename}_CYCL.CSV"))
            right_gl_data.append(DataReader.read_gl_data(f"{data_folder}/{subject}/{filename}_GL_R.CSV"))
            left_gl_data.append(DataReader.read_gl_data(f"{data_folder}/{subject}/{filename}_GL_L.CSV"))
            right_sensor_data.append(
                DataReader.read_sensor_data(f"{data_folder}/{subject}/{filename}_R.CSV")
                if not skip_huge_files
                else None
            )
            left_sensor_data.append(
                DataReader.read_sensor_data(f"{data_folder}/{subject}/{filename}_L.CSV")
                if not skip_huge_files
                else None
            )

        # Concatenated the data in a single matrix
        cycl_data = concatenate_data(cycl_data)
        right_gl_data = concatenate_data(right_gl_data)
        left_gl_data = concatenate_data(left_gl_data)
        right_sensor_data = concatenate_data(right_sensor_data) if not skip_huge_files else None
        left_sensor_data = concatenate_data(left_sensor_data) if not skip_huge_files else None

        # Print if required
        filename = "All data"
        if show_cop:
            fig_name = f"CoP ({filename})"
            cycl_data.plot(
                figure=fig_name,
                title="CoP",
                x_label="X-coordinates (m)",
                y_label="Y-coordinates (m)",
                color="blue",
            )
            right_gl_data.plot(figure=fig_name, color="orange")
            left_gl_data.plot(figure=fig_name, color="orange")

        if show_cop_displacement:
            fig_name = f"CoP displacement ({filename})"
            cycl_data.plot_displacement(
                figure=fig_name,
                title=f"CoP displacement (blue) and Jump time (orange)\n"
                f"Integral correlation = {pearsonr(cycl_data.displacement_integral, cycl_data.flight_times[1:])[0]:0.3f}\n"
                f"Ranges correlation = {pearsonr(cycl_data.displacement_ranges, cycl_data.flight_times[1:])[0]:0.3f}\n",
                x_label="Time (s)",
                y_label="CoP displacement (m)",
                color="blue",
            )
            cycl_data.plot_flight_times(
                figure=fig_name,
                y_label="Jump time (s)",
                axis_on_right=True,
                color="orange",
            )

        if show_cop_velocity:
            fig_name = f"CoP Velocity ({filename})"
            cycl_data.plot_velocity(
                figure=fig_name,
                title=f"CoP velocity (blue) and Jump time (orange)\n"
                f"Integral correlation = {pearsonr(cycl_data.impulses, cycl_data.flight_times[1:])[0]:0.3f}\n"
                f"Ranges correlation = {pearsonr(cycl_data.velocity_ranges, cycl_data.flight_times[1:])[0]:0.3f}\n",
                x_label="Time (s)",
                y_label="CoP velocity (m/s)",
                color="blue",
            )
            cycl_data.plot_flight_times(
                figure=fig_name,
                y_label="Jump time (s)",
                axis_on_right=True,
                color="orange",
            )

        if show_cop_acceleration:
            fig_name = f"CoP Acceleration ({filename})"
            cycl_data.plot_acceleration(
                figure=fig_name,
                title=f"CoP acceleration (blue) and Jump time (orange)\n"
                f"Integral correlation = {pearsonr(cycl_data.acceleration_integral, cycl_data.flight_times[1:])[0]:0.3f}\n"
                f"Ranges correlation = {pearsonr(cycl_data.acceleration_ranges, cycl_data.flight_times[1:])[0]:0.3f}\n",
                x_label="Time (s)",
                y_label="CoP acceleration (m/s/s)",
                y_lim=[-60, 200],
                color="blue",
            )
            cycl_data.plot_flight_times(
                figure=fig_name,
                y_label="Jump time (s)",
                y_lim=[1, 2],
                axis_on_right=True,
                color="orange",
            )

        if show_sensors:
            fig_name = f"Sensors ({filename})"
            right_sensor_data.plot(
                figure=fig_name,
                title="Sensors",
                x_label="Time (s)",
                y_label="Sensors",
            )
            left_sensor_data.plot(figure=fig_name)

        if show_cop or show_cop_displacement or show_cop_velocity or show_cop_acceleration or show_sensors:
            Data.show()


def concatenate_data(all_data: list):
    concatenated_data = all_data[0]
    for data in all_data[1:]:
        concatenated_data = concatenated_data.concatenate(data)
    return concatenated_data


if __name__ == "__main__":
    main()
