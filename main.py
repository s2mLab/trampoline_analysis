from misc import Data, DataReader, concatenate_data
from scipy.stats import pearsonr


def main():
    # ---- OPTIONS ---- #
    data_folder = "data"
    subjects = ("sujet1",)
    show_cop = False
    show_cop_displacement = True
    show_cop_velocity = True
    show_cop_acceleration = True
    show_sensors = True
    skip_huge_files = False
    # ----------------- #

    if show_sensors and skip_huge_files:
        raise ValueError("It is not possible to 'show_sensors' if 'skip_huge_files'")

    for subject in subjects:
        folder = f"{data_folder}/{subject}"
        filenames = DataReader.fetch_trial_names(folder)

        cycl_data = []
        force_data = []
        for filename in filenames:
            # Load data
            cycl_data.append(DataReader.read_cycl_data(f"{data_folder}/{subject}/{filename}"))
            force_data.append(
                DataReader.read_sensor_data(f"{data_folder}/{subject}/{filename}") if not skip_huge_files else None
            )

        # Concatenated the data in a single matrix
        cycl_data = concatenate_data(cycl_data)
        force_data = concatenate_data(force_data) if not skip_huge_files else None

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

        if show_cop_displacement:
            fig_name = f"CoP displacement ({filename})"
            cycl_data.plot_displacement(
                figure=fig_name,
                title=f"CoP displacement (blue) and Jump time (orange)\n"
                f"Integral correlation = {pearsonr(cycl_data.displacement_integral, cycl_data.flight_times[1:])[0]:0.3f}c\n"
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
                f"Integral correlation = {pearsonr(cycl_data.velocity_integral, cycl_data.flight_times[1:])[0]:0.3f}\n"
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
            fig_name = f"Forces ({filename})"
            force_data.plot(
                figure=fig_name,
                title="Sensor forces (blue) and Jump time (orange)\n"
                f"Integral correlation = {pearsonr(force_data.force_integral, force_data.flight_times[1:])[0]:0.3f}\n",
                x_label="Time (s)",
                y_label="Force (N)",
                color="blue",
            )
            force_data.plot_flight_times(
                figure=fig_name,
                y_label="Jump time (s)",
                y_lim=[1, 2],
                axis_on_right=True,
                color="orange",
            )
            # left_sensor_data.plot(figure=fig_name)

        if show_cop or show_cop_displacement or show_cop_velocity or show_cop_acceleration or show_sensors:
            Data.show()


if __name__ == "__main__":
    main()
