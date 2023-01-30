from matplotlib import pyplot as plt
import numpy as np


def _derivative(t: np.ndarray, data: np.ndarray, window: int = 1) -> np.ndarray:
    """
    Compute the non-sliding derivative of current data

    Parameters
    ----------
    t
        The time vector
    data
        The data to compute the derivative from
    window
        The sliding window to perform on

    Returns
    -------
    A Data structure with the value differentiated
    """

    two_windows = window * 2
    padding = np.nan * np.zeros((window, data.shape[1]))

    return np.concatenate(
        (
            padding,
            (data[:-two_windows, :] - data[two_windows:]) / (t[:-two_windows] - t[two_windows:])[:, np.newaxis],
            padding,
        )
    )


def _integral(t: np.ndarray, data: np.ndarray) -> float:
    """
    Compute the integral of the data using trapezoid

    Parameters
    ----------
    t
        The time vector
    data
        The data to compute the integral from

    Returns
    -------
    The integral of the data
    """
    return np.nansum((t[1:] - t[:-1]) * ((data[1:, :] + data[:-1, :]) / 2).T)


def _prepare_figure(
    figure: str = None,
    title: str = None,
    x_label: str = None,
    y_label: str = None,
    x_lim: list[float, float] = None,
    y_lim: list[float, float] = None,
    color: str = None,
    axis_on_right: bool = False,
    show_now: bool = False,
) -> tuple[plt.figure, plt.axis, str, bool]:
    """

    Parameters
    ----------
    figure
        The name of the figure. If two figures has the same name, they are drawn on the same graph
    title
        The title of the figure
    x_label
        The name of the X-axis
    y_label
        The name of the Y-axis
    x_lim
        The limits of the X-axis
    y_lim
        The limits of the Y-axis
    color
        The color of the plot
    show_now
        If the plot should be shown right now (blocking)

    Returns
    -------
    The figure and the axis handler
    """

    fig = plt.figure(figure)

    if not fig.axes:
        ax = plt.axes()
    else:
        ax = fig.axes[-1]
        if axis_on_right:
            ax = ax.twinx()

    if title is not None:
        ax.set_title(title)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if x_lim is not None:
        ax.set_ylim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if axis_on_right:
        ax.yaxis.tick_right()

    return fig, ax, color, show_now


class Data:
    def __init__(self, data=None, nb_sensors: int = 0, conversion_factor: float = 1):
        """
        Create a Data structure with 't' as time vector holder and 'y' as data holder
        Parameters
        ----------
        data
            Data to copy from
        nb_sensors
            The number of sensors (column) in the 'y' data holder
        conversion_factor
            The factor to convert the data when using the 'append' method
        """

        if data is not None:
            self.t = data.t
            self.y = data.y
            self.conversion_factor = data.conversion_factor
        else:
            self.t: np.ndarray = np.ndarray((0,))
            self.y: np.ndarray = np.ndarray((0, nb_sensors))
            self.conversion_factor = conversion_factor

    def append(self, t, y) -> None:
        """
        Add data to the data set

        Parameters
        ----------
        t
            The time to add
        y
            The data to add (converted with self.conversion_factor)
        """

        self.t = np.concatenate((self.t, (t,)))

        # Remove the MAX_INT and convert to m
        y = [data * self.conversion_factor if data != 2147483647 else np.nan for data in y]
        self.y = np.concatenate((self.y, (y,)))

    def concatenate(self, other):
        """
        Concatenate a data set to another, assuming the time of self is added as an offset to other

        Parameters
        ----------
        other
            The data to concatenate

        Returns
        -------
        The concatenated data
        """

        out = Data(data=self)
        time_offset = out.t[-1]
        out.t = np.concatenate((out.t, time_offset + other.t))
        out.y = np.concatenate((out.y, other.y))

        return out

    def plot(
        self,
        override_y: np.ndarray = None,
        **figure_options,
    ) -> plt.figure:
        """
        Plot the data as time dependent variables

        Parameters
        ----------
        override_y
            Force to plot this y data instead of the one in the self.y attribute
        figure_options
            see _prepare_figure inputs

        Returns
        -------
        The matplotlib figure handler if show_now was set to False
        """

        fig, ax, color, show_now = _prepare_figure(**figure_options)

        ax.plot(self.t, override_y if override_y is not None else self.y, color=color)

        if show_now:
            plt.show()

        return fig if not show_now else None

    @staticmethod
    def show() -> None:
        """
        Just a convenient method so one does not have to include matplotlib in other script just to call plt.show()
        """
        plt.show()


class CoPData(Data):
    def __init__(self, data: Data):
        super().__init__(data=data)
        self.displacement = self._compute_cop_displacement(window=2)
        self.velocity = _derivative(self.t, self.displacement, window=2)
        self.acceleration = _derivative(self.t, self.velocity, window=2)
        self.takeoffs_indices, self.landings_indices = self._compute_timings_indices()

    def concatenate(self, other):
        """
        Concatenate a data set to another, assuming the time of self is added as an offset to other

        Parameters
        ----------
        other
            The data to concatenate

        Returns
        -------
        The concatenated data
        """

        out = CoPData(super().concatenate(other))

        # Remove the extra index created from the discrepancy of the concatenated data
        out.takeoffs_indices = np.concatenate(
            (
                out.takeoffs_indices[0 : len(self.takeoffs_indices)],
                out.takeoffs_indices[len(self.takeoffs_indices) + 1 :],
            )
        )
        out.landings_indices = np.concatenate(
            (
                out.landings_indices[0 : len(self.landings_indices)],
                out.landings_indices[len(self.landings_indices) + 1 :],
            )
        )

        return out

    @property
    def mat_times(self) -> tuple[float, ...]:
        """
        Get the times in the mat
        """
        return tuple(
            self.t[takeoff] - self.t[landing]
            for takeoff, landing in zip(self.takeoffs_indices[1:], self.landings_indices[:-1])
        )

    @property
    def flight_times(self) -> tuple[float, ...]:
        """
        Get the times in the mat
        """
        return tuple(
            self.t[landing] - self.t[takeoff] for takeoff, landing in zip(self.takeoffs_indices, self.landings_indices)
        )

    @property
    def displacement_integral(self) -> tuple[float, ...]:
        """
        Get the horizontal displacement integral in the mat
        """
        return tuple(
            _integral(self.t[l:t], self.displacement[l:t])
            for t, l in zip(self.takeoffs_indices[1:], self.landings_indices[0:-1])
        )

    @property
    def displacement_ranges(self) -> tuple[float, ...]:
        """
        Get the horizontal range
        """
        return tuple(
            np.nanmax(self.displacement[l:t]) - np.nanmin(self.displacement[l:t])
            for t, l in zip(self.takeoffs_indices[1:], self.landings_indices[0:-1])
        )

    @property
    def impulses(self) -> tuple[float, ...]:
        """
        Get the horizontal impulses in the mat
        """
        return tuple(
            _integral(self.t[l:t], self.velocity[l:t])
            for t, l in zip(self.takeoffs_indices[1:], self.landings_indices[0:-1])
        )

    @property
    def velocity_ranges(self) -> tuple[float, ...]:
        """
        Get the horizontal range
        """
        return tuple(
            np.nanmax(self.velocity[l:t]) - np.nanmin(self.velocity[l:t])
            for t, l in zip(self.takeoffs_indices[1:], self.landings_indices[0:-1])
        )

    @property
    def acceleration_integral(self) -> tuple[float, ...]:
        """
        Get the horizontal acceleration integral in the mat
        """
        return tuple(
            _integral(self.t[l:t], self.acceleration[l:t])
            for t, l in zip(self.takeoffs_indices[1:], self.landings_indices[0:-1])
        )

    @property
    def acceleration_ranges(self) -> tuple[float, ...]:
        """
        Get the horizontal range
        """
        return tuple(
            np.nanmax(self.acceleration[l:t]) - np.nanmin(self.acceleration[l:t])
            for t, l in zip(self.takeoffs_indices[1:], self.landings_indices[0:-1])
        )

    def plot(
        self,
        override_y: np.ndarray = None,
        **figure_options,
    ) -> plt.figure:
        """
        Plot the data as XY position in an axis('equal') manner

        Parameters
        ----------
        override_y
            Force to plot this y data instead of the self.y attribute
        figure_options
            see _prepare_figure inputs

        Returns
        -------
        The matplotlib figure handler if show_now was set to False
        """

        fig, ax, color, show_now = _prepare_figure(**figure_options)

        ax.plot(self.y[:, 0], self.y[:, 1], color=color)
        ax.axis("equal")

        if show_now:
            plt.show()

        return fig if not show_now else None

    def plot_displacement(self, **figure_options) -> plt.figure:
        """
        Plot the CoP displacement against time

        Parameters
        ----------
        figure_options
            see _prepare_figure inputs

        Returns
        -------
        The matplotlib figure handler if show_now was set to False
        """

        return super().plot(override_y=self.displacement, **figure_options)

    def plot_velocity(self, **figure_options) -> plt.figure:
        """
        Plot the CoP velocity against time

        Parameters
        ----------
        figure_options
            see _prepare_figure inputs

        Returns
        -------
        The matplotlib figure handler if show_now was set to False
        """

        return super().plot(override_y=self.velocity, **figure_options)

    def plot_acceleration(self, **figure_options) -> plt.figure:
        """
        Plot the CoP acceleration against time

        Parameters
        ----------
        figure_options
            see _prepare_figure inputs

        Returns
        -------
        The matplotlib figure handler if show_now was set to False
        """

        return super().plot(override_y=self.acceleration, **figure_options)

    def plot_flight_times(self, factor: float = 1, **figure_options) -> plt.figure:
        """
        Plot the flight times as constant values of the flight period

        Parameters
        ----------
        figure_options
            see _prepare_figure inputs
        factor
            Proportional factor

        Returns
        -------
        The matplotlib figure handler if show_now was set to False
        """
        y = np.nan * np.ndarray(self.displacement.shape)

        for takeoff, landing, flight in zip(self.takeoffs_indices, self.landings_indices, self.flight_times):
            y[takeoff:landing] = flight
        return super().plot(override_y=y * factor, **figure_options)

    def _compute_cop_displacement(self, window: int = 1) -> np.ndarray:
        """
        Compute the CoP displacement
        Parameters
        ----------
        window
            The window to perform the filtering on

        Returns
        -------

        """

        two_windows = window * 2
        padding = np.nan * np.zeros((window, 1))

        return np.concatenate(
            (
                padding,
                np.linalg.norm(self.y[two_windows:, :] - self.y[:-two_windows, :], axis=1)[:, np.newaxis],
                padding,
            )
        )

    def _compute_timings_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the flight time for each flights in the data.
        The flight moments are defined as "nan" in the displacement data.

        Returns
        -------
        The timing indices of the jumps (takeoff and landing)
        """

        # Find all landing and takeoff indices
        currently_in_air = 1 * np.isnan(self.displacement)  # 1 for True, 0 for False
        padding = ((0,),)
        events = np.concatenate((padding, currently_in_air[1:] - currently_in_air[:-1]))
        events[:2] = 0  # Remove any possible artifact from cop_displacement starting
        landings_indices = np.where(events == -1)[0]
        takeoffs_indices = np.where(events == 1)[0]

        # Remove starting and ending artifacts and perform sanity check
        if landings_indices[0] < takeoffs_indices[0]:
            landings_indices = landings_indices[1:]
        if takeoffs_indices[-1] > landings_indices[-1]:
            takeoffs_indices = takeoffs_indices[:-1]
        if len(takeoffs_indices) != len(landings_indices):
            raise RuntimeError(
                f"The number of takeoffs ({len(takeoffs_indices)} is not equal "
                f"to number of landings {len(landings_indices)}"
            )

        return takeoffs_indices, landings_indices
