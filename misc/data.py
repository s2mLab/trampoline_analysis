from matplotlib import pyplot as plt
import numpy as np


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

        fig, ax, color, show_now = self._prepare_figure(**figure_options)

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

    @staticmethod
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