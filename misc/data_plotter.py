from matplotlib import pyplot as plt

from .data import Data


class DataPlotter:
    @staticmethod
    def plot_t(
        data: Data,
        figure_name: str | None = None,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        color: str | None = None,
        show_now: bool = False,
    ) -> plt.figure:
        """
        Interpret the data set COP position (first column being X and second being Y)

        Parameters
        ----------
        data
            The data to plot the data from
        figure_name
            The name of the figure. If two figures has the same name, they are drawn on the same graph
        title
            The title of the figure
        x_label
            The name of the x-axis
        y_label
            The name of the y-axis
        color
            The color of the plot
        show_now
            If the method should call plot.show() by itself

        Returns
        -------
        The matplotlib figure handler if show_now was set to False
        """

        fig, ax = DataPlotter._prepare_figure(figure_name=figure_name, title=title, x_label=x_label, y_label=y_label)

        ax.plot(data.t, data.y, color=color)

        if show_now:
            plt.show()

        return fig if not show_now else None

    @staticmethod
    def plot_xy(
        data: Data,
        figure_name: str|None = None,
        title: str|None = None,
        x_label: str|None = None,
        y_label: str|None = None,
        color: str|None = None,
        show_now: bool = False,
    ) -> plt.figure:
        """
        Interpret the data set COP position (first column being X and second being Y)

        Parameters
        ----------
        data
            The data to plot the data from
        figure_name
            The name of the figure. If two figures has the same name, they are drawn on the same graph
        title
            The title of the figure
        x_label
            The name of the x-axis
        y_label
            The name of the y-axis
        color
            The color of the plot
        show_now
            If the method should call plot.show() by itself

        Returns
        -------
        The matplotlib figure handler if show_now was set to False
        """

        fig, ax = DataPlotter._prepare_figure(figure_name=figure_name, title=title, x_label=x_label, y_label=y_label)

        ax.plot(data.y[:, 0], data.y[:, 1], color=color)
        ax.axis("equal")

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
        figure_name: str|None,
        title: str|None,
        x_label: str|None,
        y_label: str|None,
    ) -> tuple[plt.figure, plt.axis]:
        """

        Parameters
        ----------
        figure_name
            The name of the figure. If two figures has the same name, they are drawn on the same graph
        title
            The title of the figure
        x_label
            The name of the x-axis
        y_label
            The name of the y-axis

        Returns
        -------
        The figure and the axis handler
        """

        fig = plt.figure(figure_name)
        ax = plt.axes() if fig.axes == [] else fig.axes[-1]

        if title is not None:
            ax.set_title(title)
        if x_label is not None:
            ax.set_xlabel(x_label)
        if y_label is not None:
            ax.set_ylabel(y_label)
        return fig, ax
