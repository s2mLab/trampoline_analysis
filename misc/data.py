import numpy as np


class Data:
    def __init__(self, nb_sensors, conversion_factor: float = 1):
        """
        Create a Data structure with 't' as time vector holder and 'y' as data holder
        Parameters
        ----------
        nb_sensors
            The number of sensors (column) in the 'y' data holder
        conversion_factor
            The factor to convert the data when using the 'append' method
        """

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

        self.t = np.concatenate((self.t, (t, )))

        # Remove the MAX_INT and convert to m
        y = [data * self.conversion_factor if data != 2147483647 else np.nan for data in y]
        self.y = np.concatenate((self.y, (y, )))

    def flight_times(self) -> tuple[float, ...]:
        """
        Get the flight time for each flights in the data. The flight are defined as "nan" in one of the CoP. Data
        are assumed to be of XY

        Returns
        -------
        The flight times
        """

        # Make the XY data a single value, if any of them is nan, keep it as is
        y = np.sum(self.y, axis=1)

        # Find all landing and takeoff indices
        currently_in_air = False
        landings = []
        takeoffs = []
        for time_idx, temp in enumerate(y):
            if np.isnan(temp) and not currently_in_air:
                currently_in_air = True
                if time_idx != 0:
                    takeoffs.append(time_idx)
            elif not np.isnan(temp) and currently_in_air:
                currently_in_air = False
                if time_idx != 0:
                    landings.append(time_idx)

        # Remove starting and ending artifacts and perform sanity check
        if landings[0] < takeoffs[0]:
            landings = landings[1:]
        if takeoffs[-1] > landings[-1]:
            takeoffs = takeoffs[:-1]
        if len(takeoffs) != len(landings):
            raise RuntimeError(
                f"The number of takeoffs ({len(takeoffs)} is not equal to number of landings {len(landings)}"
            )

        # Compute time of flight for each flight
        return tuple(self.t[landing] - self.t[takeoff] for takeoff, landing in zip(takeoffs, landings))
