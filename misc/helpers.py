import numpy as np


def derivative(t: np.ndarray, data: np.ndarray, window: int = 1) -> np.ndarray:
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


def integral(t: np.ndarray, data: np.ndarray) -> float:
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
