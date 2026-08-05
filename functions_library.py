#########################################
#
# Functions library 
# Author: M Joyce
#
#########################################
import numpy as np


def find_nearest(value, array):
    index = np.abs(array - value).argmin()
    return index


def find_left_nearest(value, array):
    valid = np.where(array <= value)[0]
    if len(valid) == 0:
        raise ValueError("No array value lies to the left of the target.")
    return valid[-1]


def find_right_nearest(value, array):
    valid = np.where(array >= value)[0]
    if len(valid) == 0:
        raise ValueError("No array value lies to the right of the target.")
    return valid[0]


def which_color(target_age):
    if target_age==5:
        color='blue'
    elif target_age == 10:
        color = 'green'
    elif target_age == 15:
        color='goldenrod'
    else:
        color='purple'
    return color


def find_interpolated_nearest(value, array, number_of_points=1000):
    left_index = find_left_nearest(value, array)
    right_index = find_right_nearest(value, array)

    interpolated_array = np.linspace(
        array[left_index],
        array[right_index],
        number_of_points
    )

    nearest_index = find_nearest(value, interpolated_array)

    return interpolated_array[nearest_index]