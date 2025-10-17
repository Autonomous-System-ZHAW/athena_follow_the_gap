import rclpy
from rclpy.node import Node
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time


class AnalyseMovement:
    """
    This class collects waypoints during movement and provides
    real-time visualization of their positions. It allows
    saving waypoints, updating a live plot, and transforming
    polar waypoints into Cartesian coordinates.
    """

    def __init__(self):
        self.waypoint_list = []
        self.counter = 0
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.scatter = None

    def safe_way_point(self, x_origin, y_origin):
        """
        Save a new waypoint and update the live plot.

        Args:
            x_origin (float): X-coordinate of the origin offset.
            y_origin (float): Y-coordinate of the origin offset.
        """

        # x, y = self.calculate_x_y(waypoint, x_origin, y_origin)
        self.waypoint_list.append((x_origin, y_origin))
        self.update_plot()

    def update_plot(self):
        """
        Update the real-time plot with the stored waypoints.

        Clears the current plot and redraws all waypoints
        with axes, grid, and markers.
        """

        x_vals = [wp[0] for wp in self.waypoint_list]
        y_vals = [wp[1] for wp in self.waypoint_list]

        self.ax.clear()
        self.ax.set_title("Wegpunkte in Echtzeit")
        self.ax.set_xlim(-2.5, 25)
        self.ax.set_ylim(2.5, -15)
        self.ax.grid(True)
        self.ax.plot(x_vals, y_vals, marker="o")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def calculate_x_y(self, waypoint, x_origin, y_origin):
        """
        Convert polar waypoint coordinates into Cartesian coordinates.

        Args:
            waypoint (tuple[float, float]): Waypoint data (angle [deg], distance).
            x_origin (float): X-coordinate of the origin offset.
            y_origin (float): Y-coordinate of the origin offset.

        Returns:
            tuple[float, float]: Converted Cartesian coordinates (x, y).
        """

        x = (waypoint[1] * np.cos(np.deg2rad(waypoint[0]))) + x_origin
        y = (waypoint[1] * np.sin(np.deg2rad(waypoint[0]))) + y_origin

        return x, y
