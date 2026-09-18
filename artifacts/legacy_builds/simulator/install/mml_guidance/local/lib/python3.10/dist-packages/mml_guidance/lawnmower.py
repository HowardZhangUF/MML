#!/usr/bin/env python3
""" Generate a coverage path for a bounding polygon.
"""
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from shapely.prepared import prep


class LawnmowerPath:
    """
    The bathydrone path planner.

    Attributes:
        self.bounding: BoundingRegion object to be pathed.
    """

    def __init__(self, POINTS_PER_SLICE=10):
        self.POINTS_PER_SLICE = POINTS_PER_SLICE

    def generate_path(
        self,
        polygon_points: List[Tuple[float, float]],
        path_dist: float,
        angle: float = 0,
    ) -> Tuple[List[List[float]], float, bool, List[Polygon], Polygon]:
        """
        Generate an optimized path through the polygon.

        Args:
            polygon_points (List[Tuple[float, float]]): List of vertices of the polygon.
            path_dist (float): Distance between waypoints.

        Returns:
            Tuple[List[List[float]], float, bool, List[Polygon], Polygon]:
                - chosenPath: List of path points
                - Geom: Polygon geometry used for path generation
        """
        # Rotate polygon
        transform = [
            [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))],
            [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))],
        ]
        print("polygon_points", polygon_points)
        rotated_points = [list(np.dot(transform, point)) for point in polygon_points]

        # Generate grid and path
        geom = Polygon(rotated_points)
        print("geom", geom)
        grid = self.partition(geom, path_dist)

        path = self._generate_path_for_grid(grid)

        # Rotate path back
        inv_transform = [
            [np.cos(np.deg2rad(-angle)), -np.sin(np.deg2rad(-angle))],
            [np.sin(np.deg2rad(-angle)), np.cos(np.deg2rad(-angle))],
        ]
        path = np.dot(inv_transform, path.T).T

        return path, geom

    def _generate_path_for_grid(
        self,
        grid: List[Polygon],
    ) -> Tuple[List[List[float]], float, int]:
        """
        Generate a path through the grid cells.

        Args:
            grid (List[Polygon]): List of grid cells.

        Returns:
            Tuple[List[List[float]], float, int]:
                - path: List of path points
                - path_length: Total length of the path
                - num_turns: Number of turns in the path
        """
        path = []
        num_turns = 0
        direction = 1  # 1 = up, -1 = down

        # Get grid boundaries
        x_coords = [cell.bounds[0] for cell in grid]
        min_x = min(x_coords)
        max_x = max(x_coords)
        cell_width = abs(grid[0].bounds[2] - grid[0].bounds[0])

        curr_x = min_x
        while curr_x <= max_x:
            # Get cells in current slice
            slice_cells = [
                cell for cell in grid if abs(cell.bounds[0] - curr_x) < 1e-10
            ]

            if slice_cells:
                num_turns += 1
                start_end = self._get_slice_path(slice_cells, direction)
                interpolation = np.linspace(
                    start_end[0], start_end[1], self.POINTS_PER_SLICE
                )
                for point in interpolation:
                    path.append(point)
                direction *= -1

            curr_x += cell_width

        return np.array(path)

    @staticmethod
    def _get_slice_path(cells: List[Polygon], direction: int) -> List[List[float]]:
        """
        Generate path points for a vertical slice of cells.

        Args:
            cells (List[Polygon]): List of grid cells in the slice.
            direction (int): Direction of the path (1 = up, -1 = down).

        Returns:
            List[List[float]]: List of path points for the slice.
        """
        if direction == 1:
            start_cell = cells[0]
            end_cell = cells[-1]
        else:
            start_cell = cells[-1]
            end_cell = cells[0]

        return [
            [
                (start_cell.bounds[0] + start_cell.bounds[2]) / 2,
                (start_cell.bounds[1] + start_cell.bounds[3]) / 2,
            ],
            [
                (end_cell.bounds[0] + end_cell.bounds[2]) / 2,
                (end_cell.bounds[1] + end_cell.bounds[3]) / 2,
            ],
        ]

    @staticmethod
    def grid_bounds(geom, delta):
        """
        Input: geom (shapely polygon), delta (distance between path lines)
        Output: grid boundaries from which to draw path.
        """

        minx, miny, maxx, maxy = geom.bounds
        nx = int((maxx - minx) / delta)
        ny = int((maxy - miny) / delta)
        gx, gy = np.linspace(minx, maxx, nx), np.linspace(miny, maxy, ny)

        grid = []
        for i in range(len(gx) - 1):
            for j in range(len(gy) - 1):
                poly_ij = Polygon(
                    [
                        [gx[i], gy[j]],
                        [gx[i], gy[j + 1]],
                        [gx[i + 1], gy[j + 1]],
                        [gx[i + 1], gy[j]],
                    ]
                )
                grid.append(poly_ij)
        return grid

    def partition(self, geom, delta):
        """Define a grid of cells for a polygon."""
        prepared_geom = prep(geom)
        grid = list(filter(prepared_geom.covers, self.grid_bounds(geom, delta)))
        return grid

    @staticmethod
    def plot(bounds, path):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        # include the first bound at the end to close the loop
        bounds.append(bounds[0])
        xList, yList = list(zip(*bounds))
        ax1.plot(xList, yList, c="red", marker="o", markersize="0.25")
        xList, yList = list(zip(*path))
        ax1.plot(xList, yList, c="green", marker="o", markersize="0.5", zorder=1)
        ax1.scatter(xList, yList, c="green")
        plt.axis("equal")
        plt.show()


def main():
    """
    Example usage of the PathPlanner class.

    1. Create a PathPlanner object.
    2. Set the bounding polygon from csv.
    3. Generate and display a coverage path for the region.
    """
    bounds = [
        (-1.7, -1),
        (0.9, -1),
        (0.9, 2),
        (-1.7, 2),
    ]
    my_planner = LawnmowerPath(POINTS_PER_SLICE=5)
    best_path, _ = my_planner.generate_path(bounds, path_dist=0.4, angle=0)
    my_planner.plot(bounds, best_path)


if __name__ == "__main__":
    main()
