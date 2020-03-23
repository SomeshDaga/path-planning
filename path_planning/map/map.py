import numpy as np

class Map:
    FREE = 255
    OCCUPIED = 0

    def __init__(self, data, resolution=0.1, d_theta=np.pi/8):
        self._data = np.array(data).transpose()
        self._resolution = resolution
        self._d_theta = d_theta
        self._width = self._data.shape[0]   # x
        self._height = self._data.shape[1]  # y

    def get_width(self):
        return

    def get_data(self):
        return self._data.copy()

    def get_image(self):
        return self._data.transpose().copy()

    def get_resolution(self):
        return self._resolution

    def get_delta_theta(self):
        return self._d_theta

    def get_obstacle_boundaries(self):
        data = np.vstack((np.zeros((1, self._width)), self._data))
        data = np.hstack((np.zeros((self._height + 1, 1)), data))
        print(data.shape)
        grad_x = np.diff(data, axis=1)
        grad_y = np.diff(data, axis=0)

        boundary_cells = np.empty((0, 2), dtype=int)

        # Indexes of occupied to free boundary cells
        for axis in (0, 1):
            grad = np.diff(self._data, axis=axis)
            obs_to_free_idxs = np.argwhere(grad == Map.OCCUPIED - Map.FREE)
            free_to_obs_idxs = np.argwhere(grad == Map.FREE - Map.OCCUPIED)
            # For the free to obstacle transitions, increment index by 1
            obs_to_free_idxs[:, axis] += 1
            boundary_cells = np.vstack((boundary_cells, obs_to_free_idxs, free_to_obs_idxs))


        return np.unique(boundary_cells, axis=0)

    def worldToMap(self, pos):
        return np.floor(pos / self._resolution).astype(int)

    def mapToWorld(self, pos):
        return (self._resolution / 2) + self._resolution * pos

    def is_occupied(self, pos, world_coords=True, check_bounds=False):
        if world_coords:
            pos = self.worldToMap(pos).astype(int)
        else:
            pos = pos.astype(int)

        if not check_bounds or self.in_bounds(pos, False):
            return self._data[pos[0], pos[1]] == Map.OCCUPIED
        else:
            return True

    def in_bounds(self, pos, world_coords=True):
        if world_coords:
            pos = self.worldToMap(pos).astype(int)
        else:
            pos = pos.astype(int)

        return self._width > pos[0] >= 0 and self._height > pos[1] >= 0

    def enforce_bounds(self, pos):
        """

        :param pos: Position in map coordinates
        :return:
        """
        if pos[0] < 0:
            pos[0] = 0
        elif pos[0] >= self._width:
            pos[0] = self._width - 1

        if pos[1] < 0:
            pos[1] = 0
        elif pos[1] >= self._height:
            pos[1] = self._height - 1

        return pos

    def find_obstacle(self, pos, angle, max_dist):
        # Rasterize the line using Bresenham's algorithm
        # Use DDA to cover all cells that the ray passes through (modified Bresenham)
        cur_cell = self.worldToMap(pos)
        pos = self.mapToWorld(cur_cell)
        end = np.array([pos[0] + max_dist * np.cos(angle),
                        pos[1] + max_dist * np.sin(angle)])
        end_cell = self.worldToMap(end)

        # Cells that the ray traverses through
        cells = np.empty((0, 2), dtype=int)

        x, y = cur_cell[0], cur_cell[1]
        x2, y2 = end_cell[0], end_cell[1]
        dx = abs(x2 - x)
        dy = abs(y2 - y)
        step_x = 1 if x2 - x > 0 else -1
        step_y = 1 if y2 - y > 0 else -1
        steep = True if dy > dx else False

        if steep:
            x, y = y, x
            dx, dy = dy, dx
            step_x, step_y = step_y, step_x

        prev_error = -dx
        error = (2 * dy) - dx
        for i in range(0, dx):
            cells_to_add = []
            x += step_x

            if steep:
                x_dist = self._height - 1 - x if step_x > 0 else x
                y_dist = self._width - 1 - y if step_y > 0 else y
            else:
                x_dist = self._width - 1 - x if step_x > 0 else x
                y_dist = self._height - 1 - y if step_y > 0 else y

            if error >= 0:
                y = y + step_y
                y_dist -= 1
                # Check if we went through the lower cell along y while climbing
                if prev_error + error < 0 and x_dist >= 0:
                    if steep:
                        cells_to_add.append(np.array([y - step_y, x]))
                    else:
                        cells_to_add.append(np.array([x, y - step_y]))
                # Check if we went through the left cell along x while climbing
                elif prev_error + error > 0 and y_dist >= 0:
                    if steep:
                        cells_to_add.append(np.array([y, x - step_x]))
                    else:
                        cells_to_add.append(np.array([x - step_x, y]))
                error -= (2 * dx)

            if steep:
                in_bounds = self.in_bounds(np.array([y, x]), world_coords=False)
                if in_bounds:
                    cells_to_add.append(np.array([y, x]))
            else:
                in_bounds = self.in_bounds(np.array([x, y]), world_coords=False)
                if in_bounds:
                    cells_to_add.append(np.array([x, y]))

            # Among the cells to add, check if we hit an obstacle
            # We should check the cells in the order they were added, since the
            # they are stored in order of increasing distance
            for cell in cells_to_add:
                if self.is_occupied(cell, world_coords=False, check_bounds=False):
                    return self.mapToWorld(cell)

            if not in_bounds:
                return None

            prev_error = error
            error += (2 * dy)

        return None

    def get_obstacles_in_radius(self, pos, radius):
        obstacles = []
        distances = []

        angles = np.arange(0, 2*np.pi + self._d_theta, self._d_theta)
        for angle in angles:
            obstacle = self.find_obstacle(pos, angle, radius)
            if obstacle is not None:
                obstacles.append(obstacle)
                distances.append(np.linalg.norm(obstacle - pos))

        return obstacles, distances
