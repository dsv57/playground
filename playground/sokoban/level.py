from copy import deepcopy


class Level:
    def __init__(self, level_set, level_num):
        self.matrix = []
        self.matrix_history = []

        # Create level
        with open(f"sokoban/levels/{level_set}/level{level_num}", "r") as f:
            for row in f.read().splitlines():
                self.matrix.append(list(row))

        self.matrix.reverse()

    def get_matrix(self):
        return self.matrix

    def add_to_history(self, matrix):
        self.matrix_history.append(deepcopy(matrix))

    def get_last_matrix(self):
        if len(self.matrix_history) > 0:
            last_matrix = self.matrix_history.pop()
            self.matrix = last_matrix
            return last_matrix
        else:
            return self.matrix

    def get_player_position(self):
        # Iterate all Rows
        for i, row in enumerate(self.matrix):
            # Iterate all columns
            for j, c in enumerate(row):
                if row[j] == "@":
                    return [j, i]

    def get_boxes(self):
        # Iterate all Rows
        boxes = []
        for i, row in enumerate(self.matrix):
            # Iterate all columns
            for j, c in enumerate(row):
                if row[j] == "$":
                    boxes.append([j, i])
        return boxes

    def get_size(self):
        return (max([len(row) for row in self.matrix]), len(self.matrix))
