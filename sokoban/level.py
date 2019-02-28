import os
import copy

class Level:

    matrix = []
    matrix_history = []
    
    def __init__(self,level_set,level_num):
        
        del self.matrix[:]
        del self.matrix_history[:]
        
        # Create level
        #with open(os.path.dirname(os.path.abspath(__file__)) + '/levels/' + set + '/level' + str(level_num), 'r') as f:
        with open('sokoban/levels/' + level_set + '/level' + str(level_num), 'r') as f:
            for row in f.read().splitlines():
                self.matrix.append(list(row))
            
    def __del__(self):
        "Destructor to make sure object shuts down, etc."
        
    def get_matrix(self):
        return self.matrix

    def add_to_history(self,matrix):
        self.matrix_history.append(copy.deepcopy(matrix))

    def get_last_matrix(self):
        if len(self.matrix_history) > 0:
            last_matrix = self.matrix_history.pop()
            self.matrix = last_matrix
            return last_matrix
        else:
            return self.matrix

    def get_player_position(self):
        # Iterate all Rows
        for i in range (0,len(self.matrix)):
            # Iterate all columns
            for k in range (0,len(self.matrix[i])-1):
                if self.matrix[i][k] == "@":
                    return [k,i]

    def get_boxes(self):
        # Iterate all Rows
        boxes = []
        for i in range (0,len(self.matrix)):
            # Iterate all columns
            for k in range (0,len(self.matrix[i])-1):
                if self.matrix[i][k] == "$":
                    boxes.append([k,i])
        return boxes

    def get_size(self):
        max_row_length = 0
        # Iterate all Rows
        for i in range (0,len(self.matrix)):
            # Iterate all columns
            row_length = len(self.matrix[i])
            if row_length > max_row_length:
                max_row_length = row_length
        return [max_row_length,len(self.matrix)]
        
