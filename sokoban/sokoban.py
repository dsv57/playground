from .level import Level
from sprite import Sprite

class CannotGo(Exception):
    pass

class Sokoban:

    def __init__(self, level=1, level_set='our'):
        wall = 'sokoban/images/wall.png'
        box = 'sokoban/images/box.png'
        box_on_target = 'sokoban/images/box_on_target.png'
        space = 'sokoban/images/space.png'
        target = 'sokoban/images/target.png'
        player = 'sokoban/images/player.png'
        # player = 'sokoban/images/beetle-robot.png'
        self._images = {'#': wall, ' ': space, '$': box, '.': target, '@': player, '*': box_on_target}
        self._tiles = None
        self.load_level(level, level_set)

    def load_level(self, level=1, level_set=None):
        if not level_set:
            level_set = self._level_set
        self._level = Level(level_set, level)
        self._level_set = level_set
        self._level_number = level
        self._target_found = False
        self._player_pos = self._level.get_player_position()
        if self._tiles and self._level_number != level or self._level_set != level_set:
            self._tiles = None

    def draw_level(self, layout):
        update = self._tiles is not None
        images = self._images
        matrix = self._level.get_matrix()
        if not update:
            self._tiles = []
        for i, row in enumerate(matrix):
            if not update:
                self._tiles.append([])
            for j, c in enumerate(row):
                if update:
                    tile = self._tiles[i][j]
                    image = tile.shapes[0]
                    if image.source != images[c]:
                        image.source = images[c]
                        print('Update:', image.source, images[c])
                else:
                    tile = Sprite(images[c], x=36*j, y=36*i, trace=False)
                    self._tiles[-1].append(tile)
                    layout.add_widget(tile)

    def move_player(self, dx, dy, steps=1):
        my_level = self._level
        target_found = self._target_found
        matrix = my_level.get_matrix()
        # for row in reversed(matrix):
            # print(''.join(row))

        my_level.add_to_history(matrix)

        if steps < 0:
            dx = -dx
            dy = -dy
            steps = -steps

        while steps > 0:
            x, y = self._player_pos

            orig_tile = matrix[y][x]
            next_tile = matrix[y + dy][x + dx]
            second_tile = matrix[y + 2 * dy][x + 2 * dx]
            if next_tile in ' .':
                pass
            elif next_tile in '$*':
                if second_tile == ' ':
                    second_tile = '$'
                elif second_tile == '.':
                    second_tile = '*'
                else:
                    raise CannotGo
            else:
                raise CannotGo

            if target_found == True:
                orig_tile = '.'
                target_found = False
            else:
                orig_tile = ' '                    
            if next_tile in '.*':
                target_found = True
            next_tile = '@'
            self._player_pos = (x + dx, y + dy)

            self._target_found = target_found
            matrix[y][x] = orig_tile
            matrix[y + dy][x + dx] = next_tile
            matrix[y + 2 * dy][x + 2 * dx] = second_tile
            # print('STEP', dx, dy, steps)
            steps -= 1

        # print("Boxes remaining: " + str(len(my_level.get_boxes())))

        if len(my_level.get_boxes()) == 0:
            print("Level Completed")
            self.load_level(self._level_number + 1)
            # self.sandbox.clear_widgets()
            # self.draw_level()
