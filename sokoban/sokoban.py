import inspect

from .level import Level
from sprite import Sprite

class CannotGo(Exception):
    pass

class Sokoban:

    def __init__(self, level=1, level_set='our', trace=True):
        wall = 'sokoban/images/wall.png'
        box = 'sokoban/images/box.png'
        box_on_target = 'sokoban/images/box_on_target.png'
        space = 'sokoban/images/space.png'
        target = 'sokoban/images/target.png'
        player = 'sokoban/images/player.png'
        # player = 'sokoban/images/beetle-robot.png'
        self._images = {'#': wall, ' ': space, '$': box, '.': target, '@': player, '*': box_on_target}
        self._tiles = None
        self._do_trace = trace
        self.load_level(level, level_set)

    def load_level(self, level=None, level_set=None, clear_log=True):
        if not level:
            level = self._level_number
        if not level_set:
            level_set = self._level_set
        self._level = Level(level_set, level)
        if clear_log or self._level_number != level or self._level_set != level_set:
            self._log = []
            self._log_pos = 0
        if self._tiles and (self._level_number != level or self._level_set != level_set):
            self._tiles = None
        self._level_set = level_set
        self._level_number = level
        self._target_found = False
        self._player_pos = self._level.get_player_position()

    def replay(self, pos=None):
        if pos is None:
            pos = self._log_pos + 1
        if pos < self._log_pos:
            start = 0
            self.load_level(clear_log=False)
        else:
            start = self._log_pos
        # if pos > len(self._log): return False
        trace = None
        self._log_pos = start
        for dx, dy, steps, trace in self._log[start:pos]:
            self.move_player(dx, dy, steps, nolog=True)
        print(2222, start, pos, self._log_pos)
        return trace

    @property
    def log(self):
        return self._log

    @property
    def level(self):
        return self._level_number

    @level.setter
    def level(self, level):
        self.load_level(level)

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

    def move_player(self, dx, dy, steps=1, nolog=False):
        log = self._log
        if not nolog:
            trace = []
            if self._do_trace:
                f = inspect.currentframe()
                tb = []
                depth = 0
                while f is not None:
                    filename = f.f_code.co_filename
                    lineno = f.f_lineno
                    if depth > 4 and filename != '<code-input>':
                        break
                    # print('depth', depth, filename, lineno)
                    if filename == '<code-input>':
                        trace.append(lineno)
                    depth += 1
                    f = f.f_back
            log.append((dx, dy, steps, trace))

        level = self._level
        target_found = self._target_found
        matrix = level.get_matrix()
        # for row in reversed(matrix):
            # print(''.join(row))

        level.add_to_history(matrix)

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
                    if not nolog:
                        log.pop()
                    raise CannotGo
            else:
                if not nolog:
                    log.pop()
                raise CannotGo

            if target_found is True:
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

        self._log_pos += 1
        # for row in reversed(matrix):
        #     print(''.join(row))
        # print("Boxes remaining: " + str(len(level.get_boxes())))
        # return len(level.get_boxes()) == 0
        # if len(level.get_boxes()) == 0:
            # print("Level Completed")
            # self.load_level(self._level_number + 1)
            # self.sandbox.clear_widgets()
            # self.draw_level()

    @property
    def boxes_remaining(self):
        return len(self._level.get_boxes())
