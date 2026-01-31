"""
Improved bot for AWAP 2026 Carnegie Cookoff.
Optimized for speed and handles all edge cases.
"""

from collections import deque
from typing import Tuple, Optional, Set, List

from game_constants import Team, FoodType, ShopCosts
from robot_controller import RobotController
from item import Pan, Plate, Food


class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        
        # Cached locations (set once)
        self.counters = None
        self.cookers = None
        self.shop_loc = None
        self.submit_loc = None
        self.trash_loc = None
        
        # State
        self.state = 0
        self.sub_state = 0
        self.ingredients_needed = []
        self.plate_counter = None
        self.work_counter = None
        
        # Stuck detection
        self.stuck_counter = 0
        self.last_state = -1

    def _cache_locations(self, controller):
        """Cache all locations once at start."""
        if self.counters is not None:
            return
            
        m = controller.get_map(controller.get_team())
        self.counters = []
        self.cookers = []
        
        for x in range(m.width):
            for y in range(m.height):
                name = m.tiles[x][y].tile_name
                if name == "COUNTER":
                    self.counters.append((x, y))
                elif name == "COOKER":
                    self.cookers.append((x, y))
                elif name == "SHOP" and self.shop_loc is None:
                    self.shop_loc = (x, y)
                elif name == "SUBMIT" and self.submit_loc is None:
                    self.submit_loc = (x, y)
                elif name == "TRASH" and self.trash_loc is None:
                    self.trash_loc = (x, y)

    def _bfs_step(self, controller, start, target_x, target_y, blocked=None):
        """Get next step towards target. Returns (dx, dy) or None."""
        if blocked is None:
            blocked = set()
            
        bx, by = start
        # Already adjacent?
        if max(abs(bx - target_x), abs(by - target_y)) <= 1:
            return (0, 0)
        
        m = controller.get_map(controller.get_team())
        queue = deque([(bx, by, None)])
        visited = {(bx, by)}
        
        while queue:
            cx, cy, first_step = queue.popleft()
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) in visited or (nx, ny) in blocked:
                        continue
                    if not (0 <= nx < m.width and 0 <= ny < m.height):
                        continue
                    if not m.is_tile_walkable(nx, ny):
                        continue
                    
                    visited.add((nx, ny))
                    step = first_step if first_step else (dx, dy)
                    
                    if max(abs(nx - target_x), abs(ny - target_y)) <= 1:
                        return step
                    
                    queue.append((nx, ny, step))
        return None

    def _move_to(self, controller, bot_id, target_x, target_y):
        """Move towards target. Returns True if adjacent."""
        state = controller.get_bot_state(bot_id)
        bx, by = state['x'], state['y']
        
        if max(abs(bx - target_x), abs(by - target_y)) <= 1:
            return True
        
        step = self._bfs_step(controller, (bx, by), target_x, target_y)
        if step and step != (0, 0):
            controller.move(bot_id, step[0], step[1])
        return False

    def _get_free_counter(self, controller, exclude=None, bot_pos=None):
        """Find empty counter that is reachable."""
        if exclude is None:
            exclude = set()
        
        # Sort counters by distance to bot if bot_pos provided
        counters = self.counters
        if bot_pos:
            counters = sorted(counters, key=lambda c: max(abs(c[0]-bot_pos[0]), abs(c[1]-bot_pos[1])))
        
        # First try to find an empty reachable counter
        for cx, cy in counters:
            if (cx, cy) in exclude:
                continue
            tile = controller.get_tile(controller.get_team(), cx, cy)
            if tile and tile.item is None:
                # Check if reachable (can find path)
                if bot_pos:
                    step = self._bfs_step(controller, bot_pos, cx, cy)
                    if step is not None:
                        return (cx, cy)
                else:
                    return (cx, cy)
        
        # Fallback: any counter not in exclude, preferring reachable ones
        for cx, cy in counters:
            if (cx, cy) not in exclude:
                if bot_pos:
                    step = self._bfs_step(controller, bot_pos, cx, cy)
                    if step is not None:
                        return (cx, cy)
                else:
                    return (cx, cy)
        
        # Last resort: just return first counter
        return self.counters[0] if self.counters else None

    def _get_free_cooker(self, controller):
        """Find cooker with empty pan."""
        for kx, ky in self.cookers:
            tile = controller.get_tile(controller.get_team(), kx, ky)
            if tile and isinstance(tile.item, Pan) and tile.item.food is None:
                return (kx, ky)
        return None

    def _get_cooker_needing_pan(self, controller):
        """Find cooker without pan."""
        for kx, ky in self.cookers:
            tile = controller.get_tile(controller.get_team(), kx, ky)
            if tile and not isinstance(tile.item, Pan):
                return (kx, ky)
        return None

    def _needs_cooking(self, ing):
        return ing in ('EGG', 'MEAT')
    
    def _needs_chopping(self, ing):
        return ing in ('MEAT', 'ONIONS')
    
    def _food_type(self, name):
        return {'EGG': FoodType.EGG, 'MEAT': FoodType.MEAT, 'NOODLES': FoodType.NOODLES, 
                'ONIONS': FoodType.ONIONS, 'SAUCE': FoodType.SAUCE}.get(name)

    def _find_existing_ingredient(self, controller, ingredient):
        """Find partial matching ingredient on map."""
        # Check counters for chopped/raw version
        for cx, cy in self.counters:
            tile = controller.get_tile(controller.get_team(), cx, cy)
            if tile and isinstance(tile.item, Food) and tile.item.food_name == ingredient:
                return (cx, cy), False # False = not cooking

        # Check cookers for cooking version
        for kx, ky in self.cookers:
            tile = controller.get_tile(controller.get_team(), kx, ky)
            if tile and isinstance(tile.item, Pan) and tile.item.food and tile.item.food.food_name == ingredient:
                return (kx, ky), True # True = is cooking
        return None, False

    def play_turn(self, controller: RobotController):
        bots = controller.get_team_bot_ids(controller.get_team())
        if not bots:
            return
        
        # Debug: print all bot positions
        turn = controller.get_turn()
        team = controller.get_team().name
        positions = []
        for bid in bots:
            info = controller.get_bot_state(bid)
            positions.append(f"Bot{bid}:({info['x']},{info['y']})")
        print(f"[{team} Turn {turn}] {' | '.join(positions)} | State={self.state} SubState={self.sub_state}")
        
        self._cache_locations(controller)
        if not self.shop_loc or not self.submit_loc:
            return
        
        bot_id = bots[0]
        self._run_bot(controller, bot_id)

    def _run_bot(self, controller, bot_id):
        info = controller.get_bot_state(bot_id)
        holding = info.get('holding')
        
        sx, sy = self.shop_loc
        ux, uy = self.submit_loc
        
        # Stuck detection: move random if stuck instead of reset
        if self.state == self.last_state:
            self.stuck_counter += 1
            if self.stuck_counter > 10:
                print(f"[Bot] STUCK in state {self.state}, forcing random move")
                import random
                dx, dy = random.choice([(0,1), (0,-1), (1,0), (-1,0)])
                controller.move(bot_id, dx, dy)
                self.stuck_counter = 0
                return
        else:
            self.stuck_counter = 0
        self.last_state = self.state
        
        # Burnt food check
        if holding and holding.get('type') == 'Food' and holding.get('cooked_stage') == 2:
            self.state = 99
        
        # State 0: Pick order
        if self.state == 0:
            self.ingredients_needed = []
            self.plate_counter = None
            self.work_counter = None
            self.sub_state = 0
            
            orders = controller.get_orders(controller.get_team())
            turn = controller.get_turn()
            
            best = None
            best_score = 999
            for o in orders:
                if o['is_active'] and o['expires_turn'] > turn:
                    score = len(o['required'])
                    # Prioritize expensive/complex orders if time permits
                    if score < best_score:
                        best_score = score
                        best = o
            
            if best:
                req = list(best['required'])
                # Non-cooking first
                self.ingredients_needed = [i for i in req if not self._needs_cooking(i)] + \
                                          [i for i in req if self._needs_cooking(i)]
                self.state = 1
        
        # State 1: Ensure pan
        elif self.state == 1:
            if any(self._needs_cooking(i) for i in self.ingredients_needed):
                if self._get_free_cooker(controller):
                    self.state = 2
                else:
                    cooker = self._get_cooker_needing_pan(controller)
                    if cooker:
                        kx, ky = cooker
                        if holding:
                            if holding.get('type') == 'Pan':
                                if self._move_to(controller, bot_id, kx, ky):
                                    controller.place(bot_id, kx, ky)
                                    self.state = 2
                            else:
                                self.state = 99
                        else:
                            if self._move_to(controller, bot_id, sx, sy):
                                if controller.get_team_money(controller.get_team()) >= ShopCosts.PAN.buy_cost:
                                    controller.buy(bot_id, ShopCosts.PAN, sx, sy)
                    else:
                        self.state = 2
            else:
                self.state = 2
        
        # State 2: Buy/place plate
        elif self.state == 2:
            if holding:
                if holding.get('type') == 'Plate':
                    if not self.plate_counter:
                        bx, by = info['x'], info['y']
                        self.plate_counter = self._get_free_counter(controller, bot_pos=(bx, by))
                    
                    if self.plate_counter:
                        px, py = self.plate_counter
                        adjacent = self._move_to(controller, bot_id, px, py)
                        if adjacent:
                            if controller.place(bot_id, px, py):
                                self.state = 10
                    else:
                        print(f"Error: No plate counter available")
                else:
                    self.state = 99
            else:
                adjacent = self._move_to(controller, bot_id, sx, sy)
                money = controller.get_team_money(controller.get_team())
                if adjacent:
                    # Check if we already have a plate somewhere? Nah, simple is fine
                    if money >= ShopCosts.PLATE.buy_cost:
                        controller.buy(bot_id, ShopCosts.PLATE, sx, sy)
        
        # State 10: Next ingredient
        elif self.state == 10:
            if not self.ingredients_needed:
                self.state = 20
            else:
                self.sub_state = 0
                self.work_counter = None
                self.state = 11
        
        # State 11: Buy ingredient
        elif self.state == 11:
            if not self.ingredients_needed:
                self.state = 20
                return
            
            ing = self.ingredients_needed[0]
            ft = self._food_type(ing)
            
            if holding:
                if holding.get('type') == 'Food':
                    # verify it's the right food
                    if holding.get('food_name') == ing:
                         self.state = 12
                    else:
                         # Wrong food? trash it
                         self.state = 99
                else:
                    self.state = 99
            else:
                # CHECK IF WE ALREADY HAVE IT ON MAP
                loc, is_cooking = self._find_existing_ingredient(controller, ing)
                if loc:
                     if is_cooking:
                         # Cooking in execution
                         if self._needs_chopping(ing):
                             self.sub_state = 4 # Wait code for chop_cook
                         else:
                             self.sub_state = 1 # Wait code for cook_only
                     else:
                         self.work_counter = loc
                         
                         # Check status
                         tile = controller.get_tile(controller.get_team(), *loc)
                         if tile and isinstance(tile.item, Food):
                             is_chopped = tile.item.chopped
                             
                             if self._needs_chopping(ing):
                                 if is_chopped:
                                     self.sub_state = 2 # Pickup chopped
                                 else:
                                     self.sub_state = 1 # Chop
                             else:
                                 # Need to pick up raw egg/noodles
                                 # _cook_only needs to handle this
                                 self.sub_state = 0 
                                 
                     self.state = 12
                     return

                # Buy only if not found
                if self._move_to(controller, bot_id, sx, sy):
                    cost = ft.buy_cost
                    if controller.get_team_money(controller.get_team()) >= cost:
                        controller.buy(bot_id, ft, sx, sy)
                        self.state = 12
        
        # State 12: Process
        elif self.state == 12:
            if not self.ingredients_needed:
                self.state = 20
                return
            
            ing = self.ingredients_needed[0]
            chop = self._needs_chopping(ing)
            cook = self._needs_cooking(ing)
            
            if chop and cook:
                self._chop_cook(controller, bot_id, holding)
            elif cook:
                self._cook_only(controller, bot_id, holding)
            elif chop:
                self._chop_only(controller, bot_id, holding)
            else:
                self.state = 13
        
        # State 13: Add to plate
        elif self.state == 13:
            if self.plate_counter:
                px, py = self.plate_counter
                if self._move_to(controller, bot_id, px, py):
                    if controller.add_food_to_plate(bot_id, px, py):
                        if self.ingredients_needed:
                            self.ingredients_needed.pop(0)
                        self.state = 10
        
        # State 20: Pickup plate
        elif self.state == 20:
            if holding:
                if holding.get('type') == 'Plate':
                    self.state = 21
                else:
                    self.state = 99
            elif self.plate_counter:
                px, py = self.plate_counter
                if self._move_to(controller, bot_id, px, py):
                    if controller.pickup(bot_id, px, py):
                        self.state = 21
        
        # State 21: Submit
        elif self.state == 21:
            if self._move_to(controller, bot_id, ux, uy):
                controller.submit(bot_id, ux, uy)
                self.state = 0
        
        # State 99: Trash
        elif self.state == 99:
            if holding and self.trash_loc:
                tx, ty = self.trash_loc
                if self._move_to(controller, bot_id, tx, ty):
                    controller.trash(bot_id, tx, ty)
                    self.state = 0
                    self.sub_state = 0
                    self.ingredients_needed = []
                    self.plate_counter = None
                    self.work_counter = None
                    # Ensure we don't start immediately with bad cache
                    self.counters = None 
            else:
                self.state = 0
                self.sub_state = 0
                self.ingredients_needed = []
                self.plate_counter = None
                self.work_counter = None

    def _chop_only(self, controller, bot_id, holding):
        if not self.work_counter:
            exclude = {self.plate_counter} if self.plate_counter else set()
            bot_state = controller.get_bot_state(bot_id)
            self.work_counter = self._get_free_counter(controller, exclude, bot_pos=(bot_state['x'], bot_state['y']))
        
        if not self.work_counter:
            self.state = 99
            return
        
        wx, wy = self.work_counter
        
        if self.sub_state == 0:  # Place
            if holding:
                if holding.get('type') == 'Food':
                    if holding.get('chopped'):
                        self.state = 13
                        return
                    if self._move_to(controller, bot_id, wx, wy):
                        if controller.place(bot_id, wx, wy):
                            self.sub_state = 1
                else:
                    self.state = 99
            else:
                self.state = 11
                
        elif self.sub_state == 1:  # Chop
            if self._move_to(controller, bot_id, wx, wy):
                tile = controller.get_tile(controller.get_team(), wx, wy)
                if tile and isinstance(tile.item, Food):
                    if tile.item.chopped:
                        self.sub_state = 2
                    else:
                        controller.chop(bot_id, wx, wy)
                else:
                    self.state = 11
                    self.sub_state = 0
                    
        elif self.sub_state == 2:  # Pickup
            if holding:
                self.state = 13
                self.sub_state = 0
            else:
                if self._move_to(controller, bot_id, wx, wy):
                    if controller.pickup(bot_id, wx, wy):
                        self.state = 13
                        self.sub_state = 0

    def _cook_only(self, controller, bot_id, holding):
        if self.sub_state == 0:  # Place on cooker
            if holding:
                if holding.get('type') == 'Food':
                    stage = holding.get('cooked_stage', 0)
                    if stage == 1:
                        self.state = 13
                        return
                    elif stage == 2:
                        self.state = 99
                        return
                    
                    cooker = self._get_free_cooker(controller)
                    if cooker:
                        kx, ky = cooker
                        if self._move_to(controller, bot_id, kx, ky):
                            if controller.place(bot_id, kx, ky):
                                self.sub_state = 1
                    elif self.cookers:
                        self._move_to(controller, bot_id, *self.cookers[0])
                else:
                    self.state = 99
            else:
                # RECOVERY: Pick up from work_counter if verified
                if self.work_counter:
                     wx, wy = self.work_counter
                     tile = controller.get_tile(controller.get_team(), wx, wy)
                     if tile and isinstance(tile.item, Food):
                         if self._move_to(controller, bot_id, wx, wy):
                             controller.pickup(bot_id, wx, wy)
                         return

                for kx, ky in self.cookers:
                    tile = controller.get_tile(controller.get_team(), kx, ky)
                    if tile and isinstance(tile.item, Pan) and tile.item.food:
                        self.sub_state = 1
                        return
                self.state = 11
                self.sub_state = 0
                
        elif self.sub_state == 1:  # Wait
            for kx, ky in self.cookers:
                tile = controller.get_tile(controller.get_team(), kx, ky)
                if tile and isinstance(tile.item, Pan) and tile.item.food:
                    stage = tile.item.food.cooked_stage
                    if stage == 1:
                        if self._move_to(controller, bot_id, kx, ky):
                            if controller.take_from_pan(bot_id, kx, ky):
                                self.state = 13
                                self.sub_state = 0
                        return
                    elif stage == 2:
                        if self._move_to(controller, bot_id, kx, ky):
                            controller.take_from_pan(bot_id, kx, ky)
                        self.state = 99
                        self.sub_state = 0
                        return
                    else:
                        self._move_to(controller, bot_id, kx, ky)
                        return
            self.state = 11
            self.sub_state = 0

    def _chop_cook(self, controller, bot_id, holding):
        if not self.work_counter:
            exclude = {self.plate_counter} if self.plate_counter else set()
            bot_state = controller.get_bot_state(bot_id)
            self.work_counter = self._get_free_counter(controller, exclude, bot_pos=(bot_state['x'], bot_state['y']))
        
        if not self.work_counter:
            self.state = 99
            return
        
        wx, wy = self.work_counter
        
        if self.sub_state == 0:  # Place to chop
            if holding:
                if holding.get('type') == 'Food':
                    if holding.get('chopped'):
                        self.sub_state = 3
                        return
                    if self._move_to(controller, bot_id, wx, wy):
                        if controller.place(bot_id, wx, wy):
                            self.sub_state = 1
                else:
                    self.state = 99
            else:
                tile = controller.get_tile(controller.get_team(), wx, wy)
                if tile and isinstance(tile.item, Food):
                    self.sub_state = 1
                else:
                    for kx, ky in self.cookers:
                        t = controller.get_tile(controller.get_team(), kx, ky)
                        if t and isinstance(t.item, Pan) and t.item.food:
                            self.sub_state = 4
                            return
                    self.state = 11
                    self.sub_state = 0
                    
        elif self.sub_state == 1:  # Chop
            if self._move_to(controller, bot_id, wx, wy):
                tile = controller.get_tile(controller.get_team(), wx, wy)
                if tile and isinstance(tile.item, Food):
                    if tile.item.chopped:
                        self.sub_state = 2
                    else:
                        controller.chop(bot_id, wx, wy)
                else:
                    self.state = 11
                    self.sub_state = 0
                    
        elif self.sub_state == 2:  # Pickup chopped
            if holding:
                self.sub_state = 3
            else:
                if self._move_to(controller, bot_id, wx, wy):
                    controller.pickup(bot_id, wx, wy)
                    
        elif self.sub_state == 3:  # Place on cooker
            if holding:
                if holding.get('type') == 'Food':
                    stage = holding.get('cooked_stage', 0)
                    if stage == 1:
                        self.state = 13
                        self.sub_state = 0
                        return
                    elif stage == 2:
                        self.state = 99
                        self.sub_state = 0
                        return
                    
                    cooker = self._get_free_cooker(controller)
                    if cooker:
                        kx, ky = cooker
                        if self._move_to(controller, bot_id, kx, ky):
                            if controller.place(bot_id, kx, ky):
                                self.sub_state = 4
                    elif self.cookers:
                        self._move_to(controller, bot_id, *self.cookers[0])
                else:
                    self.state = 99
            else:
                for kx, ky in self.cookers:
                    tile = controller.get_tile(controller.get_team(), kx, ky)
                    if tile and isinstance(tile.item, Pan) and tile.item.food:
                        self.sub_state = 4
                        return
                self.sub_state = 2
                
        elif self.sub_state == 4:  # Wait for cook
            for kx, ky in self.cookers:
                tile = controller.get_tile(controller.get_team(), kx, ky)
                if tile and isinstance(tile.item, Pan) and tile.item.food:
                    stage = tile.item.food.cooked_stage
                    if stage == 1:
                        if self._move_to(controller, bot_id, kx, ky):
                            if controller.take_from_pan(bot_id, kx, ky):
                                self.state = 13
                                self.sub_state = 0
                        return
                    elif stage == 2:
                        if self._move_to(controller, bot_id, kx, ky):
                            controller.take_from_pan(bot_id, kx, ky)
                        self.state = 99
                        self.sub_state = 0
                        return
                    else:
                        self._move_to(controller, bot_id, kx, ky)
                        return
            self.state = 11
            self.sub_state = 0
