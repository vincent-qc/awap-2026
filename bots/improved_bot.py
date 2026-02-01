"""
Improved bot for AWAP 2026 Carnegie Cookoff.
Optimized for speed and handles all edge cases.
"""

from collections import deque
from typing import Tuple, Optional, Set, List

from game_constants import Team, FoodType, ShopCosts
from robot_controller import RobotController
from item import Pan, Plate, Food


class BotState:
    def __init__(self):
        self.state = 0
        self.sub_state = 0
        self.ingredients_needed = []
        self.plate_counter = None
        self.work_counter = None
        self.current_order = None
        
        # Stuck detection
        self.stuck_counter = 0
        self.last_state = -1

class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        
        # Cached locations (set once)
        self.counters = None
        self.cookers = None
        self.shop_loc = None
        self.submit_loc = None
        self.trash_loc = None
        
        # Bot States: map bot_id -> BotState
        self.bot_states = {}

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

    def _move_to(self, controller, bot_id, target_x, target_y, blocked_tiles=None):
        """Move towards target. Returns True if adjacent."""
        state = controller.get_bot_state(bot_id)
        bx, by = state['x'], state['y']
        
        if max(abs(bx - target_x), abs(by - target_y)) <= 1:
            return True
        
        step = self._bfs_step(controller, (bx, by), target_x, target_y, blocked=blocked_tiles)
        if step and step != (0, 0):
            controller.move(bot_id, step[0], step[1])
        return False

    def _get_free_counter(self, controller, exclude=None, bot_pos=None, blocked_tiles=None):
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
                    step = self._bfs_step(controller, bot_pos, cx, cy, blocked=blocked_tiles)
                    if step is not None:
                        return (cx, cy)
                else:
                    return (cx, cy)
        
        # Fallback: any counter not in exclude, preferring reachable ones
        for cx, cy in counters:
            if (cx, cy) not in exclude:
                if bot_pos:
                    step = self._bfs_step(controller, bot_pos, cx, cy, blocked=blocked_tiles)
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
        
        self._cache_locations(controller)
        if not self.shop_loc or not self.submit_loc:
            return

        # Initialize state for new bots
        for bot_id in bots:
            if bot_id not in self.bot_states:
                self.bot_states[bot_id] = BotState()
                
        # Debug: print all bot states
        turn = controller.get_turn()
        team = controller.get_team().name
        debug_info = []
        for bid in bots:
            if bid in self.bot_states:
                info = controller.get_bot_state(bid)
                state_val = self.bot_states[bid].state
                debug_info.append(f"B{bid}:({info['x']},{info['y']})[S{state_val}]")
        print(f"[{team} Turn {turn}] {' | '.join(debug_info)}")

        # Gather shared knowledge
        reserved_counters = set()
        claimed_orders = set()
        all_bot_positions = {}
        
        for bid in bots:
            # Positions
            b_info = controller.get_bot_state(bid)
            all_bot_positions[bid] = (b_info['x'], b_info['y'])
            
            # Reserved resources
            if bid in self.bot_states:
                st = self.bot_states[bid]
                if st.plate_counter:
                    reserved_counters.add(st.plate_counter)
                if st.work_counter:
                    reserved_counters.add(st.work_counter)
                if st.current_order:
                    claimed_orders.add(st.current_order['order_id'])

        # Execute each bot
        for bot_id in bots:
            if bot_id not in self.bot_states: continue

            # Exclude this bot from blocked tiles logic
            other_bots_locs = {pos for bid, pos in all_bot_positions.items() if bid != bot_id}
            
            # Exclude this bot's own resources from exclusion list
            my_state = self.bot_states[bot_id]
            my_reserved = set()
            if my_state.plate_counter: my_reserved.add(my_state.plate_counter)
            if my_state.work_counter: my_reserved.add(my_state.work_counter)
            
            others_reserved_counters = reserved_counters - my_reserved
            
            self._run_bot(controller, bot_id, my_state, others_reserved_counters, claimed_orders, other_bots_locs)

    def _run_bot(self, controller, bot_id, state, reserved_counters, claimed_orders, blocked_tiles):
        info = controller.get_bot_state(bot_id)
        holding = info.get('holding')
        bx, by = info['x'], info['y']
        
        sx, sy = self.shop_loc
        ux, uy = self.submit_loc
        
        # Stuck detection
        if state.state == state.last_state:
            state.stuck_counter += 1
            if state.stuck_counter > 10:
                print(f"[Bot {bot_id}] STUCK in state {state.state}, forcing random move")
                import random
                dx, dy = random.choice([(0,1), (0,-1), (1,0), (-1,0)])
                controller.move(bot_id, dx, dy)
                state.stuck_counter = 0
                return
        else:
            state.stuck_counter = 0
        state.last_state = state.state
        
        # Burnt food check
        if holding and holding.get('type') == 'Food' and holding.get('cooked_stage') == 2:
            state.state = 99
        
        # State 0: Pick order
        if state.state == 0:
            state.ingredients_needed = []
            state.plate_counter = None
            state.work_counter = None
            state.sub_state = 0
            state.current_order = None
            
            orders = controller.get_orders(controller.get_team())
            turn = controller.get_turn()
            
            best = None
            best_score = 999
            for o in orders:
                # Filter inactive, expired, or claimed by OTHER bots
                if o['is_active'] and o['expires_turn'] > turn:
                    if o['order_id'] in claimed_orders:
                        continue
                        
                    score = len(o['required'])
                    # Prioritize expensive/complex orders if time permits
                    if score < best_score:
                        best_score = score
                        best = o
            
            if best:
                state.current_order = best
                claimed_orders.add(best['order_id']) # Claim it immediately for subsequent bots in this turn
                
                req = list(best['required'])
                # Non-cooking first
                state.ingredients_needed = [i for i in req if not self._needs_cooking(i)] + \
                                          [i for i in req if self._needs_cooking(i)]
                state.state = 1
        
        # State 1: Ensure pan
        elif state.state == 1:
            if any(self._needs_cooking(i) for i in state.ingredients_needed):
                if self._get_free_cooker(controller):
                    state.state = 2
                else:
                    cooker = self._get_cooker_needing_pan(controller)
                    if cooker:
                        kx, ky = cooker
                        if holding:
                            if holding.get('type') == 'Pan':
                                if self._move_to(controller, bot_id, kx, ky, blocked_tiles):
                                    controller.place(bot_id, kx, ky)
                                    state.state = 2
                            else:
                                state.state = 99
                        else:
                            if self._move_to(controller, bot_id, sx, sy, blocked_tiles):
                                if controller.get_team_money(controller.get_team()) >= ShopCosts.PAN.buy_cost:
                                    controller.buy(bot_id, ShopCosts.PAN, sx, sy)
                    else:
                        # No cooker available/needing pan? Assume setup is ok or wait
                        state.state = 2
            else:
                state.state = 2
        
        # State 2: Buy/place plate
        elif state.state == 2:
            if holding:
                if holding.get('type') == 'Plate':
                    if not state.plate_counter:
                        state.plate_counter = self._get_free_counter(controller, exclude=reserved_counters, bot_pos=(bx, by), blocked_tiles=blocked_tiles)
                    
                    if state.plate_counter:
                        # Reserve it
                        reserved_counters.add(state.plate_counter)
                        
                        px, py = state.plate_counter
                        adjacent = self._move_to(controller, bot_id, px, py, blocked_tiles)
                        if adjacent:
                            if controller.place(bot_id, px, py):
                                state.state = 10
                    else:
                        print(f"Error: No plate counter available for Bot {bot_id}")
                else:
                    state.state = 99
            else:
                adjacent = self._move_to(controller, bot_id, sx, sy, blocked_tiles)
                money = controller.get_team_money(controller.get_team())
                if adjacent:
                    if money >= ShopCosts.PLATE.buy_cost:
                        controller.buy(bot_id, ShopCosts.PLATE, sx, sy)
        
        # State 10: Next ingredient
        elif state.state == 10:
            if not state.ingredients_needed:
                state.state = 20
            else:
                state.sub_state = 0
                state.work_counter = None
                state.state = 11
        
        # State 11: Buy ingredient
        elif state.state == 11:
            if not state.ingredients_needed:
                state.state = 20
                return
            
            ing = state.ingredients_needed[0]
            ft = self._food_type(ing)
            
            if holding:
                if holding.get('type') == 'Food':
                    # verify it's the right food
                    if holding.get('food_name') == ing:
                         state.state = 12
                    else:
                         state.state = 99
                else:
                    state.state = 99
            else:
                # CHECK IF WE ALREADY HAVE IT ON MAP
                loc, is_cooking = self._find_existing_ingredient(controller, ing)
                # TODO: Check if this ingredient is claimed/being used by another bot?
                # For now, simplistic check to assume if we didn't put it there, we shouldn't steal it unless desperate
                
                if loc:
                     if is_cooking:
                         # Cooking in execution
                         if self._needs_chopping(ing):
                             state.sub_state = 4 
                         else:
                             state.sub_state = 1 
                     else:
                         state.work_counter = loc
                         reserved_counters.add(loc)
                         
                         # Check status
                         tile = controller.get_tile(controller.get_team(), *loc)
                         if tile and isinstance(tile.item, Food):
                             is_chopped = tile.item.chopped
                             
                             if self._needs_chopping(ing):
                                 if is_chopped:
                                     state.sub_state = 2 # Pickup chopped
                                 else:
                                     state.sub_state = 1 # Chop
                             else:
                                 state.sub_state = 0 
                                 
                     state.state = 12
                     return

                # Buy only if not found
                if self._move_to(controller, bot_id, sx, sy, blocked_tiles):
                    cost = ft.buy_cost
                    if controller.get_team_money(controller.get_team()) >= cost:
                        controller.buy(bot_id, ft, sx, sy)
                        state.state = 12
        
        # State 12: Process
        elif state.state == 12:
            if not state.ingredients_needed:
                state.state = 20
                return
            
            ing = state.ingredients_needed[0]
            chop = self._needs_chopping(ing)
            cook = self._needs_cooking(ing)
            
            if chop and cook:
                self._chop_cook(controller, bot_id, state, holding, reserved_counters, blocked_tiles)
            elif cook:
                self._cook_only(controller, bot_id, state, holding, reserved_counters, blocked_tiles)
            elif chop:
                self._chop_only(controller, bot_id, state, holding, reserved_counters, blocked_tiles)
            else:
                state.state = 13
        
        # State 13: Add to plate
        elif state.state == 13:
            if state.plate_counter:
                px, py = state.plate_counter
                if self._move_to(controller, bot_id, px, py, blocked_tiles):
                    if controller.add_food_to_plate(bot_id, px, py):
                        if state.ingredients_needed:
                            state.ingredients_needed.pop(0)
                        state.state = 10
        
        # State 20: Pickup plate
        elif state.state == 20:
            if holding:
                if holding.get('type') == 'Plate':
                    state.state = 21
                else:
                    state.state = 99
            elif state.plate_counter:
                px, py = state.plate_counter
                if self._move_to(controller, bot_id, px, py, blocked_tiles):
                    if controller.pickup(bot_id, px, py):
                        state.state = 21
        
        # State 21: Submit
        elif state.state == 21:
            if self._move_to(controller, bot_id, ux, uy, blocked_tiles):
                controller.submit(bot_id, ux, uy)
                state.state = 0
        
        # State 99: Trash
        elif state.state == 99:
            if holding and self.trash_loc:
                tx, ty = self.trash_loc
                if self._move_to(controller, bot_id, tx, ty, blocked_tiles):
                    controller.trash(bot_id, tx, ty)
                    state.state = 0
                    state.sub_state = 0
                    state.ingredients_needed = []
                    state.plate_counter = None
                    state.work_counter = None
                    state.current_order = None
            else:
                state.state = 0
                state.sub_state = 0
                state.ingredients_needed = []
                state.plate_counter = None
                state.work_counter = None
                state.current_order = None

    def _chop_only(self, controller, bot_id, state, holding, reserved, blocked):
        if not state.work_counter:
             # Exclude global reserved + own plate counter
            exclude = set(reserved)
            if state.plate_counter: exclude.add(state.plate_counter)
            
            bot_state = controller.get_bot_state(bot_id)
            state.work_counter = self._get_free_counter(controller, exclude, bot_pos=(bot_state['x'], bot_state['y']), blocked_tiles=blocked)
        
        if not state.work_counter:
            state.state = 99
            return
        
        reserved.add(state.work_counter) # Lock it again
        wx, wy = state.work_counter
        
        if state.sub_state == 0:  # Place
            if holding:
                if holding.get('type') == 'Food':
                    if holding.get('chopped'):
                        state.state = 13
                        return
                    if self._move_to(controller, bot_id, wx, wy, blocked):
                        if controller.place(bot_id, wx, wy):
                            state.sub_state = 1
                else:
                    state.state = 99
            else:
                state.state = 11
                
        elif state.sub_state == 1:  # Chop
            if self._move_to(controller, bot_id, wx, wy, blocked):
                tile = controller.get_tile(controller.get_team(), wx, wy)
                if tile and isinstance(tile.item, Food):
                    if tile.item.chopped:
                        state.sub_state = 2
                    else:
                        controller.chop(bot_id, wx, wy)
                else:
                    state.state = 11
                    state.sub_state = 0
                    
        elif state.sub_state == 2:  # Pickup
            if holding:
                state.state = 13
                state.sub_state = 0
            else:
                if self._move_to(controller, bot_id, wx, wy, blocked):
                    if controller.pickup(bot_id, wx, wy):
                        state.state = 13
                        state.sub_state = 0

    def _cook_only(self, controller, bot_id, state, holding, reserved, blocked):
        if state.sub_state == 0:  # Place on cooker
            if holding:
                if holding.get('type') == 'Food':
                    stage = holding.get('cooked_stage', 0)
                    if stage == 1:
                        state.state = 13
                        return
                    elif stage == 2:
                        state.state = 99
                        return
                    
                    cooker = self._get_free_cooker(controller)
                    if cooker:
                        kx, ky = cooker
                        if self._move_to(controller, bot_id, kx, ky, blocked):
                            if controller.place(bot_id, kx, ky):
                                state.sub_state = 1
                    elif self.cookers:
                        self._move_to(controller, bot_id, *self.cookers[0], blocked_tiles=blocked)
                else:
                    state.state = 99
            else:
                # RECOVERY: Pick up from work_counter if verified
                if state.work_counter:
                     wx, wy = state.work_counter
                     tile = controller.get_tile(controller.get_team(), wx, wy)
                     if tile and isinstance(tile.item, Food):
                         if self._move_to(controller, bot_id, wx, wy, blocked):
                             controller.pickup(bot_id, wx, wy)
                         return

                for kx, ky in self.cookers:
                    tile = controller.get_tile(controller.get_team(), kx, ky)
                    if tile and isinstance(tile.item, Pan) and tile.item.food:
                        state.sub_state = 1
                        return
                state.state = 11
                state.sub_state = 0
                
        elif state.sub_state == 1:  # Wait
            for kx, ky in self.cookers:
                tile = controller.get_tile(controller.get_team(), kx, ky)
                if tile and isinstance(tile.item, Pan) and tile.item.food:
                    stage = tile.item.food.cooked_stage
                    if stage == 1:
                        if self._move_to(controller, bot_id, kx, ky, blocked):
                            if controller.take_from_pan(bot_id, kx, ky):
                                state.state = 13
                                state.sub_state = 0
                        return
                    elif stage == 2:
                        if self._move_to(controller, bot_id, kx, ky, blocked):
                            controller.take_from_pan(bot_id, kx, ky)
                        state.state = 99
                        state.sub_state = 0
                        return
                    else:
                        self._move_to(controller, bot_id, kx, ky, blocked)
                        return
            state.state = 11
            state.sub_state = 0

    def _chop_cook(self, controller, bot_id, state, holding, reserved, blocked):
        if not state.work_counter:
            exclude = set(reserved)
            if state.plate_counter: exclude.add(state.plate_counter)
            bot_state = controller.get_bot_state(bot_id)
            state.work_counter = self._get_free_counter(controller, exclude, bot_pos=(bot_state['x'], bot_state['y']), blocked_tiles=blocked)
        
        if not state.work_counter:
            state.state = 99
            return
        
        reserved.add(state.work_counter)
        wx, wy = state.work_counter
        
        if state.sub_state == 0:  # Place to chop
            if holding:
                if holding.get('type') == 'Food':
                    if holding.get('chopped'):
                        state.sub_state = 3
                        return
                    if self._move_to(controller, bot_id, wx, wy, blocked):
                        if controller.place(bot_id, wx, wy):
                            state.sub_state = 1
                else:
                    state.state = 99
            else:
                tile = controller.get_tile(controller.get_team(), wx, wy)
                if tile and isinstance(tile.item, Food):
                    state.sub_state = 1
                else:
                    for kx, ky in self.cookers:
                        t = controller.get_tile(controller.get_team(), kx, ky)
                        if t and isinstance(t.item, Pan) and t.item.food:
                            state.sub_state = 4
                            return
                    state.state = 11
                    state.sub_state = 0
                    
        elif state.sub_state == 1:  # Chop
            if self._move_to(controller, bot_id, wx, wy, blocked):
                tile = controller.get_tile(controller.get_team(), wx, wy)
                if tile and isinstance(tile.item, Food):
                    if tile.item.chopped:
                        state.sub_state = 2
                    else:
                        controller.chop(bot_id, wx, wy)
                else:
                    state.state = 11
                    state.sub_state = 0
                    
        elif state.sub_state == 2:  # Pickup chopped
            if holding:
                state.sub_state = 3
            else:
                if self._move_to(controller, bot_id, wx, wy, blocked):
                    controller.pickup(bot_id, wx, wy)
                    
        elif state.sub_state == 3:  # Place on cooker
            if holding:
                if holding.get('type') == 'Food':
                    stage = holding.get('cooked_stage', 0)
                    if stage == 1:
                        state.state = 13
                        state.sub_state = 0
                        return
                    elif stage == 2:
                        state.state = 99
                        state.sub_state = 0
                        return
                    
                    cooker = self._get_free_cooker(controller)
                    if cooker:
                        kx, ky = cooker
                        if self._move_to(controller, bot_id, kx, ky, blocked):
                            if controller.place(bot_id, kx, ky):
                                state.sub_state = 4
                    elif self.cookers:
                        self._move_to(controller, bot_id, *self.cookers[0], blocked_tiles=blocked)
                else:
                    state.state = 99
            else:
                for kx, ky in self.cookers:
                    tile = controller.get_tile(controller.get_team(), kx, ky)
                    if tile and isinstance(tile.item, Pan) and tile.item.food:
                        state.sub_state = 4
                        return
                state.sub_state = 2
                
        elif state.sub_state == 4:  # Wait for cook
            for kx, ky in self.cookers:
                tile = controller.get_tile(controller.get_team(), kx, ky)
                if tile and isinstance(tile.item, Pan) and tile.item.food:
                    stage = tile.item.food.cooked_stage
                    if stage == 1:
                        if self._move_to(controller, bot_id, kx, ky, blocked):
                            if controller.take_from_pan(bot_id, kx, ky):
                                state.state = 13
                                state.sub_state = 0
                        return
                    elif stage == 2:
                        if self._move_to(controller, bot_id, kx, ky, blocked):
                            controller.take_from_pan(bot_id, kx, ky)
                        state.state = 99
                        state.sub_state = 0
                        return
                    else:
                        self._move_to(controller, bot_id, kx, ky, blocked)
                        return
            state.state = 11
            state.sub_state = 0
