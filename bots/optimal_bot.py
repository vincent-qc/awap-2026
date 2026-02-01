"""
Optimal Bot for Carnegie Cookoff (AWAP 2026)

Simple, robust state machine for NOODLES + MEAT orders.
- Primary bot handles all orders sequentially
- Support bot washes dishes
- Proper error recovery with state timeouts
"""

import heapq
from typing import Tuple, Optional, List, Dict, Any

from game_constants import Team, FoodType, ShopCosts, GameConstants
from robot_controller import RobotController
from item import Pan, Plate, Food


class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.initialized = False
        
        # Cached tile locations
        self.shop_locs: List[Tuple[int, int]] = []
        self.cooker_locs: List[Tuple[int, int]] = []
        self.counter_locs: List[Tuple[int, int]] = []
        self.trash_locs: List[Tuple[int, int]] = []
        self.submit_locs: List[Tuple[int, int]] = []
        self.sink_locs: List[Tuple[int, int]] = []
        
        # State tracking per bot
        self.bot_states: Dict[int, int] = {}
        self.state_entered_turn: Dict[int, int] = {}  # Track when we entered current state
        
        # Primary resources
        self.primary_cooker: Optional[Tuple[int, int]] = None
        self.primary_counter: Optional[Tuple[int, int]] = None
        
        self.orders_completed = 0

    def initialize(self, controller: RobotController):
        """Cache tile locations"""
        m = controller.get_map()
        
        for x in range(m.width):
            for y in range(m.height):
                tile = m.tiles[x][y]
                name = getattr(tile, 'tile_name', '')
                
                if name == "SHOP":
                    self.shop_locs.append((x, y))
                elif name == "COOKER":
                    self.cooker_locs.append((x, y))
                elif name == "COUNTER":
                    self.counter_locs.append((x, y))
                elif name == "TRASH":
                    self.trash_locs.append((x, y))
                elif name == "SUBMIT":
                    self.submit_locs.append((x, y))
                elif name == "SINK":
                    self.sink_locs.append((x, y))
        
        if self.cooker_locs:
            self.primary_cooker = self.cooker_locs[0]
        if self.counter_locs:
            self.primary_counter = self.counter_locs[0]
        
        self.initialized = True

    def chebyshev(self, x1: int, y1: int, x2: int, y2: int) -> int:
        return max(abs(x1 - x2), abs(y1 - y2))

    def find_nearest(self, x: int, y: int, locs: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        if not locs:
            return None
        return min(locs, key=lambda loc: self.chebyshev(x, y, loc[0], loc[1]))

    def get_step_towards(self, controller: RobotController, bot_id: int,
                         start: Tuple[int, int], target: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """BFS to find next step toward target (adjacent to it)"""
        if self.chebyshev(start[0], start[1], target[0], target[1]) <= 1:
            return None  # Already adjacent
        
        m = controller.get_map()
        
        # Get other bot positions to avoid
        occupied = set()
        for other_id in controller.get_team_bot_ids():
            if other_id != bot_id:
                other = controller.get_bot_state(other_id)
                if other:
                    occupied.add((other['x'], other['y']))
        
        # BFS
        from collections import deque
        queue = deque([(start, None)])  # (position, first_step)
        visited = {start}
        
        while queue:
            pos, first_step = queue.popleft()
            
            if self.chebyshev(pos[0], pos[1], target[0], target[1]) <= 1:
                return first_step
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    nx, ny = pos[0] + dx, pos[1] + dy
                    
                    if (nx, ny) in visited:
                        continue
                    if not (0 <= nx < m.width and 0 <= ny < m.height):
                        continue
                    if not m.is_tile_walkable(nx, ny):
                        continue
                    # Avoid other bots for immediate step only
                    if first_step is None and (nx, ny) in occupied:
                        continue
                    
                    visited.add((nx, ny))
                    step = first_step if first_step else (dx, dy)
                    queue.append(((nx, ny), step))
        
        return None

    def move_to(self, controller: RobotController, bot_id: int, tx: int, ty: int) -> bool:
        """Move toward target. Returns True if adjacent."""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return False
        
        bx, by = bot['x'], bot['y']
        
        if self.chebyshev(bx, by, tx, ty) <= 1:
            return True
        
        step = self.get_step_towards(controller, bot_id, (bx, by), (tx, ty))
        if step:
            controller.move(bot_id, step[0], step[1])
        
        return False

    def has_pan(self, controller: RobotController, cooker: Tuple[int, int]) -> bool:
        tile = controller.get_tile(controller.get_team(), cooker[0], cooker[1])
        if not tile:
            return False
        item = getattr(tile, 'item', None)
        return isinstance(item, Pan)

    def pan_food_status(self, controller: RobotController, cooker: Tuple[int, int]) -> Tuple[bool, int]:
        """Returns (has_food, cooked_stage)"""
        tile = controller.get_tile(controller.get_team(), cooker[0], cooker[1])
        if not tile:
            return (False, 0)
        pan = getattr(tile, 'item', None)
        if not isinstance(pan, Pan) or pan.food is None:
            return (False, 0)
        return (True, pan.food.cooked_stage)

    def counter_has_item(self, controller: RobotController, counter: Tuple[int, int]) -> bool:
        tile = controller.get_tile(controller.get_team(), counter[0], counter[1])
        if not tile:
            return False
        return getattr(tile, 'item', None) is not None

    def set_state(self, bot_id: int, state: int, turn: int):
        """Set state and track when we entered it"""
        self.bot_states[bot_id] = state
        self.state_entered_turn[bot_id] = turn

    def run_primary_bot(self, controller: RobotController, bot_id: int):
        """
        Main order workflow for NOODLES + MEAT.
        
        States:
        0: Check pan on cooker
        1: Buy pan -> place on cooker
        2: Buy meat
        3: Place meat on counter
        4: Chop meat
        5: Pick up chopped meat
        6: Put meat in cooker (starts cooking)
        7: Buy plate
        8: Place plate on counter
        9: Buy noodles
        10: Add noodles to plate (holding noodles, plate on counter)
        11: Wait for meat to cook, then take it
        12: Add meat to plate (holding meat, plate on counter)
        13: Pick up plate
        14: Submit order
        99: Error recovery - trash and restart
        """
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return
        
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        turn = controller.get_turn()
        
        state = self.bot_states.get(bot_id, 0)
        state_turn = self.state_entered_turn.get(bot_id, turn)
        
        # Get key locations
        shop = self.find_nearest(bx, by, self.shop_locs)
        cooker = self.primary_cooker
        counter = self.primary_counter
        submit = self.find_nearest(bx, by, self.submit_locs)
        trash = self.find_nearest(bx, by, self.trash_locs)
        
        if not all([shop, cooker, counter, submit]):
            return
        
        # === ERROR RECOVERY ===
        # If holding burnt food, go to error recovery
        if holding and holding.get('type') == 'Food' and holding.get('cooked_stage') == 2:
            self.set_state(bot_id, 99, turn)
            state = 99
        
        # Timeout: if stuck in same state for too long, try to recover
        if turn - state_turn > 50:
            # We're stuck, try to recover
            if holding:
                self.set_state(bot_id, 99, turn)
                state = 99
            else:
                # Reset to beginning
                self.set_state(bot_id, 0, turn)
                state = 0
        
        # === STATE MACHINE ===
        
        if state == 0:  # Check pan
            if self.has_pan(controller, cooker):
                self.set_state(bot_id, 2, turn)
            else:
                self.set_state(bot_id, 1, turn)
        
        elif state == 1:  # Buy and place pan
            if holding and holding.get('type') == 'Pan':
                if self.move_to(controller, bot_id, cooker[0], cooker[1]):
                    if controller.place(bot_id, cooker[0], cooker[1]):
                        self.set_state(bot_id, 2, turn)
            elif not holding:
                if self.move_to(controller, bot_id, shop[0], shop[1]):
                    if controller.get_team_money() >= ShopCosts.PAN.buy_cost:
                        controller.buy(bot_id, ShopCosts.PAN, shop[0], shop[1])
        
        elif state == 2:  # Buy meat
            if holding and holding.get('type') == 'Food' and holding.get('food_name') == 'MEAT':
                self.set_state(bot_id, 3, turn)
            elif not holding:
                if self.move_to(controller, bot_id, shop[0], shop[1]):
                    if controller.get_team_money() >= FoodType.MEAT.buy_cost:
                        if controller.buy(bot_id, FoodType.MEAT, shop[0], shop[1]):
                            self.set_state(bot_id, 3, turn)
        
        elif state == 3:  # Place meat on counter
            if holding and holding.get('type') == 'Food' and holding.get('food_name') == 'MEAT':
                if self.move_to(controller, bot_id, counter[0], counter[1]):
                    if controller.place(bot_id, counter[0], counter[1]):
                        self.set_state(bot_id, 4, turn)
            elif not holding:
                # Lost the meat somehow, rebuy
                self.set_state(bot_id, 2, turn)
        
        elif state == 4:  # Chop meat
            if not holding:
                if self.move_to(controller, bot_id, counter[0], counter[1]):
                    if controller.chop(bot_id, counter[0], counter[1]):
                        self.set_state(bot_id, 5, turn)
        
        elif state == 5:  # Pick up chopped meat
            if not holding:
                if self.move_to(controller, bot_id, counter[0], counter[1]):
                    if controller.pickup(bot_id, counter[0], counter[1]):
                        self.set_state(bot_id, 6, turn)
            elif holding and holding.get('type') == 'Food' and holding.get('food_name') == 'MEAT':
                self.set_state(bot_id, 6, turn)
        
        elif state == 6:  # Put meat in cooker
            if holding and holding.get('type') == 'Food' and holding.get('food_name') == 'MEAT':
                # Check if cooker is available
                has_food, _ = self.pan_food_status(controller, cooker)
                if not has_food:
                    if self.move_to(controller, bot_id, cooker[0], cooker[1]):
                        if controller.place(bot_id, cooker[0], cooker[1]):
                            self.set_state(bot_id, 7, turn)
                # else wait for cooker
            elif not holding:
                # Lost meat, rebuy
                self.set_state(bot_id, 2, turn)
        
        elif state == 7:  # Buy plate
            if holding and holding.get('type') == 'Plate':
                self.set_state(bot_id, 8, turn)
            elif not holding:
                if self.move_to(controller, bot_id, shop[0], shop[1]):
                    if controller.get_team_money() >= ShopCosts.PLATE.buy_cost:
                        if controller.buy(bot_id, ShopCosts.PLATE, shop[0], shop[1]):
                            self.set_state(bot_id, 8, turn)
        
        elif state == 8:  # Place plate on counter
            if holding and holding.get('type') == 'Plate':
                if self.move_to(controller, bot_id, counter[0], counter[1]):
                    if controller.place(bot_id, counter[0], counter[1]):
                        self.set_state(bot_id, 9, turn)
            elif not holding:
                # Lost plate, rebuy
                self.set_state(bot_id, 7, turn)
        
        elif state == 9:  # Buy noodles
            if holding and holding.get('type') == 'Food' and holding.get('food_name') == 'NOODLES':
                self.set_state(bot_id, 10, turn)
            elif not holding:
                if self.move_to(controller, bot_id, shop[0], shop[1]):
                    if controller.get_team_money() >= FoodType.NOODLES.buy_cost:
                        if controller.buy(bot_id, FoodType.NOODLES, shop[0], shop[1]):
                            self.set_state(bot_id, 10, turn)
        
        elif state == 10:  # Add noodles to plate
            if holding and holding.get('type') == 'Food' and holding.get('food_name') == 'NOODLES':
                if self.move_to(controller, bot_id, counter[0], counter[1]):
                    if controller.add_food_to_plate(bot_id, counter[0], counter[1]):
                        self.set_state(bot_id, 11, turn)
            elif not holding:
                # Lost noodles, rebuy
                self.set_state(bot_id, 9, turn)
        
        elif state == 11:  # Wait for meat, then take it
            has_food, cook_stage = self.pan_food_status(controller, cooker)
            
            if has_food and cook_stage == 1:  # Cooked!
                if not holding:
                    if self.move_to(controller, bot_id, cooker[0], cooker[1]):
                        if controller.take_from_pan(bot_id, cooker[0], cooker[1]):
                            self.set_state(bot_id, 12, turn)
            elif has_food and cook_stage == 2:  # Burnt!
                if not holding:
                    if self.move_to(controller, bot_id, cooker[0], cooker[1]):
                        controller.take_from_pan(bot_id, cooker[0], cooker[1])
                self.set_state(bot_id, 99, turn)
            elif not has_food:
                # No food in pan - something went wrong, restart
                self.set_state(bot_id, 2, turn)
            # else: still cooking, wait
        
        elif state == 12:  # Add meat to plate
            if holding and holding.get('type') == 'Food' and holding.get('food_name') == 'MEAT':
                if self.move_to(controller, bot_id, counter[0], counter[1]):
                    if controller.add_food_to_plate(bot_id, counter[0], counter[1]):
                        self.set_state(bot_id, 13, turn)
            elif not holding:
                # Lost meat after taking from pan, restart
                self.set_state(bot_id, 2, turn)
        
        elif state == 13:  # Pick up plate
            if not holding:
                if self.move_to(controller, bot_id, counter[0], counter[1]):
                    if controller.pickup(bot_id, counter[0], counter[1]):
                        self.set_state(bot_id, 14, turn)
            elif holding and holding.get('type') == 'Plate':
                self.set_state(bot_id, 14, turn)
        
        elif state == 14:  # Submit
            if holding and holding.get('type') == 'Plate' and not holding.get('dirty'):
                if self.move_to(controller, bot_id, submit[0], submit[1]):
                    if controller.submit(bot_id, submit[0], submit[1]):
                        self.orders_completed += 1
                        self.set_state(bot_id, 0, turn)  # Start next order
                    # Submit might fail if no matching order, but that's ok
            elif not holding:
                # No plate, restart
                self.set_state(bot_id, 0, turn)
        
        elif state == 99:  # Error recovery
            if holding and trash:
                if self.move_to(controller, bot_id, trash[0], trash[1]):
                    if controller.trash(bot_id, trash[0], trash[1]):
                        self.set_state(bot_id, 0, turn)  # Restart
            elif not holding:
                self.set_state(bot_id, 0, turn)

    def run_support_bot(self, controller: RobotController, bot_id: int):
        """Support bot washes dishes"""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return
        
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        
        # If holding something, try to trash it
        if holding:
            trash = self.find_nearest(bx, by, self.trash_locs)
            if trash:
                if self.move_to(controller, bot_id, trash[0], trash[1]):
                    controller.trash(bot_id, trash[0], trash[1])
            return
        
        # Look for dirty dishes to wash
        for sx, sy in self.sink_locs:
            tile = controller.get_tile(controller.get_team(), sx, sy)
            if tile and getattr(tile, 'num_dirty_plates', 0) > 0:
                if self.move_to(controller, bot_id, sx, sy):
                    controller.wash_sink(bot_id, sx, sy)
                return
        
        # Nothing to do, just stay put

    def play_turn(self, controller: RobotController):
        """Main entry point"""
        if not self.initialized:
            self.initialize(controller)
        
        my_bots = controller.get_team_bot_ids()
        if not my_bots:
            return
        
        turn = controller.get_turn()
        
        # Optional: Strategic map switching
        switch_info = controller.get_switch_info()
        switch_threshold = switch_info["switch_turn"] + int(switch_info["switch_duration"] * 0.7)
        if controller.can_switch_maps() and turn >= switch_threshold and self.orders_completed >= 2:
            controller.switch_maps()
        
        for i, bot_id in enumerate(my_bots):
            bot = controller.get_bot_state(bot_id)
            if not bot:
                continue
            
            # Check if on enemy map (after switch)
            if bot.get('map_team') != controller.get_team().name:
                # On enemy map - could sabotage, but for now just idle
                continue
            
            if i == 0:
                # Primary bot works on orders
                if bot_id not in self.bot_states:
                    self.set_state(bot_id, 0, turn)
                self.run_primary_bot(controller, bot_id)
            else:
                # Support bot washes dishes
                self.run_support_bot(controller, bot_id)
