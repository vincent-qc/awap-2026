# Improved bot for AWAP 2026 Carnegie Cookoff.
# Optimized for speed and handles all edge cases.

from collections import deque
import heapq
from typing import Tuple, Optional, Set, List

from game_constants import Team, FoodType, ShopCosts
from robot_controller import RobotController
from item import Pan, Plate, Food

class BotState:
    def __init__(self):
        self.task_stage = 0
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

    def cache_locations(self, controller):
        # Cache all locations once at start.
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

    def get_next_step_astar(self, controller, start, target_x, target_y, blocked=None):
        # Get next step towards target using A* with Manhattan heuristic. Returns (dx, dy) or None.
        if blocked is None:
            blocked = set()
            
        bx, by = start
        # Already adjacent?
        if max(abs(bx - target_x), abs(by - target_y)) <= 1:
            return (0, 0)
        
        m = controller.get_map(controller.get_team())
        
        # Priority queue: (f_score, g_score, x, y, first_step)
        open_set = []
        heapq.heappush(open_set, (0, 0, bx, by, None))
        
        # Cost from start to node
        g_scores = {(bx, by): 0}
        
        while open_set:
            _, g, cx, cy, first_step = heapq.heappop(open_set)
            
            # If we are adjacent to target, we found the path
            if max(abs(cx - target_x), abs(cy - target_y)) <= 1:
                return first_step
            
            # Check neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                        
                    nx, ny = cx + dx, cy + dy
                    
                    # Bounds check
                    if not (0 <= nx < m.width and 0 <= ny < m.height):
                        continue
                    
                    # Walkability check
                    if not m.is_tile_walkable(nx, ny):
                        continue
                        
                    # Blocked check
                    if (nx, ny) in blocked:
                        continue
                    
                    new_g = g + 1
                    
                    if (nx, ny) not in g_scores or new_g < g_scores[(nx, ny)]:
                        g_scores[(nx, ny)] = new_g
                        # Manhattan heuristic
                        h = abs(nx - target_x) + abs(ny - target_y)
                        f = new_g + h
                        
                        step = first_step if first_step else (dx, dy)
                        heapq.heappush(open_set, (f, new_g, nx, ny, step))
                        
        return None

    def move_to(self, controller, bot_id, target_x, target_y, blocked_tiles=None):
        # Move towards target. Returns True if adjacent.
        state = controller.get_bot_state(bot_id)
        bx, by = state['x'], state['y']
        
        if max(abs(bx - target_x), abs(by - target_y)) <= 1:
            return True
        
        step = self.get_next_step_astar(controller, (bx, by), target_x, target_y, blocked=blocked_tiles)
        if step and step != (0, 0):
            controller.move(bot_id, step[0], step[1])
        return False

    def get_free_counter(self, controller, exclude=None, bot_pos=None, blocked_tiles=None):
        # Find empty counter that is reachable.
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
                    step = self.get_next_step_astar(controller, bot_pos, cx, cy, blocked=blocked_tiles)
                    if step is not None:
                        return (cx, cy)
                else:
                    return (cx, cy)
        
        # Fallback: any counter not in exclude, preferring reachable ones
        for cx, cy in counters:
            if (cx, cy) not in exclude:
                if bot_pos:
                    step = self.get_next_step_astar(controller, bot_pos, cx, cy, blocked=blocked_tiles)
                    if step is not None:
                        return (cx, cy)
                else:
                    return (cx, cy)
        
        # Last resort: just return first counter
        return self.counters[0] if self.counters else None

    def get_free_cooker(self, controller):
        # Find cooker with empty pan.
        for kx, ky in self.cookers:
            tile = controller.get_tile(controller.get_team(), kx, ky)
            if tile and isinstance(tile.item, Pan) and tile.item.food is None:
                return (kx, ky)
        return None

    def get_cooker_needing_pan(self, controller):
        # Find cooker without pan.
        for kx, ky in self.cookers:
            tile = controller.get_tile(controller.get_team(), kx, ky)
            if tile and not isinstance(tile.item, Pan):
                return (kx, ky)
        return None

    def needs_cooking(self, ing):
        return ing in ('EGG', 'MEAT')
    
    def needs_chopping(self, ing):
        return ing in ('MEAT', 'ONIONS')
    
    def get_food_type(self, name):
        return {'EGG': FoodType.EGG, 'MEAT': FoodType.MEAT, 'NOODLES': FoodType.NOODLES, 
                'ONIONS': FoodType.ONIONS, 'SAUCE': FoodType.SAUCE}.get(name)

    def find_existing_ingredient(self, controller, ingredient):
        # Find partial matching ingredient on map.
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

    def get_bot_inventory(self, controller, bot_id, state):
        # Get list of ingredients currently possessed by this bot.
        inventory = []
        
        # 1. Held item
        info = controller.get_bot_state(bot_id)
        holding = info.get('holding')
        if holding:
             if holding.get('type') == 'Food':
                 inventory.append(holding.get('food_name'))
             elif holding.get('type') == 'Plate':
                 # If plate has food, count it
                 for item in holding.get('food', []):
                     inventory.append(item['food_name'])
        
        # 2. Plate counter
        if state.plate_counter:
            tile = controller.get_tile(controller.get_team(), *state.plate_counter)
            if tile and tile.item:
                if isinstance(tile.item, Food):
                    inventory.append(tile.item.food_name)
                elif isinstance(tile.item, Plate):
                    for item in tile.item.food:
                         inventory.append(item.food_name)

        # 3. Work counter
        if state.work_counter:
            tile = controller.get_tile(controller.get_team(), *state.work_counter)
            if tile and tile.item and isinstance(tile.item, Food):
                inventory.append(tile.item.food_name)
                
        return inventory

    def handle_expired_order(self, controller, bot_id, state, claimed_orders):
        # Check if current order is invalid/expired and try to rescue progress.
        current = state.current_order
        if not current:
            return False
            
        turn = controller.get_turn()
        orders = controller.get_orders(controller.get_team())
        
        # Find current order object in active orders
        matching_active = next((o for o in orders if o['order_id'] == current['order_id']), None)
        
        is_expired = False
        if not matching_active:
            is_expired = True
        elif not matching_active['is_active']:
            is_expired = True
        elif matching_active['expires_turn'] <= turn:
            is_expired = True

        if is_expired:
            print(f"[Bot {bot_id}] Order {current['order_id']} expired/gone!")
            
            # Check inventory for "significant progress"
            inventory = self.get_bot_inventory(controller, bot_id, state)
            
            if not inventory:
                # No progress, just reset normally
                state.task_stage = 0
                state.current_order = None
                return True
                
            print(f"[Bot {bot_id}] Has inventory {inventory}, trying to reuse...")
            
            # Score active orders based on inventory match
            best_match = None
            best_match_score = -1
            
            for o in orders:
                if not o['is_active'] or o['expires_turn'] <= turn:
                    continue
                if o['order_id'] in claimed_orders:
                    continue
                
                # Count how many inventory items are useful for this order
                needed = list(o['required'])[:]
                score = 0
                for inv_item in inventory:
                    if inv_item in needed:
                        score += 5 # high value for reuse
                        needed.remove(inv_item)
                
                # Tie-breaker: prioritization logic (time, value) from normal selection
                score += len(o['required']) # Prefer more complex recipes if equal reuse
                
                if score > best_match_score:
                    best_match_score = score
                    best_match = o
            
            if best_match and best_match_score > 0:
                print(f"[Bot {bot_id}] Switched to Order {best_match['order_id']} to reuse ingredients")
                
                # Unclaim old order just in case (though it's expired)
                if current['order_id'] in claimed_orders:
                    claimed_orders.remove(current['order_id'])
                
                state.current_order = best_match
                claimed_orders.add(best_match['order_id'])
                
                # Re-calculate ingredients needed
                req = list(best_match['required'])
                remaining_needed = list(req)
                
                # Identify items that are ALREADY PLATED (safely done)
                plated_items = []
                
                # Check plate on counter
                if state.plate_counter:
                    tile = controller.get_tile(controller.get_team(), *state.plate_counter)
                    if tile and tile.item and isinstance(tile.item, Plate):
                        for item in tile.item.food:
                            plated_items.append(item.food_name)
                            
                # Check plate in hand
                info = controller.get_bot_state(bot_id)
                holding = info.get('holding')
                if holding and holding.get('type') == 'Plate':
                     for item in holding.get('food', []):
                         plated_items.append(item['food_name'])

                # Remove plated items from requirements
                for done_item in plated_items:
                    if done_item in remaining_needed:
                        remaining_needed.remove(done_item)
                
                # Reset needs
                state.ingredients_needed = [i for i in remaining_needed if not self.needs_cooking(i)] + \
                                          [i for i in remaining_needed if self.needs_cooking(i)]
                
                # Safe bet: State 10 will check needs and proceed.
                state.task_stage = 10 
                return True
            else:
                print(f"[Bot {bot_id}] No matching order for reuse. Resetting.")
                state.task_stage = 99 
                return True
        
        return False

    def calculate_order_heuristic(self, order, turn, team_money):
        # Calculate heuristic score for an order.
        # Score = Profit + (Penalty * 0.5) + (Urgency Bonus)
        # Returns -float('inf') if impossible to complete in time.
        
        # 1. Cost & Profit
        ingredient_cost = 0
        for ing in order['required']:
            ft = self.get_food_type(ing)
            if ft:
                ingredient_cost += ft.buy_cost
        
        profit = order['reward'] - ingredient_cost
        
        # 2. Time
        time_left = order['expires_turn'] - turn
        
        # Estimate time: 15 turns per ingredient + 10 buffer
        # This is a rough heuristic.
        estimated_time = len(order['required']) * 15 + 10
        
        if time_left < estimated_time:
            return -float('inf')
        
        # 3. Score
        # Profit base
        score = profit
        
        # Risk adjustment: prioritize avoiding high penalty
        score += order['penalty'] * 0.5
        
        # Urgency bonus: increases as deadline approaches
        # Cap time_left at 1 to avoid division by zero
        score += (500 / max(1, time_left))
        
        return score

    def play_turn(self, controller: RobotController):
        bots = controller.get_team_bot_ids(controller.get_team())
        if not bots:
            return
        
        self.cache_locations(controller)
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
                state_val = self.bot_states[bid].task_stage
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
        # Determine bot index to assign roles
        team_bots = sorted(bots)
        
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
            
            # Identify which bot is which
            is_bot_1 = (bot_id == team_bots[0])
            is_bot_2 = (len(team_bots) >= 2 and bot_id == team_bots[1])
            
            if is_bot_1:
                self.run_bot_1_behavior(controller, bot_id, my_state, others_reserved_counters, claimed_orders, other_bots_locs)
            elif is_bot_2:
                self.run_bot_2_behavior(controller, bot_id, my_state, others_reserved_counters, claimed_orders, other_bots_locs)
            else:
                 # Default behavior for extra bots
                self.run_standard_logic(controller, bot_id, my_state, others_reserved_counters, claimed_orders, other_bots_locs)


    # Bot 1 Behavior
    def run_bot_1_behavior(self, controller, bot_id, state, reserved_counters, claimed_orders, blocked_tiles):
        # Bot 1 follows standard logic.
        self.run_standard_logic(controller, bot_id, state, reserved_counters, claimed_orders, blocked_tiles)


    # Bot 2 Behavior
    def run_bot_2_behavior(self, controller, bot_id, state, reserved_counters, claimed_orders, blocked_tiles):
        # Bot 2 has smart recovery logic for expired orders.
        if state.current_order:
             if self.handle_expired_order(controller, bot_id, state, claimed_orders):
                 pass # State updated by recovery
        
        self.run_standard_logic(controller, bot_id, state, reserved_counters, claimed_orders, blocked_tiles)


    # Shared Logic
    def run_standard_logic(self, controller, bot_id, state, reserved_counters, claimed_orders, blocked_tiles):
        info = controller.get_bot_state(bot_id)
        holding = info.get('holding')
        bx, by = info['x'], info['y']
        
        sx, sy = self.shop_loc
        ux, uy = self.submit_loc
        
        # Stuck detection
        if state.task_stage == state.last_state:
            state.stuck_counter += 1
            if state.stuck_counter > 10:
                print(f"[Bot {bot_id}] STUCK in state {state.task_stage}, forcing random move")
                import random
                dx, dy = random.choice([(0,1), (0,-1), (1,0), (-1,0)])
                controller.move(bot_id, dx, dy)
                state.stuck_counter = 0
                return
        else:
            state.stuck_counter = 0
        state.last_state = state.task_stage
        
        # Burnt food check
        if holding and holding.get('type') == 'Food' and holding.get('cooked_stage') == 2:
            state.task_stage = 99
        
        # State 0: Pick order
        if state.task_stage == 0:
            state.ingredients_needed = []
            state.plate_counter = None
            state.work_counter = None
            state.sub_state = 0
            state.current_order = None
            
            orders = controller.get_orders(controller.get_team())
            turn = controller.get_turn()
            team_money = controller.get_team_money(controller.get_team())
            
            best = None
            best_score = -float('inf')
            
            for o in orders:
                # Filter inactive, expired, or claimed by OTHER bots
                if o['is_active'] and o['expires_turn'] > turn:
                    if o['order_id'] in claimed_orders:
                        continue
                        
                    score = self.calculate_order_heuristic(o, turn, team_money)
                    
                    if score > best_score:
                        best_score = score
                        best = o
            
            if best:
                state.current_order = best
                claimed_orders.add(best['order_id']) # Claim it immediately for subsequent bots in this turn
                
                req = list(best['required'])
                # Non-cooking first
                state.ingredients_needed = [i for i in req if not self.needs_cooking(i)] + \
                                          [i for i in req if self.needs_cooking(i)]
                state.task_stage = 1
        
        # State 1: Ensure pan
        elif state.task_stage == 1:
            if any(self.needs_cooking(i) for i in state.ingredients_needed):
                if self.get_free_cooker(controller):
                    state.task_stage = 2
                else:
                    cooker = self.get_cooker_needing_pan(controller)
                    if cooker:
                        kx, ky = cooker
                        if holding:
                            if holding.get('type') == 'Pan':
                                if self.move_to(controller, bot_id, kx, ky, blocked_tiles):
                                    controller.place(bot_id, kx, ky)
                                    state.task_stage = 2
                            else:
                                state.task_stage = 99
                        else:
                            if self.move_to(controller, bot_id, sx, sy, blocked_tiles):
                                if controller.get_team_money(controller.get_team()) >= ShopCosts.PAN.buy_cost:
                                    controller.buy(bot_id, ShopCosts.PAN, sx, sy)
                    else:
                        # No cooker available/needing pan? Assume setup is ok or wait
                        state.task_stage = 2
            else:
                state.task_stage = 2
        
        # State 2: Buy/place plate
        elif state.task_stage == 2:
            if holding:
                if holding.get('type') == 'Plate':
                    if not state.plate_counter:
                        state.plate_counter = self.get_free_counter(controller, exclude=reserved_counters, bot_pos=(bx, by), blocked_tiles=blocked_tiles)
                    
                    if state.plate_counter:
                        # Reserve it
                        reserved_counters.add(state.plate_counter)
                        
                        px, py = state.plate_counter
                        adjacent = self.move_to(controller, bot_id, px, py, blocked_tiles)
                        if adjacent:
                            if controller.place(bot_id, px, py):
                                state.task_stage = 10
                    else:
                        print(f"Error: No plate counter available for Bot {bot_id}")
                else:
                    state.task_stage = 99
            else:
                adjacent = self.move_to(controller, bot_id, sx, sy, blocked_tiles)
                money = controller.get_team_money(controller.get_team())
                if adjacent:
                    if money >= ShopCosts.PLATE.buy_cost:
                        controller.buy(bot_id, ShopCosts.PLATE, sx, sy)
        
        # State 10: Next ingredient
        elif state.task_stage == 10:
            if not state.ingredients_needed:
                state.task_stage = 20
            else:
                state.sub_state = 0
                state.work_counter = None
                state.task_stage = 11
        
        # State 11: Buy ingredient
        elif state.task_stage == 11:
            if not state.ingredients_needed:
                state.task_stage = 20
                return
            
            ing = state.ingredients_needed[0]
            ft = self.get_food_type(ing)
            
            if holding:
                if holding.get('type') == 'Food':
                    # verify it's the right food
                    if holding.get('food_name') == ing:
                         state.task_stage = 12
                    else:
                         state.task_stage = 99
                else:
                    state.task_stage = 99
            else:
                # CHECK IF WE ALREADY HAVE IT ON MAP
                loc, is_cooking = self.find_existing_ingredient(controller, ing)
                
                if loc:
                     if is_cooking:
                         # Cooking in execution
                         if self.needs_chopping(ing):
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
                             
                             if self.needs_chopping(ing):
                                 if is_chopped:
                                     state.sub_state = 2 # Pickup chopped
                                 else:
                                     state.sub_state = 1 # Chop
                             else:
                                 state.sub_state = 0 
                                 
                     state.task_stage = 12
                     return

                # Buy only if not found
                if self.move_to(controller, bot_id, sx, sy, blocked_tiles):
                    cost = ft.buy_cost
                    if controller.get_team_money(controller.get_team()) >= cost:
                        controller.buy(bot_id, ft, sx, sy)
                        state.task_stage = 12
        
        # State 12: Process
        elif state.task_stage == 12:
            if not state.ingredients_needed:
                state.task_stage = 20
                return
            
            ing = state.ingredients_needed[0]
            chop = self.needs_chopping(ing)
            cook = self.needs_cooking(ing)
            
            if chop and cook:
                self.process_chop_cook(controller, bot_id, state, holding, reserved_counters, blocked_tiles)
            elif cook:
                self.process_cook_only(controller, bot_id, state, holding, reserved_counters, blocked_tiles)
            elif chop:
                self.process_chop_only(controller, bot_id, state, holding, reserved_counters, blocked_tiles)
            else:
                state.task_stage = 13
        
        # State 13: Add to plate
        elif state.task_stage == 13:
            if state.plate_counter:
                px, py = state.plate_counter
                if self.move_to(controller, bot_id, px, py, blocked_tiles):
                    if controller.add_food_to_plate(bot_id, px, py):
                        if state.ingredients_needed:
                            state.ingredients_needed.pop(0)
                        state.task_stage = 10
        
        # State 20: Pickup plate
        elif state.task_stage == 20:
            if holding:
                if holding.get('type') == 'Plate':
                    state.task_stage = 21
                else:
                    state.task_stage = 99
            elif state.plate_counter:
                px, py = state.plate_counter
                if self.move_to(controller, bot_id, px, py, blocked_tiles):
                    if controller.pickup(bot_id, px, py):
                        state.task_stage = 21
        
        # State 21: Submit
        elif state.task_stage == 21:
            if self.move_to(controller, bot_id, ux, uy, blocked_tiles):
                controller.submit(bot_id, ux, uy)
                state.task_stage = 0
        
        # State 99: Trash
        elif state.task_stage == 99:
            if holding and self.trash_loc:
                tx, ty = self.trash_loc
                if self.move_to(controller, bot_id, tx, ty, blocked_tiles):
                    controller.trash(bot_id, tx, ty)
                    state.task_stage = 0
                    state.sub_state = 0
                    state.ingredients_needed = []
                    state.plate_counter = None
                    state.work_counter = None
                    state.current_order = None
            else:
                state.task_stage = 0
                state.sub_state = 0
                state.ingredients_needed = []
                state.plate_counter = None
                state.work_counter = None
                state.current_order = None

    def process_chop_only(self, controller, bot_id, state, holding, reserved, blocked):
        if not state.work_counter:
             # Exclude global reserved + own plate counter
            exclude = set(reserved)
            if state.plate_counter: exclude.add(state.plate_counter)
            
            bot_state = controller.get_bot_state(bot_id)
            state.work_counter = self.get_free_counter(controller, exclude, bot_pos=(bot_state['x'], bot_state['y']), blocked_tiles=blocked)
        
        if not state.work_counter:
            state.task_stage = 99
            return
        
        reserved.add(state.work_counter) # Lock it again
        wx, wy = state.work_counter
        
        if state.sub_state == 0:  # Place
            if holding:
                if holding.get('type') == 'Food':
                    if holding.get('chopped'):
                        state.task_stage = 13
                        return
                    if self.move_to(controller, bot_id, wx, wy, blocked):
                        if controller.place(bot_id, wx, wy):
                            state.sub_state = 1
                else:
                    state.task_stage = 99
            else:
                state.task_stage = 11
                
        elif state.sub_state == 1:  # Chop
            if self.move_to(controller, bot_id, wx, wy, blocked):
                tile = controller.get_tile(controller.get_team(), wx, wy)
                if tile and isinstance(tile.item, Food):
                    if tile.item.chopped:
                        state.sub_state = 2
                    else:
                        controller.chop(bot_id, wx, wy)
                else:
                    state.task_stage = 11
                    state.sub_state = 0
                    
        elif state.sub_state == 2:  # Pickup
            if holding:
                state.task_stage = 13
                state.sub_state = 0
            else:
                if self.move_to(controller, bot_id, wx, wy, blocked):
                    if controller.pickup(bot_id, wx, wy):
                        state.task_stage = 13
                        state.sub_state = 0

    def process_cook_only(self, controller, bot_id, state, holding, reserved, blocked):
        if state.sub_state == 0:  # Place on cooker
            if holding:
                if holding.get('type') == 'Food':
                    stage = holding.get('cooked_stage', 0)
                    if stage == 1:
                        state.task_stage = 13
                        return
                    elif stage == 2:
                        state.task_stage = 99
                        return
                    
                    cooker = self.get_free_cooker(controller)
                    if cooker:
                        kx, ky = cooker
                        if self.move_to(controller, bot_id, kx, ky, blocked):
                            if controller.place(bot_id, kx, ky):
                                state.sub_state = 1
                    elif self.cookers:
                        self.move_to(controller, bot_id, *self.cookers[0], blocked_tiles=blocked)
                else:
                    state.task_stage = 99
            else:
                # RECOVERY: Pick up from work_counter if verified
                if state.work_counter:
                     wx, wy = state.work_counter
                     tile = controller.get_tile(controller.get_team(), wx, wy)
                     if tile and isinstance(tile.item, Food):
                         if self.move_to(controller, bot_id, wx, wy, blocked):
                             controller.pickup(bot_id, wx, wy)
                         return

                for kx, ky in self.cookers:
                    tile = controller.get_tile(controller.get_team(), kx, ky)
                    if tile and isinstance(tile.item, Pan) and tile.item.food:
                        state.sub_state = 1
                        return
                state.task_stage = 11
                state.sub_state = 0
                
        elif state.sub_state == 1:  # Wait
            for kx, ky in self.cookers:
                tile = controller.get_tile(controller.get_team(), kx, ky)
                if tile and isinstance(tile.item, Pan) and tile.item.food:
                    stage = tile.item.food.cooked_stage
                    if stage == 1:
                        if self.move_to(controller, bot_id, kx, ky, blocked):
                            if controller.take_from_pan(bot_id, kx, ky):
                                state.task_stage = 13
                                state.sub_state = 0
                        return
                    elif stage == 2:
                        if self.move_to(controller, bot_id, kx, ky, blocked):
                            controller.take_from_pan(bot_id, kx, ky)
                        state.task_stage = 99
                        state.sub_state = 0
                        return
                    else:
                        self.move_to(controller, bot_id, kx, ky, blocked)
                        return
            state.task_stage = 11
            state.sub_state = 0

    def process_chop_cook(self, controller, bot_id, state, holding, reserved, blocked):
        if not state.work_counter:
            exclude = set(reserved)
            if state.plate_counter: exclude.add(state.plate_counter)
            bot_state = controller.get_bot_state(bot_id)
            state.work_counter = self.get_free_counter(controller, exclude, bot_pos=(bot_state['x'], bot_state['y']), blocked_tiles=blocked)
        
        if not state.work_counter:
            state.task_stage = 99
            return
        
        reserved.add(state.work_counter)
        wx, wy = state.work_counter
        
        if state.sub_state == 0:  # Place to chop
            if holding:
                if holding.get('type') == 'Food':
                    if holding.get('chopped'):
                        state.sub_state = 3
                        return
                    if self.move_to(controller, bot_id, wx, wy, blocked):
                        if controller.place(bot_id, wx, wy):
                            state.sub_state = 1
                else:
                    state.task_stage = 99
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
                    state.task_stage = 11
                    state.sub_state = 0
                    
        elif state.sub_state == 1:  # Chop
            if self.move_to(controller, bot_id, wx, wy, blocked):
                tile = controller.get_tile(controller.get_team(), wx, wy)
                if tile and isinstance(tile.item, Food):
                    if tile.item.chopped:
                        state.sub_state = 2
                    else:
                        controller.chop(bot_id, wx, wy)
                else:
                    state.task_stage = 11
                    state.sub_state = 0
                    
        elif state.sub_state == 2:  # Pickup chopped
            if holding:
                state.sub_state = 3
            else:
                if self.move_to(controller, bot_id, wx, wy, blocked):
                    controller.pickup(bot_id, wx, wy)
                    
        elif state.sub_state == 3:  # Place on cooker
            if holding:
                if holding.get('type') == 'Food':
                    stage = holding.get('cooked_stage', 0)
                    if stage == 1:
                        state.task_stage = 13
                        state.sub_state = 0
                        return
                    elif stage == 2:
                        state.task_stage = 99
                        state.sub_state = 0
                        return
                    
                    cooker = self.get_free_cooker(controller)
                    if cooker:
                        kx, ky = cooker
                        if self.move_to(controller, bot_id, kx, ky, blocked):
                            if controller.place(bot_id, kx, ky):
                                state.sub_state = 4
                    elif self.cookers:
                        self.move_to(controller, bot_id, *self.cookers[0], blocked_tiles=blocked)
                else:
                    state.task_stage = 99
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
                        if self.move_to(controller, bot_id, kx, ky, blocked):
                            if controller.take_from_pan(bot_id, kx, ky):
                                state.task_stage = 13
                                state.sub_state = 0
                        return
                    elif stage == 2:
                        if self.move_to(controller, bot_id, kx, ky, blocked):
                            controller.take_from_pan(bot_id, kx, ky)
                        state.task_stage = 99
                        state.sub_state = 0
                        return
                    else:
                        self.move_to(controller, bot_id, kx, ky, blocked)
                        return
            state.task_stage = 11
            state.sub_state = 0
