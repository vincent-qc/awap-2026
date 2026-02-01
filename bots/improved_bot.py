# Improved bot for AWAP 2026 Carnegie Cookoff.
# Optimized for speed and handles all edge cases.

import heapq
from collections import deque

from game_constants import FoodType, GameConstants, ShopCosts
from item import Food, Pan, Plate
from robot_controller import RobotController


class BotState:
    def __init__(self):
        self.task_stage = 0
        self.sub_state = 0
        self.ingredients_needed = []
        self.plate_counter = None
        self.work_counter = None
        self.current_order = None
        self.cooker_target = None
        self.pending_order_ids = []
        self.pending_items = []
        self.handoff_queue = []

        # Stuck detection
        self.stuck_counter = 0
        self.last_state = -1


class BotPlayer:
    HEURISTIC_EXPIRY = "expiry"
    HEURISTIC_INGREDIENTS = "ingredients"

    def __init__(self, map_copy):
        self.map = map_copy

        # Cached locations (set once)
        self.counters = None
        self.cookers = None
        self.shop_loc = None
        self.submit_loc = None
        self.trash_loc = None
        self.shops = None
        self.ingredient_sources = None
        self.landmarks = []
        self.boxes = None
        self.cooker_priority = []
        self.cooker_cluster_scores = {}
        self.cached_map = None
        self.dist_from_shop = None
        self.dist_from_submit = None
        self.dist_from_counters = None
        self.dist_from_cookers = None
        self.shared_handoff_counters = []
        self.primary_handoff_counter = None
        self.bot_dist_maps = {}
        self.handoff_happened_turn = False
        self.last_handoff_items = []
        self.debug_board = True
        self.helper_map_active = False

        # Bot States: map bot_id -> BotState
        self.bot_states = {}

    def cache_locations(self, controller):
        # Cache all locations once at start.
        if self.counters is not None:
            return

        m = controller.get_map(controller.get_team())
        self.cached_map = m
        self.counters = []
        self.cookers = []
        self.shops = []
        self.ingredient_sources = []
        self.landmarks = []
        self.boxes = []
        self.cooker_priority = []
        self.cooker_cluster_scores = {}
        self.dist_from_shop = None
        self.dist_from_submit = None
        self.dist_from_counters = None
        self.dist_from_cookers = None
        self.shared_handoff_counters = []
        self.primary_handoff_counter = None
        self.bot_dist_maps = {}

        for x in range(m.width):
            for y in range(m.height):
                name = m.tiles[x][y].tile_name
                if name == "COUNTER":
                    self.counters.append((x, y))
                elif name == "COOKER":
                    self.cookers.append((x, y))
                elif name == "SHOP" and self.shop_loc is None:
                    self.shop_loc = (x, y)
                if name == "SHOP":
                    self.shops.append((x, y))
                if name in ("SHOP", "BOX"):
                    self.ingredient_sources.append((x, y))
                elif name == "SUBMIT" and self.submit_loc is None:
                    self.submit_loc = (x, y)
                elif name == "TRASH" and self.trash_loc is None:
                    self.trash_loc = (x, y)
                elif name == "BOX":
                    self.boxes.append((x, y))

        # Landmark weighting: shop highest, counters/chop second, submit lowest.
        if self.shop_loc:
            self.landmarks.append((self.shop_loc, 5))
        if self.counters:
            for c in self.counters:
                self.landmarks.append((c, 2))
        if self.submit_loc:
            self.landmarks.append((self.submit_loc, 1))

        # Precompute shortest-path distance grids for landmark proximity.
        self.dist_from_shop = self._build_dist_map(
            self._adjacent_walkables_multi(self.shops))
        self.dist_from_submit = self._build_dist_map(
            self._adjacent_walkables(self.submit_loc))
        self.dist_from_counters = self._build_dist_map(
            self._adjacent_walkables_multi(self.counters))
        self.dist_from_cookers = self._build_dist_map(
            self._adjacent_walkables_multi(self.cookers))

        # Precompute cooker priority by shortest path closeness to shop + counters + submit.
        self.cooker_cluster_scores = {}
        for cooker in self.cookers:
            score = 0
            if self.dist_from_shop is not None:
                score += 5 * \
                    self._distance_to_tile(self.dist_from_shop, cooker)
            if self.dist_from_counters is not None:
                score += 2 * \
                    self._distance_to_tile(self.dist_from_counters, cooker)
            if self.dist_from_submit is not None:
                score += 1 * \
                    self._distance_to_tile(self.dist_from_submit, cooker)
            self.cooker_cluster_scores[cooker] = score
        self.cooker_priority = sorted(
            self.cookers,
            key=lambda c: (self.cooker_cluster_scores.get(c, 0), c))

        # Shared handoff counters reachable from both shops and cookers.
        self.shared_handoff_counters = []
        if self.dist_from_shop and self.dist_from_cookers:
            for c in self.counters:
                if (self._distance_to_tile(self.dist_from_shop, c) < 9999 and
                        self._distance_to_tile(self.dist_from_cookers, c) < 9999):
                    self.shared_handoff_counters.append(c)
        if self.shared_handoff_counters:
            self.primary_handoff_counter = self.shared_handoff_counters[0]

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

        step = self.get_next_step_astar(
            controller, (bx, by), target_x, target_y, blocked=blocked_tiles)
        if step and step != (0, 0):
            controller.move(bot_id, step[0], step[1])
        return False

    def get_free_counter(self, controller, exclude=None, bot_pos=None, blocked_tiles=None, anchor_pos=None):
        # Find empty counter that is reachable.
        if exclude is None:
            exclude = set()

        def dist(a, b):
            return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

        # Prefer nearest to bot; use anchor as tie-breaker to keep a tight cluster.
        counters = self.counters
        if counters:
            if bot_pos:
                counters = sorted(
                    counters,
                    key=lambda c: (dist(c, bot_pos), dist(c, anchor_pos) if anchor_pos else 0))
            elif anchor_pos:
                counters = sorted(counters, key=lambda c: dist(c, anchor_pos))

        # First try to find an empty reachable counter
        for cx, cy in counters:
            if (cx, cy) in exclude:
                continue
            tile = controller.get_tile(controller.get_team(), cx, cy)
            if tile and tile.item is None:
                # Check if reachable (can find path)
                if bot_pos:
                    step = self.get_next_step_astar(
                        controller, bot_pos, cx, cy, blocked=blocked_tiles)
                    if step is not None:
                        return (cx, cy)
                else:
                    return (cx, cy)

        # Fallback: any counter not in exclude, preferring reachable ones
        for cx, cy in counters:
            if (cx, cy) not in exclude:
                if bot_pos:
                    step = self.get_next_step_astar(
                        controller, bot_pos, cx, cy, blocked=blocked_tiles)
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

    def _chebyshev(self, a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def _tile_base_char(self, tile_name):
        return {
            "WALL": "#",
            "FLOOR": ".",
            "COUNTER": "C",
            "COOKER": "K",
            "SINK": "S",
            "SINKTABLE": "T",
            "TRASH": "R",
            "SUBMIT": "U",
            "SHOP": "$",
            "BOX": "B",
        }.get(tile_name, "?")

    def _tile_item_char(self, tile):
        item = getattr(tile, "item", None)
        if isinstance(item, Food):
            return item.food_name[0].lower()
        if isinstance(item, Plate):
            return "P"
        if isinstance(item, Pan):
            if item.food:
                return item.food.food_name[0].upper()
            return "k"
        return None

    def debug_print_board(self, controller):
        if not self.cached_map:
            return
        m = self.cached_map
        rows = []
        for y in range(m.height - 1, -1, -1):
            row = []
            for x in range(m.width):
                tile = controller.get_tile(controller.get_team(), x, y)
                base = self._tile_base_char(tile.tile_name) if tile else "?"
                item = self._tile_item_char(tile)
                row.append(item if item else base)
            rows.append("".join(row))
        print("[BOARD]")
        for r in rows:
            print(r)

    def debug_print_bots(self, controller, bots):
        parts = []
        for bid in bots:
            info = controller.get_bot_state(bid)
            holding = info.get('holding')
            held = None
            if holding:
                if holding.get('type') == 'Food':
                    held = holding.get('food_name')
                elif holding.get('type') == 'Plate':
                    held = "Plate"
                else:
                    held = holding.get('type')
            parts.append(f"B{bid}@({info['x']},{info['y']}) hold={held}")
        print("[BOTS] " + " | ".join(parts))

    def _adjacent_walkables(self, loc):
        if not loc or not self.cached_map:
            return []
        x, y = loc
        m = self.cached_map
        out = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if not m.in_bounds(nx, ny):
                    continue
                if m.is_tile_walkable(nx, ny):
                    out.append((nx, ny))
        return out

    def _adjacent_walkables_multi(self, locs):
        out = []
        for loc in locs or []:
            out.extend(self._adjacent_walkables(loc))
        return out

    def _build_dist_map(self, sources):
        if not self.cached_map or not sources:
            return None
        m = self.cached_map
        dist = [[None for _ in range(m.height)] for _ in range(m.width)]
        q = deque()
        for sx, sy in sources:
            if not m.in_bounds(sx, sy) or not m.is_tile_walkable(sx, sy):
                continue
            if dist[sx][sy] is None:
                dist[sx][sy] = 0
                q.append((sx, sy))
        while q:
            cx, cy = q.popleft()
            base = dist[cx][cy]
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if not m.in_bounds(nx, ny) or not m.is_tile_walkable(nx, ny):
                        continue
                    if dist[nx][ny] is None:
                        dist[nx][ny] = base + 1
                        q.append((nx, ny))
        return dist

    def _distance_to_tile(self, dist_map, tile_pos):
        if not dist_map or not self.cached_map or not tile_pos:
            return 9999
        m = self.cached_map
        x, y = tile_pos
        if m.in_bounds(x, y) and m.is_tile_walkable(x, y):
            d = dist_map[x][y]
            return d if d is not None else 9999
        best = 9999
        for nx, ny in self._adjacent_walkables(tile_pos):
            d = dist_map[nx][ny]
            if d is not None and d < best:
                best = d
        return best

    def choose_shop_for_bot(self, dist_map, bot_pos=None):
        if not self.shops:
            return self.shop_loc
        best = None
        best_dist = 9999
        for sx, sy in self.shops:
            d = self._distance_to_tile(
                dist_map, (sx, sy)) if dist_map else self._chebyshev(bot_pos, (sx, sy))
            if d < best_dist:
                best_dist = d
                best = (sx, sy)
        return best or self.shop_loc

    def _has_access(self, dist_map, target_pos):
        if not dist_map or not target_pos:
            return False
        return self._distance_to_tile(dist_map, target_pos) < 9999

    def _is_helper_mode(self, dist_map):
        # Helper mode: bot cannot reach submit on this side of the map.
        return dist_map is not None and not self._has_access(dist_map, self.submit_loc)

    def _others_can_access_cookers(self, other_dist_maps):
        for dm in other_dist_maps or []:
            if any(self._has_access(dm, c) for c in self.cookers or []):
                return True
        return False

    def _can_access_any_cooker(self, dist_map):
        return any(self._has_access(dist_map, c) for c in self.cookers or [])

    def _can_access_any_shop(self, dist_map):
        if not dist_map or not self.shops:
            return False
        return any(self._has_access(dist_map, s) for s in self.shops)

    def _find_handoff_food(self, controller, dist_map, require_cookable=True, require_raw=False, require_cooked=False, item_name=None):
        counters = self.shared_handoff_counters or self.counters or []
        for cx, cy in counters:
            if dist_map and not self._has_access(dist_map, (cx, cy)):
                continue
            tile = controller.get_tile(controller.get_team(), cx, cy)
            if tile and isinstance(tile.item, Food):
                if require_cookable and not tile.item.can_cook:
                    continue
                if require_raw and tile.item.cooked_stage != 0:
                    continue
                if require_cooked and tile.item.cooked_stage != 1:
                    continue
                if item_name and tile.item.food_name != item_name:
                    continue
                return (cx, cy)
        return None

    def _find_handoff_counter(self, controller, dist_map, other_dist_maps):
        if not dist_map:
            return None
        best = None
        best_dist = 9999
        counters = self.shared_handoff_counters or self.counters or []
        shared = []
        for cx, cy in counters:
            tile = controller.get_tile(controller.get_team(), cx, cy)
            if not tile or tile.item is not None:
                continue
            if self._distance_to_tile(dist_map, (cx, cy)) >= 9999:
                continue
            if any(self._distance_to_tile(odm, (cx, cy)) < 9999 for odm in other_dist_maps):
                shared.append((cx, cy))

        # Prefer counters reachable by both bots; otherwise fall back to any reachable.
        candidates = shared if shared else [
            c for c in counters if self._distance_to_tile(dist_map, c) < 9999]
        for cx, cy in candidates:
            tile = controller.get_tile(controller.get_team(), cx, cy)
            if not tile or tile.item is not None:
                continue
            d = self._distance_to_tile(dist_map, (cx, cy))
            if d < best_dist:
                best_dist = d
                best = (cx, cy)
        return best

    def _attempt_handoff(self, controller, bot_id, state, dist_map, other_dist_maps, blocked_tiles, order_id_override=None, item_name_override=None):
        info = controller.get_bot_state(bot_id)
        holding = info.get('holding')
        if not holding:
            return False
        counter = self._find_handoff_counter(
            controller, dist_map, other_dist_maps)
        if not counter:
            if self.debug_board:
                held_name = holding.get('food_name') if holding.get(
                    'type') == 'Food' else holding.get('type')
                print(
                    f"[Bot {bot_id}] Handoff skipped: no counter for {held_name}")
            return False
        cx, cy = counter
        if self.move_to(controller, bot_id, cx, cy, blocked_tiles):
            if controller.place(bot_id, cx, cy):
                held_type = holding.get('type')
                held_name = holding.get(
                    'food_name') if held_type == 'Food' else None
                handoff_item = item_name_override or held_name
                order_id = order_id_override
                print(
                    f"[Bot {bot_id}] Handoff placed at ({cx},{cy}) item={held_type}:{held_name}")
                if state.current_order:
                    order_id = state.current_order['order_id']
                    self.add_pending_order(
                        state, order_id, held_name)
                    state.current_order = None
                if handoff_item:
                    self.last_handoff_items.append(handoff_item)
                if order_id is not None:
                    for st in self.bot_states.values():
                        if st is not state:
                            st.handoff_queue.append((order_id, handoff_item))
                # Reset to replan after handoff.
                state.task_stage = 0
                state.sub_state = 0
                state.ingredients_needed = []
                state.work_counter = None
                state.plate_counter = None
                self.handoff_happened_turn = True
                return True
            if self.debug_board:
                tile = controller.get_tile(controller.get_team(), cx, cy)
                tile_item = getattr(tile, "item", None)
                if isinstance(tile_item, Food):
                    item_desc = f"{tile_item.food_name}:{tile_item.cooked_stage}"
                else:
                    item_desc = type(
                        tile_item).__name__ if tile_item else "None"
                print(
                    f"[Bot {bot_id}] Handoff place failed at ({cx},{cy}) tile_item={item_desc}")
        elif self.debug_board:
            print(
                f"[Bot {bot_id}] Handoff move toward ({cx},{cy}) blocked")
        return False

    def _location_score(self, pos):
        # Lower score is better (more central to weighted landmarks).
        if not self.landmarks:
            return 0
        score = 0
        for loc, weight in self.landmarks:
            if loc == self.shop_loc and self.dist_from_shop is not None:
                dist = self._distance_to_tile(self.dist_from_shop, pos)
            elif loc == self.submit_loc and self.dist_from_submit is not None:
                dist = self._distance_to_tile(self.dist_from_submit, pos)
            else:
                dist = self._distance_to_tile(self.dist_from_counters, pos)
            score += weight * dist
        return score

    def _location_score(self, pos):
        # Lower score is better (more central to weighted landmarks).
        if not self.landmarks:
            return 0
        return sum(weight * self._chebyshev(pos, loc) for loc, weight in self.landmarks)

    def choose_cooker(self, controller, anchor_pos=None, bot_pos=None, blocked_tiles=None, want_food=False):
        # Pick a cooker in the closest cluster; tie-break by anchor/bot.
        best = None
        best_dist = (float('inf'), float('inf'))

        for kx, ky in self.cooker_priority:
            tile = controller.get_tile(controller.get_team(), kx, ky)
            if not tile or not isinstance(tile.item, Pan):
                continue
            if want_food and tile.item.food is None:
                continue
            if not want_food and tile.item.food is not None:
                continue

            if bot_pos:
                step = self.get_next_step_astar(
                    controller, bot_pos, kx, ky, blocked=blocked_tiles)
                if step is None:
                    continue

            # Fully bias toward cluster closeness; use anchor/bot only to break ties.
            dist_source = self.cooker_cluster_scores.get((kx, ky), 0)
            tie_break = 0
            if anchor_pos:
                tie_break = self._chebyshev((kx, ky), anchor_pos)
            elif bot_pos:
                tie_break = self._chebyshev((kx, ky), bot_pos)
            dist = (dist_source, tie_break)

            if dist < best_dist:
                best_dist = dist
                best = (kx, ky)

        return best

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
                return (cx, cy), False  # False = not cooking

        # Check cookers for cooking version
        for kx, ky in self.cookers:
            tile = controller.get_tile(controller.get_team(), kx, ky)
            if tile and isinstance(tile.item, Pan) and tile.item.food and tile.item.food.food_name == ingredient:
                return (kx, ky), True  # True = is cooking
        return None, False

    def find_box_with_ingredient(self, controller, ingredient):
        # Only use boxes as a fallback source.
        for bx, by in self.boxes or []:
            tile = controller.get_tile(controller.get_team(), bx, by)
            if not tile or tile.item is None:
                continue
            if isinstance(tile.item, Food) and tile.item.food_name == ingredient:
                return (bx, by)
        return None

    def get_world_inventory(self, controller):
        inventory = []
        # Counters and plates on counters
        for cx, cy in self.counters or []:
            tile = controller.get_tile(controller.get_team(), cx, cy)
            if not tile:
                continue
            if isinstance(tile.item, Food):
                inventory.append(tile.item.food_name)
            elif isinstance(tile.item, Plate):
                for item in tile.item.food:
                    inventory.append(item.food_name)

        # Cookers (pans)
        for kx, ky in self.cookers or []:
            tile = controller.get_tile(controller.get_team(), kx, ky)
            if tile and isinstance(tile.item, Pan) and tile.item.food:
                inventory.append(tile.item.food.food_name)

        return inventory

    def find_plate_on_counter(self, controller):
        for cx, cy in self.counters or []:
            tile = controller.get_tile(controller.get_team(), cx, cy)
            if tile and isinstance(tile.item, Plate):
                return (cx, cy)
        return None

    def get_plated_inventory(self, controller):
        plated = []
        # Plates on counters
        for cx, cy in self.counters or []:
            tile = controller.get_tile(controller.get_team(), cx, cy)
            if tile and isinstance(tile.item, Plate):
                for item in tile.item.food:
                    plated.append(item.food_name)

        # Plates in hand
        for bid in controller.get_team_bot_ids(controller.get_team()):
            info = controller.get_bot_state(bid)
            holding = info.get('holding')
            if holding and holding.get('type') == 'Plate':
                for item in holding.get('food', []):
                    plated.append(item['food_name'])
        return plated

    def recompute_remaining_for_order(self, controller, state, order):
        required = list(order['required'])
        plated = self.get_plated_inventory(controller)
        for item in plated:
            if item in required:
                required.remove(item)
        pending_first = []
        if state.pending_items:
            for item in state.pending_items:
                if item in required:
                    pending_first.append(item)
                    required.remove(item)
        ordered = pending_first + [i for i in required if not self.needs_cooking(i)] + \
            [i for i in required if self.needs_cooking(i)]
        state.current_order = order
        state.ingredients_needed = ordered
        state.task_stage = 20 if not state.ingredients_needed else 10

    def add_pending_order(self, state, order_id, item_name=None):
        if order_id is None:
            return
        if order_id not in state.pending_order_ids:
            state.pending_order_ids.append(order_id)
        if item_name:
            state.pending_items.append(item_name)

    def clear_pending_order(self, order_id):
        if order_id is None:
            return
        for st in self.bot_states.values():
            if order_id in st.pending_order_ids:
                st.pending_order_ids = [
                    oid for oid in st.pending_order_ids if oid != order_id]
            if st.pending_items:
                st.pending_items = []

    def resume_pending_order(self, controller, state, claimed_orders, turn):
        if not state.pending_order_ids:
            return False
        orders = controller.get_orders(controller.get_team())
        pending_id = state.pending_order_ids[0]
        pending = next(
            (o for o in orders if o['order_id'] == pending_id), None)
        if not pending or not pending['is_active'] or pending['expires_turn'] <= turn:
            state.pending_order_ids = state.pending_order_ids[1:]
            state.pending_items = []
            return False

        required = list(pending['required'])
        plated = self.get_plated_inventory(controller)
        for item in plated:
            if item in required:
                required.remove(item)

        pending_first = []
        if state.pending_items:
            for item in state.pending_items:
                if item in required:
                    pending_first.append(item)
                    required.remove(item)

        state.current_order = pending
        claimed_orders.add(pending['order_id'])
        state.ingredients_needed = pending_first + \
            [i for i in required if not self.needs_cooking(i)] + \
            [i for i in required if self.needs_cooking(i)]
        state.task_stage = 1
        return True

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
            tile = controller.get_tile(
                controller.get_team(), *state.plate_counter)
            if tile and tile.item:
                if isinstance(tile.item, Food):
                    inventory.append(tile.item.food_name)
                elif isinstance(tile.item, Plate):
                    for item in tile.item.food:
                        inventory.append(item.food_name)

        # 3. Work counter
        if state.work_counter:
            tile = controller.get_tile(
                controller.get_team(), *state.work_counter)
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
        matching_active = next(
            (o for o in orders if o['order_id'] == current['order_id']), None)

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

            print(
                f"[Bot {bot_id}] Has inventory {inventory}, trying to reuse...")

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
                        score += 5  # high value for reuse
                        needed.remove(inv_item)

                # Tie-breaker: prioritization logic (time, value) from normal selection
                # Prefer more complex recipes if equal reuse
                score += len(o['required'])

                if score > best_match_score:
                    best_match_score = score
                    best_match = o

            if best_match and best_match_score > 0:
                print(
                    f"[Bot {bot_id}] Switched to Order {best_match['order_id']} to reuse ingredients")

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
                    tile = controller.get_tile(
                        controller.get_team(), *state.plate_counter)
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
                print(
                    f"[Bot {bot_id}] No matching order for reuse. Resetting.")
                state.task_stage = 99
                return True

        return False

    def _dist_from_grid(self, dist_map, pos):
        if not dist_map or not pos or not self.cached_map:
            return 9999
        x, y = pos
        if self.cached_map.in_bounds(x, y) and self.cached_map.is_tile_walkable(x, y):
            d = dist_map[x][y]
            return d if d is not None else 9999
        return self._distance_to_tile(dist_map, pos)

    def _min_dist_to_cookers(self, dist_map):
        if not dist_map or not self.cookers:
            return 9999
        return min(self._distance_to_tile(dist_map, c) for c in self.cookers)

    def _min_dist_to_counters(self, dist_map):
        if not dist_map or not self.counters:
            return 9999
        return min(self._distance_to_tile(dist_map, c) for c in self.counters)

    def _min_handoff_shop_to_cooker(self):
        if not self.dist_from_shop or not self.dist_from_cookers or not self.shared_handoff_counters:
            return 9999
        best = 9999
        for c in self.shared_handoff_counters:
            d = self._distance_to_tile(
                self.dist_from_shop, c) + self._distance_to_tile(self.dist_from_cookers, c)
            if d < best:
                best = d
        return best

    def simulate_order_time(self, controller, bot_id, state, order, turn):
        # Lower-bound time estimate to complete the order from current bot state.
        time_left = order['expires_turn'] - turn
        if time_left <= 0:
            return float('inf')

        info = controller.get_bot_state(bot_id)
        bot_pos = (info['x'], info['y'])

        required = list(order['required'])
        inventory = self.get_bot_inventory(controller, bot_id, state)
        for item in inventory:
            if item in required:
                required.remove(item)

        if not required:
            return self._dist_from_grid(self.dist_from_submit, bot_pos) + 1

        dist_bot_to_shop = self._dist_from_grid(self.dist_from_shop, bot_pos)
        dist_shop_to_counter = self._min_dist_to_counters(self.dist_from_shop)
        dist_shop_to_cooker = self._min_dist_to_cookers(self.dist_from_shop)
        dist_counter_to_cooker = self._min_dist_to_cookers(
            self.dist_from_counters)
        dist_shop_to_submit = self._distance_to_tile(
            self.dist_from_shop, self.submit_loc)
        handoff_shop_to_cooker = self._min_handoff_shop_to_cooker()

        if dist_shop_to_cooker >= 9999 and handoff_shop_to_cooker < 9999:
            dist_shop_to_cooker = handoff_shop_to_cooker
        if dist_counter_to_cooker >= 9999 and handoff_shop_to_cooker < 9999:
            dist_counter_to_cooker = handoff_shop_to_cooker

        if min(dist_bot_to_shop, dist_shop_to_counter, dist_shop_to_cooker,
               dist_counter_to_cooker, dist_shop_to_submit) >= 9999:
            return float('inf')

        BUY = 1
        PLACE = 1
        PICKUP = 1
        CHOP = 1
        TAKE = 1
        SUBMIT = 1
        COOK_TIME = GameConstants.COOK_PROGRESS

        total = dist_bot_to_shop
        for ing in required:
            chop = self.needs_chopping(ing)
            cook = self.needs_cooking(ing)
            if chop and cook:
                total += dist_shop_to_counter + dist_counter_to_cooker
                total += BUY + PLACE + CHOP + PICKUP + PLACE + TAKE + COOK_TIME
            elif cook:
                total += dist_shop_to_cooker
                total += BUY + PLACE + TAKE + COOK_TIME
            elif chop:
                total += dist_shop_to_counter
                total += BUY + PLACE + CHOP + PICKUP
            else:
                total += dist_shop_to_counter
                total += BUY + PLACE

        total += dist_shop_to_submit + SUBMIT
        return total

    def _order_is_doable(self, controller, bot_id, state, order, turn):
        my_dist = self.bot_dist_maps.get(bot_id)
        if my_dist and not self._has_access(my_dist, self.submit_loc):
            return False
        time_left = order['expires_turn'] - turn
        if time_left <= 0:
            return False
        sim_time = self.simulate_order_time(
            controller, bot_id, state, order, turn)
        return sim_time <= time_left

    def calculate_order_heuristic(self, controller, bot_id, state, order, turn, team_money, heuristic=None):
        # Returns -float('inf') if impossible to complete in time.
        if not self._order_is_doable(controller, bot_id, state, order, turn):
            return -float('inf')

        ingredient_count = len(order['required'])
        time_left = order['expires_turn'] - turn

        if heuristic == self.HEURISTIC_EXPIRY:
            # Primary key: earliest expiration.
            # Secondary key: fewer ingredients.
            return (-time_left * 1_000_000) - ingredient_count

        # Default: strictly fewer ingredients, time as tie-breaker.
        return (-ingredient_count * 1_000_000) - time_left

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
                debug_info.append(
                    f"B{bid}:({info['x']},{info['y']})[S{state_val}]")
        print(f"[{team} Turn {turn}] {' | '.join(debug_info)}")
        if self.debug_board:
            self.debug_print_bots(controller, bots)
            self.debug_print_board(controller)

        # Reset per-turn handoff flag.
        self.handoff_happened_turn = False
        self.last_handoff_items = []

        # Compute reachability per bot for handoff logic.
        self.bot_dist_maps = {}
        for bid in bots:
            info = controller.get_bot_state(bid)
            start = (info['x'], info['y'])
            self.bot_dist_maps[bid] = self._build_dist_map([start])
        # Helper map if any bot cannot reach submit.
        self.helper_map_active = any(
            dm and not self._has_access(dm, self.submit_loc)
            for dm in self.bot_dist_maps.values())

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
            if bot_id not in self.bot_states:
                continue

            # Exclude this bot from blocked tiles logic
            other_bots_locs = {
                pos for bid, pos in all_bot_positions.items() if bid != bot_id}

            # Exclude this bot's own resources from exclusion list
            my_state = self.bot_states[bot_id]
            my_reserved = set()
            if my_state.plate_counter:
                my_reserved.add(my_state.plate_counter)
            if my_state.work_counter:
                my_reserved.add(my_state.work_counter)

            others_reserved_counters = reserved_counters - my_reserved

            # Identify which bot is which
            is_bot_1 = (bot_id == team_bots[0])
            is_bot_2 = (len(team_bots) >= 2 and bot_id == team_bots[1])

            if is_bot_1:
                self.run_bot_1_behavior(
                    controller, bot_id, my_state, others_reserved_counters, claimed_orders, other_bots_locs,
                    order_heuristic=self.HEURISTIC_EXPIRY)
            elif is_bot_2:
                self.run_bot_2_behavior(
                    controller, bot_id, my_state, others_reserved_counters, claimed_orders, other_bots_locs,
                    order_heuristic=self.HEURISTIC_INGREDIENTS)
            else:
                # Default behavior for extra bots
                self.run_standard_logic(
                    controller, bot_id, my_state, others_reserved_counters, claimed_orders, other_bots_locs,
                    order_heuristic=self.HEURISTIC_INGREDIENTS)

    # Bot 1 Behavior

    def run_bot_1_behavior(self, controller, bot_id, state, reserved_counters, claimed_orders, blocked_tiles, order_heuristic):
        # Bot 1 follows standard logic (expiry-first ordering).
        self.run_standard_logic(controller, bot_id, state,
                                reserved_counters, claimed_orders, blocked_tiles, order_heuristic)

    # Bot 2 Behavior

    def run_bot_2_behavior(self, controller, bot_id, state, reserved_counters, claimed_orders, blocked_tiles, order_heuristic):
        # Bot 2 has smart recovery logic for expired orders.
        if state.current_order:
            if self.handle_expired_order(controller, bot_id, state, claimed_orders):
                pass  # State updated by recovery

        self.run_standard_logic(controller, bot_id, state,
                                reserved_counters, claimed_orders, blocked_tiles, order_heuristic)

    # Shared Logic

    def run_standard_logic(self, controller, bot_id, state, reserved_counters, claimed_orders, blocked_tiles, order_heuristic):
        info = controller.get_bot_state(bot_id)
        holding = info.get('holding')
        bx, by = info['x'], info['y']

        my_dist = self.bot_dist_maps.get(bot_id)
        is_helper = self._is_helper_mode(my_dist)
        if not self.helper_map_active:
            self.run_normal_logic(
                controller, bot_id, state, reserved_counters, claimed_orders, blocked_tiles, order_heuristic)
            return

        other_dist_maps = [
            dm for bid, dm in self.bot_dist_maps.items() if bid != bot_id]
        shop_pos = self.choose_shop_for_bot(my_dist, bot_pos=(bx, by))
        if not shop_pos:
            return
        sx, sy = shop_pos
        ux, uy = self.submit_loc

        # Helper-side behavior on split maps: cook and handoff without ordering.
        if my_dist and not self._has_access(my_dist, (ux, uy)):
            if self.debug_board:
                h_desc = None
                if holding:
                    if holding.get('type') == 'Food':
                        h_desc = f"{holding.get('food_name')}:{holding.get('cooked_stage', 0)}"
                    else:
                        h_desc = holding.get('type')
                print(
                    f"[Bot {bot_id}] Helper mode holding={h_desc} queue={state.handoff_queue}")
            if state.handoff_queue:
                order_id, target_item = state.handoff_queue[0]
                if holding and holding.get('type') == 'Food':
                    held_name = holding.get('food_name')
                    if target_item and held_name != target_item:
                        # Try to rotate queue to match held item if possible.
                        match_idx = next(
                            (i for i, (_, item) in enumerate(
                                state.handoff_queue) if item == held_name),
                            None)
                        if match_idx is not None and match_idx != 0:
                            state.handoff_queue = state.handoff_queue[match_idx:] + \
                                state.handoff_queue[:match_idx]
                            order_id, target_item = state.handoff_queue[0]
                            if self.debug_board:
                                print(
                                    f"[Bot {bot_id}] Helper queue rotated for {held_name}")
                    if target_item and held_name != target_item:
                        # Put it back to avoid contaminating queue.
                        if self._attempt_handoff(controller, bot_id, state, my_dist, other_dist_maps, blocked_tiles,
                                                 order_id_override=order_id, item_name_override=held_name):
                            return
                    stage = holding.get('cooked_stage', 0)
                    if stage == 1:
                        if self._attempt_handoff(controller, bot_id, state, my_dist, other_dist_maps, blocked_tiles,
                                                 order_id_override=order_id, item_name_override=target_item):
                            state.handoff_queue.pop(0)
                            return
                    elif stage == 2:
                        state.task_stage = 99
                        return
                    else:
                        cooker = self.choose_cooker(
                            controller,
                            anchor_pos=state.work_counter or state.plate_counter,
                            bot_pos=(bx, by),
                            blocked_tiles=blocked_tiles,
                            want_food=False)
                        if cooker and self.move_to(controller, bot_id, cooker[0], cooker[1], blocked_tiles):
                            controller.place(bot_id, cooker[0], cooker[1])
                        return
                else:
                    handoff_food = self._find_handoff_food(
                        controller, my_dist, require_cookable=True, require_raw=True, item_name=target_item)
                    if handoff_food and self.move_to(controller, bot_id, handoff_food[0], handoff_food[1], blocked_tiles):
                        if controller.pickup(bot_id, handoff_food[0], handoff_food[1]):
                            return
            if holding and holding.get('type') == 'Food':
                held_name = holding.get('food_name')
                stage = holding.get('cooked_stage', 0)
                if stage == 1:
                    if self._attempt_handoff(controller, bot_id, state, my_dist, other_dist_maps, blocked_tiles):
                        return
                elif stage == 2:
                    state.task_stage = 99
                    return
                else:
                    if self.needs_chopping(held_name) and self.needs_cooking(held_name):
                        if state.task_stage != 12 or not state.ingredients_needed or state.ingredients_needed[0] != held_name:
                            state.ingredients_needed = [held_name]
                            state.sub_state = 0
                            state.task_stage = 12
                        self.process_chop_cook(
                            controller, bot_id, state, holding, reserved_counters, blocked_tiles, held_name,
                            my_dist=my_dist, other_dist_maps=other_dist_maps, helper_mode=is_helper)
                        return
                    if self.needs_cooking(held_name):
                        if state.task_stage != 12 or not state.ingredients_needed or state.ingredients_needed[0] != held_name:
                            state.ingredients_needed = [held_name]
                            state.sub_state = 0
                            state.task_stage = 12
                        self.process_cook_only(
                            controller, bot_id, state, holding, reserved_counters, blocked_tiles, held_name,
                            my_dist=my_dist, other_dist_maps=other_dist_maps, helper_mode=is_helper)
                        return
                    if self.needs_chopping(held_name):
                        if state.task_stage != 12 or not state.ingredients_needed or state.ingredients_needed[0] != held_name:
                            state.ingredients_needed = [held_name]
                            state.sub_state = 0
                            state.task_stage = 12
                        self.process_chop_only(
                            controller, bot_id, state, holding, reserved_counters, blocked_tiles,
                            my_dist=my_dist, other_dist_maps=other_dist_maps, helper_mode=is_helper)
                        return
                    if self._attempt_handoff(controller, bot_id, state, my_dist, other_dist_maps, blocked_tiles):
                        return
                    if self.primary_handoff_counter:
                        hx, hy = self.primary_handoff_counter
                        self.move_to(controller, bot_id, hx, hy, blocked_tiles)
                    return
            else:
                # If cooked food is ready on a cooker, retrieve it.
                for kx, ky in self.cookers or []:
                    if my_dist and not self._has_access(my_dist, (kx, ky)):
                        continue
                    tile = controller.get_tile(controller.get_team(), kx, ky)
                    if tile and isinstance(tile.item, Pan) and tile.item.food:
                        if tile.item.food.cooked_stage == 1:
                            if self.move_to(controller, bot_id, kx, ky, blocked_tiles):
                                controller.take_from_pan(bot_id, kx, ky)
                            return
                # If cooking is in progress, stay near the cooker instead of re-picking.
                for kx, ky in self.cookers or []:
                    if my_dist and not self._has_access(my_dist, (kx, ky)):
                        continue
                    tile = controller.get_tile(controller.get_team(), kx, ky)
                    if tile and isinstance(tile.item, Pan) and tile.item.food:
                        if tile.item.food.cooked_stage == 0:
                            self.move_to(controller, bot_id,
                                         kx, ky, blocked_tiles)
                            return
                # Otherwise, look for new handoff items to cook.
                handoff_food = self._find_handoff_food(
                    controller, my_dist, require_cookable=True, require_raw=True)
                if handoff_food:
                    print(f"[Bot {bot_id}] Helper pickup at {handoff_food}")
                if handoff_food and self.move_to(controller, bot_id, handoff_food[0], handoff_food[1], blocked_tiles):
                    if controller.pickup(bot_id, handoff_food[0], handoff_food[1]):
                        info2 = controller.get_bot_state(bot_id)
                        holding2 = info2.get('holding')
                        if holding2 and holding2.get('type') == 'Food':
                            state.ingredients_needed = [
                                holding2.get('food_name')]
                        else:
                            state.ingredients_needed = []
                        state.sub_state = 0
                        state.task_stage = 12
                return

        # If a handoff happened this turn, re-evaluate order progress immediately.
        if self.handoff_happened_turn and my_dist and self._has_access(my_dist, (ux, uy)):
            if state.current_order:
                self.recompute_remaining_for_order(
                    controller, state, state.current_order)
                return
            if state.pending_order_ids:
                if self.resume_pending_order(controller, state, claimed_orders, controller.get_turn()):
                    return
            # Try to complete any active order from available inventory.
            orders = controller.get_orders(controller.get_team())
            inventory = self.get_world_inventory(controller)
            for o in orders:
                if not o['is_active'] or o['expires_turn'] <= controller.get_turn():
                    continue
                if o['order_id'] in claimed_orders:
                    continue
                required = list(o['required'])
                for item in inventory:
                    if item in required:
                        required.remove(item)
                if not required:
                    self.recompute_remaining_for_order(controller, state, o)
                    claimed_orders.add(o['order_id'])
                    return

        # Stuck detection
        if state.task_stage == state.last_state:
            state.stuck_counter += 1
            if state.stuck_counter > 10:
                if state.task_stage == 12 and holding and holding.get('type') == 'Food' and holding.get('cooked_stage', 0) == 0:
                    if self._attempt_handoff(controller, bot_id, state, my_dist, other_dist_maps, blocked_tiles):
                        state.stuck_counter = 0
                        return
                print(
                    f"[Bot {bot_id}] STUCK in state {state.task_stage}, forcing random move")
                import random
                dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
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

            # Helper mode for split maps: pick up cookable handoff items if no shop access.
            if not holding and my_dist and not self._can_access_any_shop(my_dist):
                handoff_food = self._find_handoff_food(
                    controller, my_dist, require_cookable=True, require_raw=True)
                if handoff_food:
                    hx, hy = handoff_food
                    if self.move_to(controller, bot_id, hx, hy, blocked_tiles):
                        if controller.pickup(bot_id, hx, hy):
                            info2 = controller.get_bot_state(bot_id)
                            holding2 = info2.get('holding')
                            if holding2 and holding2.get('type') == 'Food':
                                state.ingredients_needed = [
                                    holding2.get('food_name')]
                            else:
                                state.ingredients_needed = []
                            state.task_stage = 12
                            return

            if state.pending_order_ids:
                if self.resume_pending_order(controller, state, claimed_orders, controller.get_turn()):
                    return
                if len(state.pending_order_ids) >= 3:
                    # Stop taking new orders until pending ones are cleared.
                    return

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

                    score = self.calculate_order_heuristic(
                        controller, bot_id, state, o, turn, team_money, order_heuristic)

                    if score > best_score:
                        best_score = score
                        best = o

            if best:
                state.current_order = best
                # Claim it immediately for subsequent bots in this turn
                claimed_orders.add(best['order_id'])

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
                                    controller.buy(
                                        bot_id, ShopCosts.PAN, sx, sy)
                    else:
                        # No cooker available/needing pan? Assume setup is ok or wait
                        state.task_stage = 2
            else:
                state.task_stage = 2

        # State 2: Buy/place plate
        elif state.task_stage == 2:
            if not state.plate_counter:
                existing_plate = self.find_plate_on_counter(controller)
                if existing_plate:
                    state.plate_counter = existing_plate
                    state.task_stage = 10
                    return
            if holding:
                if holding.get('type') == 'Plate':
                    if not state.plate_counter:
                        anchor = self.cooker_priority[0] if self.cooker_priority else None
                        state.plate_counter = self.get_free_counter(
                            controller,
                            exclude=reserved_counters,
                            bot_pos=(bx, by),
                            blocked_tiles=blocked_tiles,
                            anchor_pos=anchor)

                    if state.plate_counter:
                        if my_dist and not self._has_access(my_dist, state.plate_counter):
                            if self._attempt_handoff(controller, bot_id, state, my_dist, other_dist_maps, blocked_tiles):
                                return
                            state.plate_counter = None
                            state.task_stage = 2
                            return
                        # Reserve it
                        reserved_counters.add(state.plate_counter)

                        px, py = state.plate_counter
                        adjacent = self.move_to(
                            controller, bot_id, px, py, blocked_tiles)
                        if adjacent:
                            if controller.place(bot_id, px, py):
                                state.task_stage = 10
                    else:
                        print(
                            f"Error: No plate counter available for Bot {bot_id}")
                else:
                    state.task_stage = 99
            else:
                adjacent = self.move_to(
                    controller, bot_id, sx, sy, blocked_tiles)
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
                loc, is_cooking = self.find_existing_ingredient(
                    controller, ing)

                if loc:
                    if my_dist and not self._has_access(my_dist, loc):
                        # Ignore unreachable items; try to get it ourselves or handoff.
                        loc = None
                        is_cooking = False
                    if loc:
                        if is_cooking:
                            # Cooking in execution
                            if self.needs_chopping(ing):
                                state.sub_state = 4
                            else:
                                state.sub_state = 1
                        else:
                            tile = controller.get_tile(
                                controller.get_team(), *loc)
                            if self.needs_cooking(ing) and my_dist and not self._can_access_any_cooker(my_dist):
                                # Can't cook on this side; ignore raw handoff items, but allow cooked pickups.
                                if tile and isinstance(tile.item, Food) and tile.item.cooked_stage == 1:
                                    if self.debug_board:
                                        print(
                                            f"[Bot {bot_id}] Accept cooked {tile.item.food_name} at {loc}")
                                else:
                                    if self.debug_board and tile and isinstance(tile.item, Food):
                                        print(
                                            f"[Bot {bot_id}] Ignore raw {tile.item.food_name} at {loc} (no cooker access)")
                                    loc = None
                                    is_cooking = False
                            if loc:
                                state.work_counter = loc
                                reserved_counters.add(loc)

                                # Check status
                                if tile and isinstance(tile.item, Food):
                                    is_chopped = tile.item.chopped

                                    if self.needs_chopping(ing):
                                        if is_chopped:
                                            state.sub_state = 2  # Pickup chopped
                                        else:
                                            state.sub_state = 1  # Chop
                                    else:
                                        state.sub_state = 0

                        if loc:
                            state.task_stage = 12
                            return

                # Buy only if not found
                if my_dist and not self._has_access(my_dist, (sx, sy)):
                    # Can't access shop; try to handoff if holding something.
                    if self._attempt_handoff(controller, bot_id, state, my_dist, other_dist_maps, blocked_tiles):
                        return
                cost = ft.buy_cost
                money = controller.get_team_money(controller.get_team())
                if money < cost:
                    # Only use boxes when we can't afford the shop item.
                    box_loc = self.find_box_with_ingredient(controller, ing)
                    if box_loc:
                        bx2, by2 = box_loc
                        if my_dist and not self._has_access(my_dist, (bx2, by2)):
                            if self._attempt_handoff(controller, bot_id, state, my_dist, other_dist_maps, blocked_tiles):
                                return
                        if self.move_to(controller, bot_id, bx2, by2, blocked_tiles):
                            if controller.pickup(bot_id, bx2, by2):
                                state.task_stage = 12
                        return

                if self.move_to(controller, bot_id, sx, sy, blocked_tiles):
                    if money >= cost:
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
            if self.debug_board:
                held_desc = None
                if holding:
                    if holding.get('type') == 'Food':
                        held_desc = f"{holding.get('food_name')}:{holding.get('cooked_stage', 0)}"
                    else:
                        held_desc = holding.get('type')
                can_cook = self._can_access_any_cooker(
                    my_dist) if my_dist else False
                can_shop = self._can_access_any_shop(
                    my_dist) if my_dist else False
                can_submit = self._has_access(
                    my_dist, (ux, uy)) if my_dist else False
                print(
                    f"[Bot {bot_id}] State12 ing={ing} sub={state.sub_state} holding={held_desc} can_cook={can_cook} can_shop={can_shop} can_submit={can_submit}")

            if chop and cook:
                self.process_chop_cook(
                    controller, bot_id, state, holding, reserved_counters, blocked_tiles, ing,
                    my_dist=my_dist, other_dist_maps=other_dist_maps, helper_mode=is_helper)
            elif cook:
                self.process_cook_only(
                    controller, bot_id, state, holding, reserved_counters, blocked_tiles, ing,
                    my_dist=my_dist, other_dist_maps=other_dist_maps, helper_mode=is_helper)
            elif chop:
                self.process_chop_only(
                    controller, bot_id, state, holding, reserved_counters, blocked_tiles,
                    my_dist=my_dist, other_dist_maps=other_dist_maps, helper_mode=is_helper)
            else:
                state.task_stage = 13

        # State 13: Add to plate
        elif state.task_stage == 13:
            if holding and holding.get('type') == 'Food':
                if holding.get('cooked_stage', 0) == 1:
                    if my_dist and not self._has_access(my_dist, (ux, uy)):
                        if self._attempt_handoff(controller, bot_id, state, my_dist, other_dist_maps, blocked_tiles):
                            return
            if not state.plate_counter:
                state.task_stage = 2
                return

            px, py = state.plate_counter
            if my_dist and not self._has_access(my_dist, (px, py)):
                if self._attempt_handoff(controller, bot_id, state, my_dist, other_dist_maps, blocked_tiles):
                    return
                state.plate_counter = None
                state.task_stage = 2
                return
            tile = controller.get_tile(controller.get_team(), px, py)
            if not tile or not isinstance(tile.item, Plate):
                # Plate was taken or moved; re-acquire a new plate.
                state.plate_counter = None
                state.task_stage = 2
                return

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
            else:
                if not state.plate_counter:
                    state.task_stage = 2
                    return

                px, py = state.plate_counter
                if my_dist and not self._has_access(my_dist, (px, py)):
                    state.plate_counter = None
                    state.task_stage = 0
                    return
                tile = controller.get_tile(controller.get_team(), px, py)
                if not tile or not isinstance(tile.item, Plate):
                    # Plate missing; go buy/place another.
                    state.plate_counter = None
                    state.task_stage = 2
                    return

                if self.move_to(controller, bot_id, px, py, blocked_tiles):
                    if controller.pickup(bot_id, px, py):
                        state.task_stage = 21

        # State 21: Submit
        elif state.task_stage == 21:
            if holding and holding.get('type') != 'Plate':
                state.task_stage = 99
                return
            if not holding:
                state.task_stage = 20
                return

            if my_dist and not self._has_access(my_dist, (ux, uy)):
                if self._attempt_handoff(controller, bot_id, state, my_dist, other_dist_maps, blocked_tiles):
                    return
            if self.move_to(controller, bot_id, ux, uy, blocked_tiles):
                if controller.can_submit(bot_id, ux, uy):
                    if controller.submit(bot_id, ux, uy):
                        print(f"[Bot {bot_id}] submit OK")
                        state.pending_order_id = None
                        state.pending_items = []
                        state.task_stage = 0
                    else:
                        state.task_stage = 20
                else:
                    print(
                        f"[Bot {bot_id}] submit blocked: can_submit=False holding={holding}")

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

    def run_normal_logic(self, controller, bot_id, state, reserved_counters, claimed_orders, blocked_tiles, order_heuristic):
        # Snapshot3 normal-map logic (no helper/handoff behaviors).
        info = controller.get_bot_state(bot_id)
        holding = info.get('holding')
        bx, by = info['x'], info['y']

        sx, sy = self.shop_loc
        ux, uy = self.submit_loc

        # Stuck detection
        if state.task_stage == state.last_state:
            state.stuck_counter += 1
            if state.stuck_counter > 10:
                print(
                    f"[Bot {bot_id}] STUCK in state {state.task_stage}, forcing random move")
                import random
                dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
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

                    score = self.calculate_order_heuristic(
                        controller, bot_id, state, o, turn, team_money, order_heuristic)

                    if score > best_score:
                        best_score = score
                        best = o

            if best:
                state.current_order = best
                # Claim it immediately for subsequent bots in this turn
                claimed_orders.add(best['order_id'])

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
                                    controller.buy(
                                        bot_id, ShopCosts.PAN, sx, sy)
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
                        anchor = self.cooker_priority[0] if self.cooker_priority else None
                        state.plate_counter = self.get_free_counter(
                            controller,
                            exclude=reserved_counters,
                            bot_pos=(bx, by),
                            blocked_tiles=blocked_tiles,
                            anchor_pos=anchor)

                    if state.plate_counter:
                        # Reserve it
                        reserved_counters.add(state.plate_counter)

                        px, py = state.plate_counter
                        adjacent = self.move_to(
                            controller, bot_id, px, py, blocked_tiles)
                        if adjacent:
                            if controller.place(bot_id, px, py):
                                state.task_stage = 10
                    else:
                        print(
                            f"Error: No plate counter available for Bot {bot_id}")
                else:
                    state.task_stage = 99
            else:
                adjacent = self.move_to(
                    controller, bot_id, sx, sy, blocked_tiles)
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
                loc, is_cooking = self.find_existing_ingredient(
                    controller, ing)

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
                                    state.sub_state = 2  # Pickup chopped
                                else:
                                    state.sub_state = 1  # Chop
                            else:
                                state.sub_state = 0

                    state.task_stage = 12
                    return

                # Buy only if not found
                cost = ft.buy_cost
                money = controller.get_team_money(controller.get_team())
                if money < cost:
                    # Only use boxes when we can't afford the shop item.
                    box_loc = self.find_box_with_ingredient(controller, ing)
                    if box_loc:
                        bx2, by2 = box_loc
                        if self.move_to(controller, bot_id, bx2, by2, blocked_tiles):
                            if controller.pickup(bot_id, bx2, by2):
                                state.task_stage = 12
                        return

                if self.move_to(controller, bot_id, sx, sy, blocked_tiles):
                    if money >= cost:
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
                self.process_chop_cook(
                    controller, bot_id, state, holding, reserved_counters, blocked_tiles, ing)
            elif cook:
                self.process_cook_only(
                    controller, bot_id, state, holding, reserved_counters, blocked_tiles, ing)
            elif chop:
                self.process_chop_only(
                    controller, bot_id, state, holding, reserved_counters, blocked_tiles)
            else:
                state.task_stage = 13

        # State 13: Add to plate
        elif state.task_stage == 13:
            if not state.plate_counter:
                state.task_stage = 2
                return

            px, py = state.plate_counter
            tile = controller.get_tile(controller.get_team(), px, py)
            if not tile or not isinstance(tile.item, Plate):
                # Plate was taken or moved; re-acquire a new plate.
                state.plate_counter = None
                state.task_stage = 2
                return

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
            else:
                if not state.plate_counter:
                    state.task_stage = 2
                    return

                px, py = state.plate_counter
                tile = controller.get_tile(controller.get_team(), px, py)
                if not tile or not isinstance(tile.item, Plate):
                    # Plate missing; go buy/place another.
                    state.plate_counter = None
                    state.task_stage = 2
                    return

                if self.move_to(controller, bot_id, px, py, blocked_tiles):
                    if controller.pickup(bot_id, px, py):
                        state.task_stage = 21

        # State 21: Submit
        elif state.task_stage == 21:
            if holding and holding.get('type') != 'Plate':
                state.task_stage = 99
                return
            if not holding:
                state.task_stage = 20
                return

            if self.move_to(controller, bot_id, ux, uy, blocked_tiles):
                if controller.can_submit(bot_id, ux, uy):
                    if controller.submit(bot_id, ux, uy):
                        state.task_stage = 0
                else:
                    state.task_stage = 20

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

    def process_chop_only(self, controller, bot_id, state, holding, reserved, blocked, my_dist=None, other_dist_maps=None, helper_mode=False):
        if not state.work_counter:
            # Exclude global reserved + own plate counter
            exclude = set(reserved)
            if state.plate_counter:
                exclude.add(state.plate_counter)

            bot_state = controller.get_bot_state(bot_id)
            anchor = self.cooker_priority[0] if self.cooker_priority else None
            state.work_counter = self.get_free_counter(
                controller,
                exclude,
                bot_pos=(bot_state['x'], bot_state['y']),
                blocked_tiles=blocked,
                anchor_pos=anchor)

        if not state.work_counter:
            state.task_stage = 99
            return

        if helper_mode and my_dist and not self._has_access(my_dist, state.work_counter):
            if self._attempt_handoff(controller, bot_id, state, my_dist, other_dist_maps or [], blocked):
                return
            state.task_stage = 99
            return

        reserved.add(state.work_counter)  # Lock it again
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

    def process_cook_only(self, controller, bot_id, state, holding, reserved, blocked, ing, my_dist=None, other_dist_maps=None, helper_mode=False):
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

                    if helper_mode and my_dist and not any(
                            self._has_access(my_dist, c) for c in self.cookers or []):
                        # No cooker access on this side; handoff raw cookable items.
                        if self._attempt_handoff(controller, bot_id, state, my_dist, other_dist_maps or [], blocked):
                            return
                        if self.primary_handoff_counter:
                            hx, hy = self.primary_handoff_counter
                            self.move_to(controller, bot_id, hx, hy, blocked)
                        return

                    bot_state = controller.get_bot_state(bot_id)
                    anchor = state.work_counter or state.plate_counter
                    cooker = None
                    if state.cooker_target:
                        tx, ty = state.cooker_target
                        t = controller.get_tile(controller.get_team(), tx, ty)
                        if t and isinstance(t.item, Pan) and t.item.food is None:
                            cooker = state.cooker_target
                        else:
                            state.cooker_target = None

                    if not cooker:
                        cooker = self.choose_cooker(
                            controller,
                            anchor_pos=anchor,
                            bot_pos=(bot_state['x'], bot_state['y']),
                            blocked_tiles=blocked,
                            want_food=False)
                        state.cooker_target = cooker

                    if cooker:
                        kx, ky = cooker
                        if helper_mode:
                            step = self.get_next_step_astar(
                                controller, (bot_state['x'], bot_state['y']), kx, ky, blocked=blocked)
                            if step is None:
                                if self._attempt_handoff(controller, bot_id, state, my_dist, other_dist_maps or [], blocked):
                                    return
                        if self.move_to(controller, bot_id, kx, ky, blocked):
                            if controller.place(bot_id, kx, ky):
                                state.sub_state = 1
                    elif self.cookers:
                        self.move_to(controller, bot_id, *
                                     self.cookers[0], blocked_tiles=blocked)
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
            # Prefer the assigned cooker to minimize travel.
            target = state.cooker_target
            candidates = []
            if target:
                candidates.append(target)
            else:
                bot_state = controller.get_bot_state(bot_id)
                anchor = state.work_counter or state.plate_counter
                picked = self.choose_cooker(
                    controller,
                    anchor_pos=anchor,
                    bot_pos=(bot_state['x'], bot_state['y']),
                    blocked_tiles=blocked,
                    want_food=True)
                if picked:
                    state.cooker_target = picked
                    candidates.append(picked)

            for kx, ky in candidates:
                tile = controller.get_tile(controller.get_team(), kx, ky)
                if not tile or not isinstance(tile.item, Pan) or not tile.item.food:
                    continue
                stage = tile.item.food.cooked_stage
                if stage == 1:
                    if self.move_to(controller, bot_id, kx, ky, blocked):
                        if controller.take_from_pan(bot_id, kx, ky):
                            state.task_stage = 13
                            state.sub_state = 0
                            state.cooker_target = None
                    return
                elif stage == 2:
                    if self.move_to(controller, bot_id, kx, ky, blocked):
                        controller.take_from_pan(bot_id, kx, ky)
                    state.task_stage = 99
                    state.sub_state = 0
                    state.cooker_target = None
                    return
                else:
                    if holding is None and len(state.ingredients_needed) > 1 and state.ingredients_needed[0] == ing:
                        # Defer cooking: work on other ingredients while it finishes.
                        state.ingredients_needed.pop(0)
                        state.ingredients_needed.append(ing)
                        state.task_stage = 10
                        state.sub_state = 0
                        return
                    self.move_to(controller, bot_id, kx, ky, blocked)
                    return
            state.task_stage = 11
            state.sub_state = 0

    def process_chop_cook(self, controller, bot_id, state, holding, reserved, blocked, ing, my_dist=None, other_dist_maps=None, helper_mode=False):
        if not state.work_counter:
            exclude = set(reserved)
            if state.plate_counter:
                exclude.add(state.plate_counter)
            bot_state = controller.get_bot_state(bot_id)
            anchor = self.cooker_priority[0] if self.cooker_priority else None
            state.work_counter = self.get_free_counter(
                controller,
                exclude,
                bot_pos=(bot_state['x'], bot_state['y']),
                blocked_tiles=blocked,
                anchor_pos=anchor)

        if not state.work_counter:
            state.task_stage = 99
            return

        if helper_mode and my_dist and not self._has_access(my_dist, state.work_counter):
            if self._attempt_handoff(controller, bot_id, state, my_dist, other_dist_maps or [], blocked):
                return
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

                    if helper_mode and my_dist and not any(
                            self._has_access(my_dist, c) for c in self.cookers or []):
                        if self._attempt_handoff(controller, bot_id, state, my_dist, other_dist_maps or [], blocked):
                            return
                        if self.primary_handoff_counter:
                            hx, hy = self.primary_handoff_counter
                            self.move_to(controller, bot_id, hx, hy, blocked)
                        return

                    bot_state = controller.get_bot_state(bot_id)
                    anchor = state.work_counter or state.plate_counter
                    cooker = None
                    if state.cooker_target:
                        tx, ty = state.cooker_target
                        t = controller.get_tile(controller.get_team(), tx, ty)
                        if t and isinstance(t.item, Pan) and t.item.food is None:
                            cooker = state.cooker_target
                        else:
                            state.cooker_target = None

                    if not cooker:
                        cooker = self.choose_cooker(
                            controller,
                            anchor_pos=anchor,
                            bot_pos=(bot_state['x'], bot_state['y']),
                            blocked_tiles=blocked,
                            want_food=False)
                        state.cooker_target = cooker

                    if cooker:
                        kx, ky = cooker
                        if helper_mode:
                            step = self.get_next_step_astar(
                                controller, (bot_state['x'], bot_state['y']), kx, ky, blocked=blocked)
                            if step is None:
                                if self._attempt_handoff(controller, bot_id, state, my_dist, other_dist_maps or [], blocked):
                                    return
                        if self.move_to(controller, bot_id, kx, ky, blocked):
                            if controller.place(bot_id, kx, ky):
                                state.sub_state = 4
                    elif self.cookers:
                        self.move_to(controller, bot_id, *
                                     self.cookers[0], blocked_tiles=blocked)
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
            # Prefer the assigned cooker to minimize travel.
            target = state.cooker_target
            candidates = []
            if target:
                candidates.append(target)
            else:
                bot_state = controller.get_bot_state(bot_id)
                anchor = state.work_counter or state.plate_counter
                picked = self.choose_cooker(
                    controller,
                    anchor_pos=anchor,
                    bot_pos=(bot_state['x'], bot_state['y']),
                    blocked_tiles=blocked,
                    want_food=True)
                if picked:
                    state.cooker_target = picked
                    candidates.append(picked)

            for kx, ky in candidates:
                tile = controller.get_tile(controller.get_team(), kx, ky)
                if not tile or not isinstance(tile.item, Pan) or not tile.item.food:
                    continue
                stage = tile.item.food.cooked_stage
                if stage == 1:
                    if self.move_to(controller, bot_id, kx, ky, blocked):
                        if controller.take_from_pan(bot_id, kx, ky):
                            state.task_stage = 13
                            state.sub_state = 0
                            state.cooker_target = None
                    return
                elif stage == 2:
                    if self.move_to(controller, bot_id, kx, ky, blocked):
                        controller.take_from_pan(bot_id, kx, ky)
                    state.task_stage = 99
                    state.sub_state = 0
                    state.cooker_target = None
                    return
                else:
                    if holding is None and len(state.ingredients_needed) > 1 and state.ingredients_needed[0] == ing:
                        # Defer cooking: work on other ingredients while it finishes.
                        state.ingredients_needed.pop(0)
                        state.ingredients_needed.append(ing)
                        state.task_stage = 10
                        state.sub_state = 0
                        return
                    self.move_to(controller, bot_id, kx, ky, blocked)
                    return
            state.task_stage = 11
            state.sub_state = 0
