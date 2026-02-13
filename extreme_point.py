import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ============================================================
# Data Classes
# ============================================================

class Bin:
    def __init__(self, ID, length, height, cost, a, b):
        self.ID = ID
        self.length = length
        self.height = height
        self.cost = cost
        self.a = a  # Cut parameter (0 if no cut)
        self.b = b  # Cut parameter (0 if no cut)
    
    def has_cut(self):
        return self.a > 0 and self.b > 0
    
    def get_cut_y(self, x):
        """Get y-coordinate of cut line at position x."""
        if not self.has_cut() or x >= self.a:
            return 0
        return self.b * (1 - x / self.a)


class Item:
    def __init__(self, ID, length, height, rotate, fragile, perishable, radioactive):
        self.ID = ID
        self.length = length
        self.height = height
        self.rotate = rotate
        self.fragile = fragile
        self.perishable = perishable
        self.radioactive = radioactive


class PlacedItem:
    def __init__(self, item, x, y, rotated=False):
        self.item = item
        self.x = x
        self.y = y
        self.rotated = rotated
        self.width = item.height if rotated else item.length
        self.height = item.length if rotated else item.height
        self.x2 = x + self.width
        self.y2 = y + self.height
    
    def contains_point(self, px, py):
        """Check if point (px, py) is strictly inside this item."""
        return self.x < px < self.x2 and self.y < py < self.y2
    
    def overlaps(self, x, y, w, h):
        """Check if rectangle (x, y, w, h) overlaps with this item."""
        return not (x >= self.x2 or x + w <= self.x or y >= self.y2 or y + h <= self.y)


# ============================================================
# Bin Packing Logic
# ============================================================

class BinPacker:
    def __init__(self, bin_obj, pack_right_to_left=False):
        self.bin = bin_obj
        self.placed_items = []
        self.extreme_points = set()  # Use set for automatic deduplication
        self.pack_right_to_left = pack_right_to_left
        self._init_extreme_points()
    
    def _init_extreme_points(self):
        """Initialize starting extreme points."""
        if self.pack_right_to_left:
            # Start at right corner for right-to-left packing
            self.extreme_points.add((self.bin.length, 0))
        elif self.bin.has_cut():
            # Start at (a, 0) - first valid point after cut
            self.extreme_points.add((self.bin.a, 0))
        else:
            self.extreme_points.add((0, 0))
    
    def _is_point_valid(self, x, y):
        """Check if an extreme point is valid (not inside items, above cut)."""
        # Check cut constraint
        if self.bin.has_cut() and x < self.bin.a:
            if y < self.bin.get_cut_y(x) - 0.001:
                return False
        
        # Check not inside any placed item
        for pi in self.placed_items:
            if pi.contains_point(x, y):
                return False
        
        return True
    
    def _update_extreme_points(self, placed_item, used_ep):
        """Update extreme points after placing an item."""
        # Remove the EP we just used
        self.extreme_points.discard(used_ep)
        
        if self.pack_right_to_left:
            # For right-to-left packing: new EPs at bottom-left and top-right
            new_eps = [
                (placed_item.x, placed_item.y),   # Bottom-left (pack more to the left)
                (placed_item.x2, placed_item.y2), # Top-right (stack on top)
            ]
        else:
            # Standard left-to-right packing
            new_eps = [
                (placed_item.x2, placed_item.y),  # Bottom-right
                (placed_item.x, placed_item.y2),  # Top-left
            ]
        
        for ep in new_eps:
            x, y = ep
            # Check bounds
            if 0 <= x <= self.bin.length and 0 <= y < self.bin.height:
                # Check validity (not inside items, above cut)
                if self._is_point_valid(x, y):
                    self.extreme_points.add(ep)
    
    def _check_placement(self, item, x, y, rotated):
        """Check if item can be placed at (x, y) with given rotation."""
        if rotated and item.rotate == 0:
            return False
        
        w = item.height if rotated else item.length
        h = item.length if rotated else item.height
        
        # For right-to-left packing, x is the RIGHT edge, so item goes from x-w to x
        if self.pack_right_to_left:
            actual_x = x - w
            if actual_x < 0 or y < 0 or x > self.bin.length or y + h > self.bin.height:
                return False
            # Use actual_x for remaining checks
            x = actual_x
        else:
            # Bounds check
            if x < 0 or y < 0 or x + w > self.bin.length or y + h > self.bin.height:
                return False
        
        # Cut check - all corners must be above cut line
        if self.bin.has_cut():
            corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
            for cx, cy in corners:
                if cx < self.bin.a and cy < self.bin.b:
                    if cx / self.bin.a + cy / self.bin.b < 1 - 0.001:
                        return False
        
        # Overlap check
        for pi in self.placed_items:
            if pi.overlaps(x, y, w, h):
                return False
        
        # Support check
        if not self._check_support(x, y, w):
            return False
        
        # Fragility check
        if not self._check_fragility(x, y, w):
            return False
        
        return True
    
    def _check_support(self, x, y, width):
        """Check if both bottom corners are supported."""
        def is_supported(cx, cy):
            # Ground support
            if abs(cy) < 1.0:
                if self.bin.has_cut() and cx < self.bin.a - 0.001:
                    return False
                return True
            # Cut line support
            if self.bin.has_cut() and 0 <= cx <= self.bin.a:
                if abs(cy - self.bin.get_cut_y(cx)) < 1.0:
                    return True
            # Item support
            for pi in self.placed_items:
                if abs(cy - pi.y2) < 1.0 and pi.x - 0.001 <= cx <= pi.x2 + 0.001:
                    return True
            return False
        
        return is_supported(x, y) and is_supported(x + width, y)
    
    def _check_fragility(self, x, y, width):
        """Check we're not placing on top of fragile items."""
        for pi in self.placed_items:
            if pi.item.fragile == 1 and abs(y - pi.y2) < 0.001:
                if not (x >= pi.x2 or x + width <= pi.x):
                    return False
        return True
    
    def _calculate_residual_space(self, x, y):
        """Calculate available space at point (x, y)."""
        RS_x = self.bin.length - x
        RS_y = self.bin.height - y
        
        for pi in self.placed_items:
            if pi.x >= x and pi.y < y + RS_y and pi.y2 > y:
                RS_x = min(RS_x, pi.x - x)
            if pi.y >= y and pi.x < x + RS_x and pi.x2 > x:
                RS_y = min(RS_y, pi.y - y)
        
        return max(0, RS_x), max(0, RS_y)
    
    def find_best_ep(self, item):
        """Find best EP for item. Returns (x, y, rotated, merit) or None."""
        best = None
        best_merit = float('inf')
        
        for (x, y) in self.extreme_points:
            RS_x, RS_y = self._calculate_residual_space(x, y)
            
            # Try normal orientation
            if self._check_placement(item, x, y, False):
                merit = (RS_x - item.length) + (RS_y - item.height)
                if merit < best_merit:
                    best_merit = merit
                    best = (x, y, False)
            
            # Try rotated orientation
            if item.rotate == 1 and self._check_placement(item, x, y, True):
                merit = (RS_x - item.height) + (RS_y - item.length)
                if merit < best_merit:
                    best_merit = merit
                    best = (x, y, True)
        
        return best
    
    def place_item(self, item, x, y, rotated):
        """Place item and update extreme points."""
        used_ep = (x, y)
        
        # For right-to-left, x is the right edge, convert to left edge for PlacedItem
        if self.pack_right_to_left:
            w = item.height if rotated else item.length
            actual_x = x - w
            placed = PlacedItem(item, actual_x, y, rotated)
        else:
            placed = PlacedItem(item, x, y, rotated)
        
        self.placed_items.append(placed)
        self._update_extreme_points(placed, used_ep)
        return placed
    
    def check_compatibility(self, item):
        """Check perishable/radioactive compatibility."""
        has_perishable = any(pi.item.perishable == 1 for pi in self.placed_items)
        has_radioactive = any(pi.item.radioactive == 1 for pi in self.placed_items)
        
        if item.perishable == 1 and has_radioactive:
            return False
        if item.radioactive == 1 and has_perishable:
            return False
        return True


# ============================================================
# Main Algorithm
# ============================================================

def run_packing(items, bins):
    """Run EP-BFD bin packing algorithm with two-phase approach:
       Phase 1: Pack into bins WITH cuts (right-to-left for efficient space use)
       Phase 2: Pack remaining items into bins WITHOUT cuts (pallets)
    """
    sorted_items = sorted(items, key=lambda x: x.length * x.height, reverse=True)
    
    # Separate bins by cut status
    bins_no_cut = [b for b in bins if not b.has_cut()]
    bins_with_cut = [b for b in bins if b.has_cut()]
    
    open_packers = {}  # bin_id -> BinPacker
    bin_counter = 0
    
    # ============ PHASE 1: Pack into bins WITH cuts (right-to-left) ============
    print("=== Phase 1: Packing into bins WITH cuts (right-to-left) ===")
    available_with_cut = list(bins_with_cut)
    remaining_items = []
    
    for item in sorted_items:
        placed = False
        
        # Try existing bins with cuts
        best_packer_id = None
        best_placement = None
        best_merit = float('inf')
        
        for pid, packer in open_packers.items():
            if not packer.check_compatibility(item):
                continue
            result = packer.find_best_ep(item)
            if result:
                x, y, rotated = result
                RS_x, RS_y = packer._calculate_residual_space(x, y)
                w = item.height if rotated else item.length
                h = item.length if rotated else item.height
                merit = (RS_x - w) + (RS_y - h)
                if merit < best_merit:
                    best_merit = merit
                    best_packer_id = pid
                    best_placement = result
        
        if best_packer_id is not None:
            x, y, rotated = best_placement
            open_packers[best_packer_id].place_item(item, x, y, rotated)
            placed = True
        else:
            # Open new bin with cut (right-to-left packing)
            for bin_template in sorted(available_with_cut, key=lambda b: (b.cost, -(b.length * b.height))):
                packer = BinPacker(bin_template, pack_right_to_left=True)
                result = packer.find_best_ep(item)
                if result:
                    x, y, rotated = result
                    packer.place_item(item, x, y, rotated)
                    open_packers[bin_counter] = packer
                    available_with_cut.remove(bin_template)
                    bin_counter += 1
                    placed = True
                    break
        
        if not placed:
            remaining_items.append(item)
    
    print(f"Phase 1 complete: {len(sorted_items) - len(remaining_items)} items placed in {len(open_packers)} bins")
    print(f"Items remaining for Phase 2: {len(remaining_items)}")
    
    # ============ PHASE 2: Pack remaining into bins WITHOUT cuts (pallets) ============
    if remaining_items:
        print("\n=== Phase 2: Packing remaining items into pallets (no cuts) ===")
        available_no_cut = list(bins_no_cut)
        unplaced = []
        
        # Re-sort remaining items by area
        remaining_items = sorted(remaining_items, key=lambda x: x.length * x.height, reverse=True)
        
        for item in remaining_items:
            placed = False
            
            # Try existing bins without cuts
            best_packer_id = None
            best_placement = None
            best_merit = float('inf')
            
            for pid, packer in open_packers.items():
                if packer.bin.has_cut():
                    continue  # Skip cut bins in phase 2
                if not packer.check_compatibility(item):
                    continue
                result = packer.find_best_ep(item)
                if result:
                    x, y, rotated = result
                    RS_x, RS_y = packer._calculate_residual_space(x, y)
                    w = item.height if rotated else item.length
                    h = item.length if rotated else item.height
                    merit = (RS_x - w) + (RS_y - h)
                    if merit < best_merit:
                        best_merit = merit
                        best_packer_id = pid
                        best_placement = result
            
            if best_packer_id is not None:
                x, y, rotated = best_placement
                open_packers[best_packer_id].place_item(item, x, y, rotated)
                placed = True
            else:
                # Open new bin without cut (standard left-to-right)
                for bin_template in sorted(available_no_cut, key=lambda b: (b.cost, -(b.length * b.height))):
                    packer = BinPacker(bin_template, pack_right_to_left=False)
                    result = packer.find_best_ep(item)
                    if result:
                        x, y, rotated = result
                        packer.place_item(item, x, y, rotated)
                        open_packers[bin_counter] = packer
                        available_no_cut.remove(bin_template)
                        bin_counter += 1
                        placed = True
                        break
            
            if not placed:
                unplaced.append(item)
        
        print(f"Phase 2 complete: {len(remaining_items) - len(unplaced)} additional items placed")
    else:
        unplaced = []
    
    return open_packers, unplaced


# ============================================================
# Visualization
# ============================================================

def plot_bins(packers, title="Bin Packing Results"):
    num_bins = len(packers)
    if num_bins == 0:
        print("No bins to plot.")
        return
    
    fig, axes = plt.subplots(1, num_bins, figsize=(6 * num_bins, 5))
    if num_bins == 1:
        axes = [axes]
    
    colors = ['#FF00FF', '#00FF00', '#FF0000', '#0000FF', '#FFA500', '#00FFFF', 
              '#FFFF00', '#800080', '#006400', '#FFC0CB', '#A52A2A', '#808000']
    
    for idx, (bin_id, packer) in enumerate(packers.items()):
        ax = axes[idx]
        bin_obj = packer.bin
        
        # Draw bin boundary
        ax.add_patch(patches.Rectangle((0, 0), bin_obj.length, bin_obj.height, 
                                        fill=False, edgecolor='black', linewidth=2))
        
        # Draw cut region with hatching
        if bin_obj.has_cut():
            ax.plot([bin_obj.a, 0], [0, bin_obj.b], 'k-', linewidth=2)
            triangle = plt.Polygon([(0, 0), (bin_obj.a, 0), (0, bin_obj.b)], 
                                   color='lightgray', alpha=0.5, hatch='///')
            ax.add_patch(triangle)
        
        # Draw placed items
        for pi in packer.placed_items:
            color = '#808080' if pi.item.fragile == 1 else colors[pi.item.ID % len(colors)]
            rect = patches.Rectangle((pi.x, pi.y), pi.width, pi.height,
                                      linewidth=1.5, edgecolor='black', 
                                      facecolor=color, alpha=0.8)
            ax.add_patch(rect)
            # Label: ID,(fragile,perishable,radioactive)
            label = f"{pi.item.ID},({pi.item.fragile},{pi.item.perishable},{pi.item.radioactive})"
            ax.text(pi.x + pi.width/2, pi.y + pi.height/2, label,
                   ha='center', va='center', fontsize=7, fontweight='bold')
        
        # Axis settings
        ax.set_xlim(0, bin_obj.length)
        ax.set_ylim(0, bin_obj.height)
        ax.set_xlabel('Length')
        ax.set_ylabel('Height')
        ax.set_title(f'ULD {bin_id}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('bin_packing_result.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# Load Data and Run
# ============================================================

with open('B.pickle', 'rb') as handle:
    B = pickle.load(handle)
with open('I.pickle', 'rb') as handle:
    I = pickle.load(handle)

Bins = [Bin(b, v[1][0], v[1][1], v[1][3], v[1][4], v[1][5]) for b, v in enumerate(B.values())]
Items = [Item(i, v[0], v[1], v[2], v[3], v[4], v[5]) for i, v in enumerate(I.values())]

# Run algorithm
packers, unplaced = run_packing(Items, Bins)

# Output results
print(f"Bins used: {len(packers)}")
print(f"Items placed: {len(Items) - len(unplaced)}/{len(Items)}")
if unplaced:
    print(f"Unplaced: {[i.ID for i in unplaced]}")

# Save outputs
bins_used = [p.bin.ID for p in packers.values()]
Items_in_Bin = {k: [pi.item.ID for pi in p.placed_items] for k, p in packers.items()}
I_info_solution = {}
for p in packers.values():
    for pi in p.placed_items:
        I_info_solution[pi.item.ID] = [float(pi.x), float(pi.y), float(pi.width), float(pi.height)]

# Sort by item ID (0-24)
I_info_solution = dict(sorted(I_info_solution.items()))

with open('bins_used.pickle', 'wb') as f:
    pickle.dump(bins_used, f)
with open('Items_in_Bin.pickle', 'wb') as f:
    pickle.dump(Items_in_Bin, f)
with open('I_info_solution.pickle', 'wb') as f:
    pickle.dump(I_info_solution, f)

plot_bins(packers)

