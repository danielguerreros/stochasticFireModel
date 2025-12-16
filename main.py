import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import animation

def make_fire_gif(history, gif_path="fire_spread.gif", interval=100):
    """
    Create a GIF animation from fire simulation history.

    history  : list of (time, grid) tuples
    gif_path : output gif filename
    interval : milliseconds between frames
    """

    cmap = ListedColormap(["green", "red", "black"])

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(history[0][1], cmap=cmap, origin="lower")
    time_text = ax.text(
        0.02, 0.95, "",
        transform=ax.transAxes,
        color="white",
        fontsize=12
    )
    ax.axis("off")

    def update(frame):
        t, grid = history[frame]
        im.set_data(grid)
        time_text.set_text(f"t = {t:.2f}")
        return im, time_text

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(history),
        interval=interval,
        blit=False
    )

    ani.save(gif_path, writer="pillow")
    plt.close(fig)

def plot_final_fire(grid):
    """
    Visualize final fire state.
    FUEL = 0 (green)
    BURNING = 1 (red)
    OUT = 2 (black)
    """
    cmap = ListedColormap(["green", "red", "black"])

    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap=cmap, origin="lower")
    plt.colorbar(
        ticks=[0, 1, 2],
        label="State"
    )
    plt.clim(-0.5, 2.5)
    plt.title("Final Fire State")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()

# States
FUEL = 0     # F
BURNING = 1  # B
OUT = 2      # O


class FireSimulator:
    def __init__(self, n, m, lam, lam_o, ignition_site, neighbourhood):
        """
        n, m         : grid size
        lam          : fire spread rate (λ)
        lam_o        : burnout rate (λ_o)
        ignition_site: tuple (i, j)
        neighbourhood: type or neighbourhood, von neumann (4 neighbors) or moore (8 neighbors)
        """
        self.n = n
        self.m = m
        self.lam = lam
        self.lam_o = lam_o

        self.grid = np.full((n, m), FUEL, dtype=int)
        self.grid[ignition_site] = BURNING

        self.time = 0.0
        self.history = [(self.time, self.grid.copy())]
        self.neighbourhood = neighbourhood

    def neighbors(self, i, j):
        if self.neighbourhood == "von_neumann":
            directions = [
                (-1, 0, self.lam),
                (1, 0, self.lam),
                (0, -1, self.lam),
                (0, 1, self.lam),
            ]

        elif self.neighbourhood == "moore":
            lam_diag = self.lam / np.sqrt(2)
            directions = [
                (-1, 0, self.lam),
                (1, 0, self.lam),
                (0, -1, self.lam),
                (0, 1, self.lam),
                (-1, -1, lam_diag),
                (-1,  1, lam_diag),
                ( 1, -1, lam_diag),
                ( 1,  1, lam_diag),
            ]

        for di, dj, rate in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.n and 0 <= nj < self.m:
                yield ni, nj, rate

    def step(self):
        """Perform one discrete event"""
        burning_sites = list(zip(*np.where(self.grid == BURNING)))
        N = len(burning_sites)

        if N == 0:
            return False  # simulation ends

        events = []
        rates = []

        for i, j in burning_sites:
            # Burnout event
            rate_o = self.lam_o
            events.append(("burnout", i, j))
            rates.append(rate_o)

            # Spread events
            for ni, nj, rate in self.neighbors(i, j):
                if self.grid[ni, nj] == FUEL:
                    events.append(("spread", i, j, ni, nj))
                    rates.append(rate)

        total_rate = sum(rates)
        if total_rate == 0:
            return False
    
        #total_rate = N * (4 * self.lam + self.lam_o)

        # Time to next event
        dt = np.random.exponential(1 / total_rate)
        self.time += dt

        # Choose event
        event = random.choices(events, weights=rates, k=1)[0]
        
        if event[0] == "burnout":
            _, i, j = event
            self.grid[i, j] = OUT

        elif event[0] == "spread":
            _, i, j, ni, nj = event
            self.grid[ni, nj] = BURNING

        self.history.append((self.time, self.grid.copy()))
        return True

    def run(self, max_steps=1000):
        """Run simulation until fire dies out or step limit reached"""
        steps = 0
        while steps < max_steps and self.step():
            steps += 1
        return self.time, steps

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    sim = FireSimulator(
        n=50,
        m=50,
        lam=5/100,# 5 m/min / 100m,
        lam_o= 1/240, # 1/120 min,
        ignition_site=(25, 25),
        neighbourhood='von_neumann'
    )

    final_time, steps = sim.run(max_steps=100)

    print("Simulation finished")
    print("Final time:", final_time)
    print("Total events:", steps)

    # Count final states
    unique, counts = np.unique(sim.grid, return_counts=True)
    state_counts = dict(zip(unique, counts))
    print("Fuel:", state_counts.get(FUEL, 0))
    print("Burning:", state_counts.get(BURNING, 0))
    print("Burnt out:", state_counts.get(OUT, 0))
    # after running your simulation
    make_fire_gif(sim.history, gif_path="fire_spread.gif")
    # plot_final_fire(sim.grid)
    