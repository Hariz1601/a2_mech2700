import numpy as np
import matplotlib.pyplot as plt

def calculate_stable_timestep(CFL, D, dx, dy, vx):
    dt_x = CFL * (dx ** 2) / D
    dt_y = CFL * (dy ** 2) / D
    dt_vx = CFL * dx / abs(vx)
    return min(dt_x, dt_y, dt_vx)

def update_solution(u, dt, dx, vx, D):
    nx, ny = u.shape
    new_u = np.copy(u)

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            du_dx = (u[i+1, j] - 2 * u[i, j] + u[i-1, j]) / (dx**2)
            du_dy = (u[i, j+1] - 2 * u[i, j] + u[i, j-1]) / (dx**2)
            new_u[i,j] = u[i,j] + dt * (( D * (du_dx + du_dy)) - vx * ((u[i+1,j] - u[i,j])/ dx))

    new_u[0, :] = new_u[1, :]  # du/dx = 0 at left boundary
    new_u[-1, :] = new_u[-2, :]  # du/dx = 0 at right boundary
    new_u[:, 0] = new_u[:, 1]  # du/dy = 0 at bottom boundary
    new_u[:, -1] = new_u[:, -2]  # du/dy = 0 at top boundary

    return new_u

def main():
    tmax = 20.0
    dx = 0.2
    nx, ny = int(10 / dx), int(4 / dx)
    vx = -0.25
    D = 0.01
    CFL = 0.45
    dt = calculate_stable_timestep(CFL, D, dx, dx, vx)  # Calculate the stable time step

    u = np.zeros((nx, ny))
    u[int(nx/2), int(ny/2)] = 10**4 / (dx**2)

    t = 0.0
    particle_density_at_B = []

    while t <= tmax:
        if abs(t % 5) < dt:
            # Record the full solution array every ≈5 seconds
            np.save('save_data.npy', u)

        # Calculate particle number density at location B (3, 2) and record it
        particle_density_at_B.append(u[3, 2])

        u = update_solution(u, dt, dx, vx, D)
        t += dt

    threshold = 10
    exceeded_threshold = False
    first_exceeds_time = None

    for i, density in enumerate(particle_density_at_B):
        if density > threshold:
            exceeded_threshold = True
            first_exceeds_time = i * 5  # Convert the index to time in seconds
        break
    if exceeded_threshold:
        print(f"The particle count at location B exceeds the threshold of 10 at {first_exceeds_time} seconds.")
    else:
        print("The particle count at location B does not exceed the threshold of 10 during the simulation.")


    # Generate contour plots at 5s, 10s, 15s, and 20s
    for t in [5, 10, 15, 20]:
        plt.figure()
        u = np.load('save_data.npy')
        plt.contourf(u.T, cmap='viridis')
        plt.colorbar()
        plt.title(f"Particle Density at t = {t} seconds")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    # Create a graph of particle number density at location B
    plt.figure()
    time_points = np.arange(0, tmax + dt, dt)
    plt.plot(time_points, particle_density_at_B)
    plt.title("Particle Density at Location B Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Particle Density at B")
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()
