"""
Compute key performance metrics
"""
import numpy as np

def analyze_training(attendance_history, optimal=60, name="Model"):
    """Compute metrics for a training run"""
    
    # Last 100 episodes (converged behavior)
    final_attendance = attendance_history[-100:]
    
    # Metrics
    mean_attendance = np.mean(final_attendance)
    std_attendance = np.std(final_attendance)
    mae = np.mean(np.abs(np.array(final_attendance) - optimal))
    
    # Time at optimal (within ±5)
    near_optimal = np.sum(np.abs(np.array(final_attendance) - optimal) <= 5)
    pct_near_optimal = 100 * near_optimal / len(final_attendance)
    
    print(f"\n{'='*50}")
    print(f"{name} - Performance Metrics (Last 100 episodes)")
    print(f"{'='*50}")
    print(f"Mean Attendance:     {mean_attendance:.2f} (optimal: {optimal})")
    print(f"Std Deviation:       {std_attendance:.2f}")
    print(f"MAE from optimal:    {mae:.2f}")
    print(f"% Near Optimal (±5): {pct_near_optimal:.1f}%")
    print(f"{'='*50}\n")
    
    return {
        'mean': mean_attendance,
        'std': std_attendance,
        'mae': mae,
        'pct_optimal': pct_near_optimal
    }

# Example usage
print("Baseline Results:")
print("From your training log:")
print("Episodes 900-1000: oscillating between 0-100")
print("Mean ≈ 50, but high variance")

print("\nImproved Results:")
print("Should show:")
print("- Lower variance")
print("- Mean closer to 60")
print("- Higher % time near optimal")

