import argparse
from solver.solver import ClockSolver

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chronos Clock Solver')
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='Enable verbose output with debug information')
    parser.add_argument('--no-window', action='store_true',
                        help='Run without GUI window (headless mode)')
    parser.add_argument('--no-notifications', action='store_true',
                        help='Disable desktop notifications')
    
    args = parser.parse_args()
    
    solver = ClockSolver(
        verbose=args.verbose, 
        show_window=not args.no_window,
        enable_notifications=not args.no_notifications
    )
    solver.run()