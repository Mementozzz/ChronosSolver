import argparse
from solver.solver import ClockSolver

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chronos Clock Solver')
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='Enable verbose output with debug information')
    parser.add_argument('--no-window', action='store_true',
                        help='Run without GUI window (headless mode)')
    
    args = parser.parse_args()
    
    solver = ClockSolver(verbose=args.verbose, show_window=not args.no_window)
    solver.run()