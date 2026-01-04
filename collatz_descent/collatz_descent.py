#!/usr/bin/env python3
"""
Collatz Direct Descent Simulator
Simulates descent from N = ak + b to row m of the Collatz tree
where k is a symbolic variable representing any non-negative integer
"""

def simulate_descent(initial_a, initial_b):
    """
    Simulate the descent from N = ak + b to row m.
    
    Args:
        initial_a: Initial coefficient of k (should be 2^m)
        initial_b: Initial constant term (0 to 2^m - 1)
    
    Returns:
        dict: Results including final values, path, n, m, f
    """
    a = initial_a
    b = initial_b
    steps = 0
    odd_steps = 0
    f = 0
    path = []
    trace = [{'a': a, 'b': b, 'step': 0, 'type': 'start'}]
    
    # Continue while a is even
    while a % 2 == 0:
        steps += 1
        
        if b % 2 == 0:
            # Both even: divide both by 2
            path.append('E')
            a = a // 2
            b = b // 2
            trace.append({
                'a': a,
                'b': b,
                'step': steps,
                'type': 'even',
                'odd_steps': odd_steps,
                'f': f
            })
        else:
            # a even, b odd: apply odd rule
            path.append('O')
            odd_steps += 1
            
            # Calculate f contribution
            current_m = steps - 1
            f = 3 * f + (2 ** current_m)
            
            a = (3 * a) // 2
            b = (3 * b + 1) // 2
            
            trace.append({
                'a': a,
                'b': b,
                'step': steps,
                'type': 'odd',
                'odd_steps': odd_steps,
                'f': f
            })
    
    return {
        'initial_a': initial_a,
        'initial_b': initial_b,
        'final_a': a,
        'final_b': b,
        'n': odd_steps,
        'm': steps,
        'f': f,
        'path': ''.join(path),
        'trace': trace
    }


def format_expression(a, b):
    """Format ak + b expression nicely."""
    if a == 1:
        coef = "k"
    else:
        coef = f"{a}k"
    
    if b == 0:
        return coef
    else:
        return f"{coef} + {b}"


def print_result(result, show_trace=False):
    """Print a single descent result."""
    print(f"\n{'='*70}")
    print(f"Starting: N = {format_expression(result['initial_a'], result['initial_b'])}")
    print(f"Final:    M = {format_expression(result['final_a'], result['final_b'])}")
    print(f"-"*70)
    print(f"Path:       {result['path']}")
    print(f"Odd steps:  n = {result['n']}")
    print(f"Total steps: m = {result['m']}")
    print(f"f value:    f = {result['f']}")
    
    if show_trace:
        print(f"\nStep-by-step trace:")
        for step in result['trace']:
            step_type = step['type'].upper()
            expr = format_expression(step['a'], step['b'])
            if step['type'] == 'start':
                print(f"  Step {step['step']}: [START] N = {expr}")
            else:
                print(f"  Step {step['step']}: [{step_type}] N = {expr} (n={step['odd_steps']}, f={step['f']})")


def analyze_row(m, show_trace=False):
    """
    Analyze all descents from row 0 to row m.
    
    Args:
        m: Row depth (determines starting coefficient a = 2^m)
        show_trace: Whether to show step-by-step trace for each result
    """
    initial_a = 2 ** m
    modulus = 2 ** m
    
    print(f"\n{'='*70}")
    print(f"COLLATZ TREE ROW ANALYSIS: m = {m}")
    print(f"{'='*70}")
    print(f"Starting form: N = {initial_a}k + b (where a = 2^{m} = {initial_a})")
    print(f"Testing b from 0 to {modulus - 1} ({modulus} residue classes)")
    print(f"k represents any non-negative integer (k ≥ 0)")
    
    results = []
    
    # Simulate descent for each b value
    for b in range(modulus):
        result = simulate_descent(initial_a, b)
        results.append(result)
        print_result(result, show_trace)
    
    # Summary statistics
    print(f"\n{'='*70}")
    print(f"SUMMARY FOR ROW m = {m}")
    print(f"{'='*70}")
    
    # Count unique final coefficients
    final_coefficients = set(r['final_a'] for r in results)
    print(f"Unique final coefficients (odd values): {sorted(final_coefficients)}")
    print(f"Number of distinct final coefficients: {len(final_coefficients)}")
    
    # Group by final coefficient
    print(f"\nMapping by final coefficient:")
    for coef in sorted(final_coefficients):
        b_values = [r['initial_b'] for r in results if r['final_a'] == coef]
        print(f"  {coef}k + b: b ∈ {{{', '.join(map(str, b_values))}}}")
    
    # Path statistics
    path_lengths = {}
    for r in results:
        path = r['path']
        if path not in path_lengths:
            path_lengths[path] = []
        path_lengths[path].append(r['initial_b'])
    
    print(f"\nPath distribution:")
    for path in sorted(path_lengths.keys()):
        b_values = path_lengths[path]
        print(f"  {path}: b ∈ {{{', '.join(map(str, b_values))}}}")
    
    return results


def main():
    """Main function with example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Collatz Direct Descent Simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python collatz_descent.py --m 3
  python collatz_descent.py --m 4 --trace
  python collatz_descent.py --m 5
        """
    )
    parser.add_argument('--m', type=int, required=True,
                        help='Row depth (m value, determines a = 2^m)')
    parser.add_argument('--trace', action='store_true',
                        help='Show step-by-step trace for each descent')
    
    args = parser.parse_args()
    
    if args.m < 0:
        print("Error: m must be non-negative")
        return
    
    if args.m > 20:
        print("Warning: Large m values may take significant time/memory")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            return
    
    analyze_row(args.m, show_trace=args.trace)


if __name__ == '__main__':
    main()
