#!/usr/bin/env python3
"""
Simple hyperparameter search for Tsetlin Machine - Working version
Streams output instead of capturing to avoid hangs
"""

import subprocess
import re

def run_experiment(board_size=5, number_of_boards=50000, epochs=100, 
                   number_of_clauses=8000, T=12000, s=3.0, depth=3,
                   hypervector_size=512, hypervector_bits=2, 
                   message_size=256, message_bits=2, max_included_literals=32, name=""):
    """Run experiment and return best test accuracy"""
    
    cmd = [
        "uv", "run", "src/swstl.py",
        "--board-size", str(board_size),
        "--number-of-boards", str(number_of_boards),
        "--epochs", str(epochs),
        "--number-of-clauses", str(number_of_clauses),
        "--T", str(T),
        "--s", str(s),
        "--depth", str(depth),
        "--hypervector-size", str(hypervector_size),
        "--hypervector-bits", str(hypervector_bits),
        "--message-size", str(message_size),
        "--message-bits", str(message_bits),
        "--max-included-literals", str(max_included_literals),
    ]

    print(f"\nRunning experiment: {name}", flush=True)
    process = subprocess.Popen(cmd)
    process.wait()




def main():
    print("=" * 70, flush=True)
    print("SIMPLE HYPERPARAMETER SEARCH", flush=True)
    print("=" * 70, flush=True)
    
    # Track overall best and failures
    best_accuracy = 0.0
    best_params = {}
    failed_experiments = []
    total_experiments = 0
    
    # Base parameters
    base = {
        'board_size': 5,
        'number_of_boards': 50000,
        'epochs': 100,
        'number_of_clauses': 8000,
        'T': 12000,
        's': 3.0,
        'depth': 3,
        'hypervector_size': 512,
        'hypervector_bits': 2,
        'message_size': 256,
        'message_bits': 2,
        'max_included_literals': 300,
    }
    
    # Test 1: Number of clauses
    print("\n" + "="*70, flush=True)
    print("Testing number_of_clauses", flush=True)
    print("="*70, flush=True)
    for clauses in [100, 300, 800, 2000, 4000, 8000, 12000, 16000, 20000]:
        total_experiments += 1
        params = {**base, 'number_of_clauses': clauses}
        acc = run_experiment(**params, name=f"clauses_{clauses}")
        if acc is None:
            failed_experiments.append(('number_of_clauses', clauses))
        elif acc > best_accuracy:
            best_accuracy = acc
            best_params = params.copy()
            print(f"  🎯 NEW BEST: {best_accuracy:.2f}%", flush=True)
    
    # Test 2: T value
    print("\n" + "="*70, flush=True)
    print("Testing T", flush=True)
    print("="*70, flush=True)
    for t_val in [100, 500, 2000, 8000, 12000, 16000, 20000]:
        total_experiments += 1
        params = {**base, 'T': t_val}
        acc = run_experiment(**params, name=f"T_{t_val}")
        if acc is None:
            failed_experiments.append(('T', t_val))
        elif acc > best_accuracy:
            best_accuracy = acc
            best_params = params.copy()
            print(f"  🎯 NEW BEST: {best_accuracy:.2f}%", flush=True)
    
    # Test 3: s value
    print("\n" + "="*70, flush=True)
    print("Testing s", flush=True)
    print("="*70, flush=True)
    for s_val in [1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0]:
        total_experiments += 1
        params = {**base, 's': s_val}
        acc = run_experiment(**params, name=f"s_{s_val}")
        if acc is None:
            failed_experiments.append(('s', s_val))
        elif acc > best_accuracy:
            best_accuracy = acc
            best_params = params.copy()
            print(f"  🎯 NEW BEST: {best_accuracy:.2f}%", flush=True)
    
    # Test 4: depth
    print("\n" + "="*70, flush=True)
    print("Testing depth", flush=True)
    print("="*70, flush=True)
    for depth_val in [1, 2, 3, 4, 5]:
        total_experiments += 1
        params = {**base, 'depth': depth_val}
        acc = run_experiment(**params, name=f"depth_{depth_val}")
        if acc is None:
            failed_experiments.append(('depth', depth_val))
        elif acc > best_accuracy:
            best_accuracy = acc
            best_params = params.copy()
            print(f"  🎯 NEW BEST: {best_accuracy:.2f}%", flush=True)
    
    # Test 5: hypervector size
    print("\n" + "="*70, flush=True)
    print("Testing hypervector_size", flush=True)
    print("="*70, flush=True)
    for hv_size in [10, 20, 60, 100, 128, 256, 512, 1024]:
        total_experiments += 1
        params = {**base, 'hypervector_size': hv_size}
        acc = run_experiment(**params, name=f"hv_size_{hv_size}")
        if acc is None:
            failed_experiments.append(('hypervector_size', hv_size))
        elif acc > best_accuracy:
            best_accuracy = acc
            best_params = params.copy()
            print(f"  🎯 NEW BEST: {best_accuracy:.2f}%", flush=True)
    
    # Print results
    print("\n" + "=" * 70, flush=True)
    print("SEARCH SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"Total experiments: {total_experiments}", flush=True)
    print(f"Successful: {total_experiments - len(failed_experiments)}", flush=True)
    print(f"Failed: {len(failed_experiments)}", flush=True)
    
    if failed_experiments:
        print("\n❌ Failed experiments:", flush=True)
        for param_name, param_value in failed_experiments:
            print(f"  {param_name}={param_value}", flush=True)
    
    if best_accuracy > 0:
        print("\n" + "=" * 70, flush=True)
        print("🏆 BEST RESULT", flush=True)
        print("=" * 70, flush=True)
        print(f"Best Test Accuracy: {best_accuracy:.2f}%", flush=True)
        print("\nBest Parameters:", flush=True)
        for key, value in best_params.items():
            print(f"  {key}: {value}", flush=True)
        
        print("\nCommand to reproduce:", flush=True)
        print(f"uv run src/swstl.py \\", flush=True)
        print(f"  --board-size {best_params['board_size']} \\", flush=True)
        print(f"  --number-of-boards {best_params['number_of_boards']} \\", flush=True)
        print(f"  --epochs {best_params['epochs']} \\", flush=True)
        print(f"  --number-of-clauses {best_params['number_of_clauses']} \\", flush=True)
        print(f"  --T {best_params['T']} \\", flush=True)
        print(f"  --s {best_params['s']} \\", flush=True)
        print(f"  --depth {best_params['depth']} \\", flush=True)
        print(f"  --hypervector-size {best_params['hypervector_size']} \\", flush=True)
        print(f"  --hypervector-bits {best_params['hypervector_bits']} \\", flush=True)
        print(f"  --message-size {best_params['message_size']} \\", flush=True)
        print(f"  --message-bits {best_params['message_bits']} \\", flush=True)
        print(f"  --max-included-literals {best_params['max_included_literals']}", flush=True)
    else:
        print("\n⚠️  WARNING: All experiments failed!", flush=True)
        print("Check the error messages above for details.", flush=True)

if __name__ == "__main__":
    main()
