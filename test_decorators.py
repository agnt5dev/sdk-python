#!/usr/bin/env python3
"""
Test script for AGNT5 function decorators.

This tests the @function decorator and function registry without
requiring the full Rust worker infrastructure.
"""

import sys
import os
import json

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agnt5.decorators import function, get_registered_functions, get_function_metadata, invoke_function


@function("add_numbers")
def add_numbers(ctx, a: int, b: int) -> int:
    """Add two numbers together."""
    print(f"🧮 Adding {a} + {b} = {a + b}")
    return a + b


@function("greet_user")  
def greet_user(ctx, name: str) -> str:
    """Greet a user by name."""
    print(f"👋 Greeting user: {name}")
    return f"Hello, {name}!"


@function()  # Uses function name as handler name
def multiply(ctx, x: float, y: float) -> float:
    """Multiply two numbers."""
    print(f"✖️ Multiplying {x} * {y} = {x * y}")
    return x * y


def test_registration():
    """Test function registration."""
    print("🧪 Testing function registration...")
    
    functions = get_registered_functions()
    print(f"📋 Registered functions: {list(functions.keys())}")
    
    expected_functions = ["add_numbers", "greet_user", "multiply"]
    for func_name in expected_functions:
        if func_name in functions:
            print(f"✅ {func_name} registered")
        else:
            print(f"❌ {func_name} NOT registered")
            return False
    
    print("✅ All functions registered correctly")
    return True


def test_metadata():
    """Test function metadata extraction."""
    print("\n🧪 Testing function metadata...")
    
    functions = get_registered_functions()
    
    for func_name, func in functions.items():
        metadata = get_function_metadata(func)
        print(f"📋 {func_name} metadata: {metadata}")
        
        if not metadata:
            print(f"❌ No metadata for {func_name}")
            return False
            
        if metadata.get('name') != func_name:
            print(f"❌ Wrong name in metadata for {func_name}")
            return False
    
    print("✅ All function metadata extracted correctly")
    return True


def test_invocation():
    """Test function invocation."""
    print("\n🧪 Testing function invocation...")
    
    # Test add_numbers
    print("Testing add_numbers(5, 3)...")
    input_data = json.dumps({"a": 5, "b": 3}).encode('utf-8')
    result = invoke_function("add_numbers", input_data, {"test": True})
    result_obj = json.loads(result.decode('utf-8'))
    
    if result_obj == 8:
        print("✅ add_numbers returned correct result: 8")
    else:
        print(f"❌ add_numbers returned wrong result: {result_obj}")
        return False
    
    # Test greet_user
    print("Testing greet_user('Alice')...")
    input_data = json.dumps({"name": "Alice"}).encode('utf-8')
    result = invoke_function("greet_user", input_data, {"test": True})
    result_obj = json.loads(result.decode('utf-8'))
    
    if result_obj == "Hello, Alice!":
        print("✅ greet_user returned correct result")
    else:
        print(f"❌ greet_user returned wrong result: {result_obj}")
        return False
    
    # Test multiply
    print("Testing multiply(4.5, 2.0)...")
    input_data = json.dumps({"x": 4.5, "y": 2.0}).encode('utf-8')
    result = invoke_function("multiply", input_data, {"test": True})
    result_obj = json.loads(result.decode('utf-8'))
    
    if result_obj == 9.0:
        print("✅ multiply returned correct result: 9.0")
    else:
        print(f"❌ multiply returned wrong result: {result_obj}")
        return False
    
    print("✅ All function invocations worked correctly")
    return True


def main():
    print("🚀 Testing AGNT5 Function Decorators")
    print("=" * 50)
    
    # Run tests
    success = True
    success &= test_registration()
    success &= test_metadata()
    success &= test_invocation()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())