# ml-runtime-cpp

**INTRODUCTION :-**

I started this project to move away from using ML frameworks as mere black boxes. I wanted to know what actually happens when I call matmul or autograd.

My goal is NOT to compete with PyTorch, ONNX Runtime, or TensorFlow, but to understand how such systems are built at the lowest level.

But I *will* try to get as close as I can to PyTorch in terms of performance, at least in one specific area...

**CURRENT STATE :-**

A Tensor struct with :

-> Explicit shape handling

-> Manual memory allocation

Basic numerical operations :

-> Elementwise add

-> **2D** matrix multiplication (will generalize to higher dimensions later)

**NEXT ON MY TO-DO LIST :-**

-> Static execution graphs

-> Operator nodes and graph traversal

-> Gradient tracking and basic autograd

Basically a simple horizontal slice that I can layer features on top of in the future without getting scope creeped to the point of abandonment.

**DESIGN NOTES :-**

Using raw pointers as of yet just as a learning experience. Cuz you need to understand the problem (manual memory allocation/freeing troubles) to appreciate the solution (smart pointers).

DONT GIVE UP ON THIS FUTURE ME!
