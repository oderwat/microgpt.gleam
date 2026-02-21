# microgpt.gleam

A port of [Karpathy's microgpt.py](https://github.com/karpathy/microgpt) to
[Gleam](https://gleam.run/) — the most atomic way to train and run inference
for a GPT, now in a functional language on the BEAM.

## What it does

Trains a tiny GPT-2 style character-level language model on a list of names
(`input.txt`), then generates new, hallucinated names via temperature sampling.
The entire algorithm — dataset loading, tokenizer, model (multi-head attention,
RMSNorm, ReLU MLP), Adam optimizer, training loop, and inference — lives in two
files with zero ML framework dependencies.

## How it works with Gleam's immutable data

The central challenge of porting an autograd engine to Gleam is that all data is
immutable. In Python, `Value` objects form a mutable graph where each node holds
its data, gradient, and pointers to children. In Gleam, we use an append-only
**tape** instead.

### The Tape

The tape (`tape.gleam`) is a `Dict(Int, Node)` mapping integer IDs to nodes.
Each `Node` records:

- **data** — the forward-pass result (a `Float`)
- **children** — IDs of the inputs that produced this node
- **local_grads** — partial derivatives with respect to each child

Every forward operation (add, mul, exp, softmax, ...) appends a new node and
returns the updated tape plus the new node's ID. The tape is threaded through
every computation as an accumulator — Gleam's way of managing state without
mutation:

```gleam
let #(tape, a) = tape.var(tape, 3.0)
let #(tape, b) = tape.var(tape, 2.0)
let #(tape, c) = tape.mul(tape, a, b)   // c = 6.0, node 2
let #(tape, d) = tape.add(tape, c, a)   // d = 9.0, node 3
```

Each line consumes the old tape and produces a new one with the additional node.
The `#(Tape, Int)` return type — a tuple of the updated tape and the new node
ID — is the fundamental pattern throughout the codebase.

### Threading state through lists

Many neural network operations apply a function to every element of a list
(map over embedding rows, compute attention for each head, etc.). In a mutable
language you'd just loop and mutate. In Gleam, `map_tape` threads the tape
through as an accumulator:

```gleam
pub fn map_tape(tape, items, f) -> #(Tape, List(b)) {
  list.fold(items, #(tape, []), fn(acc, item) {
    let #(tape, results) = acc
    let #(tape, result) = f(tape, item)
    #(tape, [result, ..results])
  })
  |> fn(pair) { #(pair.0, list.reverse(pair.1)) }
}
```

This is the Gleam equivalent of `mapAccumL` in Haskell — each step sees the
tape with all previously created nodes, and the final tape contains every node
from the entire traversal.

For operations on parallel lists (like adding token and position embeddings
element-wise), `zip_map_tape` does the same without allocating an intermediate
zip list.

### Backward pass

The backward pass computes gradients via reverse-mode automatic differentiation:

1. **Topological sort** from the loss node, walking children recursively
2. **Traverse in reverse topological order** (root first, leaves last)
3. For each node, multiply its gradient by each local gradient and accumulate
   into the child's gradient entry

The gradient table is a `Dict(Int, Float)` — again immutable, rebuilt at each
accumulation step. The chain rule application is a direct recursion over
children and local_grads lists simultaneously, avoiding intermediate list
allocations.

### Tape reset

Between training steps, `tape.reset(tape, num_params)` creates a fresh tape
containing only the parameter nodes (IDs 0 through num_params-1) with their
current data values. All intermediate computation nodes from the previous
forward/backward pass are discarded. This keeps memory bounded without needing
garbage collection of individual nodes.

### Performance considerations

Gleam compiles to Erlang and runs on the BEAM VM. Some choices reflect this:

- **Direct Erlang math calls** — `@external(erlang, "math", "exp")` etc. bypass
  the `gleam/float` wrappers that return `Result(Float, Nil)` tuples. This
  didn't bring a noticeable speedup in practice, but makes the code cleaner
  by avoiding `let assert Ok(v) = float.power(...)` at every call site.
- **Primitive ops** — `sub`, `div`, `neg`, and `scale` are each a single tape
  node rather than being composed from `add`/`mul`/`constant` (which would
  create 2-3 nodes each). This reduces the tape size significantly.
- **No intermediate list allocations** — Hot paths like the backward pass
  accumulation and dot product use direct recursion over parallel lists instead
  of `list.zip` + `list.fold`, which would allocate a throwaway list of tuples.

## Performance

Measured on a Mac mini M4 with 64 GB RAM, training 1000 steps:

| Version | Time |
|---------|------|
| Python (`microgpt.py`) | ~1m 15s |
| Gleam (on BEAM/OTP) | ~1m 11s |

The Gleam version is slightly faster despite using fully immutable data
structures and running on a VM designed for concurrency rather than
number crunching.

## Running

```sh
gleam run
```

This trains for 1000 steps on `input.txt` (a list of names) and then generates
20 new names.

## Project structure

```
src/
  microgpt.gleam       — model, training loop, inference
  microgpt/tape.gleam  — autograd tape engine
microgpt.py            — Karpathy's original Python for reference
input.txt              — training data (one name per line)
```

## Authorship

This port was largely written by Claude Code (Opus 4.6) with human guidance.

## License

MIT — see [LICENSE](LICENSE).
