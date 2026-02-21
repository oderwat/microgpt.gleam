//// Autograd engine — a computation graph ("tape") for reverse-mode automatic
//// differentiation. Each forward operation appends a node to the tape, recording
//// the result, its children, and the local gradients (partial derivatives).
//// The backward pass walks the graph in reverse topological order to compute
//// gradients via the chain rule.
////
//// Port of the Value class from Karpathy's microgpt.py.

import gleam/dict.{type Dict}
import gleam/float
import gleam/int
import gleam/list
import gleam/result
import gleam/set.{type Set}

/// Direct Erlang math calls — bypass gleam@float {ok, V} wrapper overhead.
@external(erlang, "math", "exp")
pub fn math_exp(x: Float) -> Float

@external(erlang, "math", "log")
pub fn math_log(x: Float) -> Float

@external(erlang, "math", "pow")
pub fn math_pow(base: Float, exponent: Float) -> Float

@external(erlang, "math", "sqrt")
pub fn math_sqrt(x: Float) -> Float

// --- Types ---

pub type Node {
  Node(data: Float, children: List(Int), local_grads: List(Float))
}

/// The tape is an append-only graph of computation nodes.
pub type Tape {
  Tape(nodes: Dict(Int, Node), next_id: Int)
}

// --- Constructors ---

pub fn new() -> Tape {
  Tape(nodes: dict.new(), next_id: 0)
}

/// Create a new leaf variable (parameter or input) on the tape.
pub fn var(tape: Tape, data: Float) -> #(Tape, Int) {
  let id = tape.next_id
  let node = Node(data: data, children: [], local_grads: [])
  #(Tape(nodes: dict.insert(tape.nodes, id, node), next_id: id + 1), id)
}

/// Create a constant (not a parameter, but needed for arithmetic).
pub fn constant(tape: Tape, data: Float) -> #(Tape, Int) {
  var(tape, data)
}

// --- Accessors ---

pub fn data(tape: Tape, id: Int) -> Float {
  let assert Ok(node) = dict.get(tape.nodes, id)
  node.data
}

/// Update the data of a node (used by optimizer to update parameters).
pub fn set_data(tape: Tape, id: Int, new_data: Float) -> Tape {
  let assert Ok(node) = dict.get(tape.nodes, id)
  Tape(..tape, nodes: dict.insert(tape.nodes, id, Node(..node, data: new_data)))
}

// --- Forward operations ---

fn push(
  tape: Tape,
  data: Float,
  children: List(Int),
  local_grads: List(Float),
) -> #(Tape, Int) {
  let id = tape.next_id
  let node = Node(data: data, children: children, local_grads: local_grads)
  #(Tape(nodes: dict.insert(tape.nodes, id, node), next_id: id + 1), id)
}

pub fn add(tape: Tape, a: Int, b: Int) -> #(Tape, Int) {
  let da = data(tape, a)
  let db = data(tape, b)
  push(tape, da +. db, [a, b], [1.0, 1.0])
}

pub fn mul(tape: Tape, a: Int, b: Int) -> #(Tape, Int) {
  let da = data(tape, a)
  let db = data(tape, b)
  push(tape, da *. db, [a, b], [db, da])
}

/// Raise a tape variable to a constant power: a^p
pub fn pow(tape: Tape, a: Int, p: Float) -> #(Tape, Int) {
  let da = data(tape, a)
  push(tape, math_pow(da, p), [a], [p *. math_pow(da, p -. 1.0)])
}

pub fn log(tape: Tape, a: Int) -> #(Tape, Int) {
  let da = data(tape, a)
  push(tape, math_log(da), [a], [1.0 /. da])
}

pub fn exp(tape: Tape, a: Int) -> #(Tape, Int) {
  let da = data(tape, a)
  let e = math_exp(da)
  push(tape, e, [a], [e])
}

pub fn relu(tape: Tape, a: Int) -> #(Tape, Int) {
  let da = data(tape, a)
  let out = float.max(0.0, da)
  let grad = case da >. 0.0 {
    True -> 1.0
    False -> 0.0
  }
  push(tape, out, [a], [grad])
}

pub fn neg(tape: Tape, a: Int) -> #(Tape, Int) {
  let da = data(tape, a)
  push(tape, 0.0 -. da, [a], [-1.0])
}

pub fn sub(tape: Tape, a: Int, b: Int) -> #(Tape, Int) {
  let da = data(tape, a)
  let db = data(tape, b)
  push(tape, da -. db, [a, b], [1.0, -1.0])
}

pub fn div(tape: Tape, a: Int, b: Int) -> #(Tape, Int) {
  let da = data(tape, a)
  let db = data(tape, b)
  // d(a/b)/da = 1/b,  d(a/b)/db = -a/b²
  push(tape, da /. db, [a, b], [
    1.0 /. db,
    0.0 -. da /. { db *. db },
  ])
}

/// Multiply a tape variable by a constant float — true primitive, no extra nodes.
pub fn scale(tape: Tape, a: Int, s: Float) -> #(Tape, Int) {
  let da = data(tape, a)
  push(tape, da *. s, [a], [s])
}

// --- Composite operations ---

/// Sum a list of tape variables.
pub fn sum(tape: Tape, ids: List(Int)) -> #(Tape, Int) {
  case ids {
    [] -> var(tape, 0.0)
    [x] -> #(tape, x)
    [x, ..rest] -> {
      let #(tape, rest_sum) = sum(tape, rest)
      add(tape, x, rest_sum)
    }
  }
}

/// Dot product of two equal-length lists of tape variables.
pub fn dot(tape: Tape, a: List(Int), b: List(Int)) -> #(Tape, Int) {
  let #(tape, products) = dot_mul(tape, a, b, [])
  sum(tape, products)
}

fn dot_mul(
  tape: Tape,
  a: List(Int),
  b: List(Int),
  acc: List(Int),
) -> #(Tape, List(Int)) {
  case a, b {
    [ai, ..as_], [bi, ..bs] -> {
      let #(tape, p) = mul(tape, ai, bi)
      dot_mul(tape, as_, bs, [p, ..acc])
    }
    _, _ -> #(tape, list.reverse(acc))
  }
}

/// Linear layer: y = W @ x, where W is List(List(Int)) [nout x nin], x is List(Int) [nin].
pub fn linear(
  tape: Tape,
  x: List(Int),
  w: List(List(Int)),
) -> #(Tape, List(Int)) {
  map_tape(tape, w, fn(tape, row) { dot(tape, row, x) })
}

/// Softmax with numerical stability (subtract max before exp).
pub fn softmax(tape: Tape, logits: List(Int)) -> #(Tape, List(Int)) {
  let max_val =
    list.fold(logits, -1.0e30, fn(acc, id) { float.max(acc, data(tape, id)) })
  let #(tape, max_id) = constant(tape, max_val)
  let #(tape, shifted) =
    map_tape(tape, logits, fn(tape, id) { sub(tape, id, max_id) })
  let #(tape, exps) = map_tape(tape, shifted, fn(tape, id) { exp(tape, id) })
  let #(tape, total) = sum(tape, exps)
  map_tape(tape, exps, fn(tape, id) { div(tape, id, total) })
}

/// RMSNorm: x_i * (mean(x^2) + eps)^(-0.5)
pub fn rmsnorm(tape: Tape, x: List(Int)) -> #(Tape, List(Int)) {
  let n = int_to_float(list.length(x))
  let #(tape, squares) = map_tape(tape, x, fn(tape, id) { mul(tape, id, id) })
  let #(tape, sq_sum) = sum(tape, squares)
  let #(tape, ms) = scale(tape, sq_sum, 1.0 /. n)
  let #(tape, eps) = constant(tape, 1.0e-5)
  let #(tape, ms_eps) = add(tape, ms, eps)
  let #(tape, inv_rms) = pow(tape, ms_eps, -0.5)
  map_tape(tape, x, fn(tape, xi) { mul(tape, xi, inv_rms) })
}

// --- Backward pass ---

/// Opaque gradient table (Dict-backed, compatible with tape_ets API)
pub type GradTable =
  Dict(Int, Float)

/// Compute gradients of all nodes reachable from `root` via reverse-mode autodiff.
pub fn backward(tape: Tape, root: Int) -> GradTable {
  let topo = topo_sort(tape, root)
  let grads = dict.from_list([#(root, 1.0)])
  list.fold(topo, grads, fn(grads, id) {
    let assert Ok(node) = dict.get(tape.nodes, id)
    let g = result.unwrap(dict.get(grads, id), 0.0)
    acc_grads(grads, node.children, node.local_grads, g)
  })
}

fn acc_grads(
  grads: GradTable,
  children: List(Int),
  local_grads: List(Float),
  g: Float,
) -> GradTable {
  case children, local_grads {
    [c, ..cs], [lg, ..lgs] -> {
      let child_grad = result.unwrap(dict.get(grads, c), 0.0)
      let grads = dict.insert(grads, c, child_grad +. lg *. g)
      acc_grads(grads, cs, lgs, g)
    }
    _, _ -> grads
  }
}

/// Get gradient for a specific variable, defaulting to 0.0.
pub fn grad(grads: GradTable, id: Int) -> Float {
  result.unwrap(dict.get(grads, id), 0.0)
}

// --- Topological sort ---

/// Returns nodes in reverse topological order (root first, leaves last).
fn topo_sort(tape: Tape, root: Int) -> List(Int) {
  let #(_, order) = topo_visit(tape, root, set.new(), [])
  order
}

fn topo_visit(
  tape: Tape,
  id: Int,
  visited: Set(Int),
  order: List(Int),
) -> #(Set(Int), List(Int)) {
  case set.contains(visited, id) {
    True -> #(visited, order)
    False -> {
      let visited = set.insert(visited, id)
      let assert Ok(node) = dict.get(tape.nodes, id)
      let #(visited, order) =
        topo_visit_children(tape, node.children, visited, order)
      #(visited, [id, ..order])
    }
  }
}

fn topo_visit_children(
  tape: Tape,
  children: List(Int),
  visited: Set(Int),
  order: List(Int),
) -> #(Set(Int), List(Int)) {
  case children {
    [] -> #(visited, order)
    [c, ..cs] -> {
      let #(visited, order) = topo_visit(tape, c, visited, order)
      topo_visit_children(tape, cs, visited, order)
    }
  }
}

// --- Reset tape for next training step ---

/// Clear computation graph, keeping only parameter nodes 0..num_params-1.
/// Rebuilds a fresh tape with just the param data values.
pub fn reset(tape: Tape, num_params: Int) -> Tape {
  reset_loop(tape, new(), 0, num_params)
}

fn reset_loop(old: Tape, new_tape: Tape, id: Int, num_params: Int) -> Tape {
  case id >= num_params {
    True -> new_tape
    False -> {
      let d = data(old, id)
      let #(new_tape, _) = var(new_tape, d)
      reset_loop(old, new_tape, id + 1, num_params)
    }
  }
}

// --- Utility: thread tape through a list (like mapAccumL) ---

pub fn map_tape(
  tape: Tape,
  items: List(a),
  f: fn(Tape, a) -> #(Tape, b),
) -> #(Tape, List(b)) {
  let #(tape, rev_results) =
    list.fold(items, #(tape, []), fn(acc, item) {
      let #(tape, results) = acc
      let #(tape, result) = f(tape, item)
      #(tape, [result, ..results])
    })
  #(tape, list.reverse(rev_results))
}

/// Thread tape through two parallel lists without allocating a zip list.
pub fn zip_map_tape(
  tape: Tape,
  as_: List(a),
  bs: List(b),
  f: fn(Tape, a, b) -> #(Tape, c),
) -> #(Tape, List(c)) {
  let #(tape, rev) = zip_map_tape_loop(tape, as_, bs, f, [])
  #(tape, list.reverse(rev))
}

fn zip_map_tape_loop(
  tape: Tape,
  as_: List(a),
  bs: List(b),
  f: fn(Tape, a, b) -> #(Tape, c),
  acc: List(c),
) -> #(Tape, List(c)) {
  case as_, bs {
    [a, ..ar], [b, ..br] -> {
      let #(tape, r) = f(tape, a, b)
      zip_map_tape_loop(tape, ar, br, f, [r, ..acc])
    }
    _, _ -> #(tape, acc)
  }
}

/// Thread tape through a list, also passing the index.
pub fn indexed_map_tape(
  tape: Tape,
  items: List(a),
  f: fn(Tape, a, Int) -> #(Tape, b),
) -> #(Tape, List(b)) {
  let #(tape, _, rev_results) =
    list.fold(items, #(tape, 0, []), fn(acc, item) {
      let #(tape, idx, results) = acc
      let #(tape, result) = f(tape, item, idx)
      #(tape, idx + 1, [result, ..results])
    })
  #(tape, list.reverse(rev_results))
}

// --- Helpers ---

fn int_to_float(n: Int) -> Float {
  int.to_float(n)
}
