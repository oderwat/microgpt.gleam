//// microgpt.gleam — The most atomic way to train and run inference for a GPT
//// in pure, dependency-free Gleam. Port of Karpathy's microgpt.py.
////
//// Everything is here: dataset loading, tokenizer, model architecture (GPT-2
//// style with RMSNorm, multi-head attention, ReLU MLP), Adam optimizer,
//// training loop, and inference with temperature sampling.

import gleam/dict.{type Dict}
import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/result
import gleam/string
import gleam_community/maths
import microgpt/tape.{type Tape}
import simplifile

// --- Hyperparameters ---

const n_layer = 1

const n_embd = 16

const block_size = 16

const n_head = 4

const head_dim = 4

// n_embd / n_head

const num_steps = 1000

const learning_rate = 0.01

const beta1 = 0.85

const beta2 = 0.99

const eps_adam = 1.0e-8

const gauss_std = 0.08

const temperature = 0.5

// --- Random number utilities (using float.random / erlang rand) ---

type RandAlgo {
  Exsss
}

@external(erlang, "rand", "seed")
fn seed_random(algo: RandAlgo, seed: Int) -> Nil

/// Box-Muller transform: generate a gaussian random number.
fn gauss(mean: Float, std: Float) -> Float {
  let u1 = float.random()
  let u2 = float.random()
  let z =
    tape.math_sqrt(-2.0 *. tape.math_log(u1))
    *. maths.cos(2.0 *. maths.pi() *. u2)
  mean +. std *. z
}

/// Weighted random choice: pick an index given a list of weights.
fn weighted_choice(weights: List(Float)) -> Int {
  let total = float.sum(weights)
  let r = float.random() *. total
  weighted_pick(weights, r, 0)
}

fn weighted_pick(weights: List(Float), r: Float, idx: Int) -> Int {
  case weights {
    [] -> int.max(0, idx - 1)
    [w, ..rest] ->
      case r <=. w {
        True -> idx
        False -> weighted_pick(rest, r -. w, idx + 1)
      }
  }
}

/// Shuffle a list using random sort keys.
fn shuffle_list(items: List(a)) -> List(a) {
  items
  |> list.map(fn(x) { #(float.random(), x) })
  |> list.sort(fn(a, b) { float.compare(a.0, b.0) })
  |> list.map(fn(pair) { pair.1 })
}

// --- Helpers ---

fn int_to_float(n: Int) -> Float {
  int.to_float(n)
}

fn float_round(f: Float, decimals: Int) -> Float {
  let factor = tape.math_pow(10.0, int_to_float(decimals))
  int.to_float(float.round(f *. factor)) /. factor
}

/// Generate a list [from, from+1, ..., to] inclusive. Empty if from > to.
fn range(from: Int, to: Int) -> List(Int) {
  case from > to {
    True -> []
    False -> range_loop(from, to, [])
  }
}

fn range_loop(current: Int, stop: Int, acc: List(Int)) -> List(Int) {
  case current > stop {
    True -> list.reverse(acc)
    False -> range_loop(current + 1, stop, [current, ..acc])
  }
}

/// Index into a list (0-based). Panics on out of bounds.
fn nth(items: List(Int), idx: Int) -> Int {
  let assert [x, ..] = list.drop(items, idx)
  x
}

// --- State dict: named weight matrices ---

type StateDict =
  Dict(String, List(List(Int)))

// --- KV cache: per-layer list of key/value vectors ---

type KVCache =
  Dict(Int, List(List(Int)))

// --- Main ---

pub fn main() {
  seed_random(Exsss, 69)

  // --- Dataset ---
  let assert Ok(content) = simplifile.read("input.txt")
  let docs =
    string.split(content, "\n")
    |> list.filter(fn(line) { string.trim(line) != "" })
    |> list.map(string.trim)
  let docs = shuffle_list(docs)
  io.println("num docs: " <> int.to_string(list.length(docs)))

  // --- Tokenizer ---
  let all_chars =
    docs
    |> list.flat_map(string.to_graphemes)
    |> list.unique()
    |> list.sort(string.compare)
  let bos = list.length(all_chars)
  let vocab_size = list.length(all_chars) + 1
  io.println("vocab size: " <> int.to_string(vocab_size))

  let char_to_idx =
    list.index_map(all_chars, fn(ch, i) { #(ch, i) })
    |> dict.from_list()
  let idx_to_char =
    list.index_map(all_chars, fn(ch, i) { #(i, ch) })
    |> dict.from_list()

  // --- Initialize parameters on the tape ---
  let t = tape.new()
  let #(t, state, params) = init_params(t, vocab_size)
  io.println("num params: " <> int.to_string(list.length(params)))

  // --- Adam optimizer buffers ---
  let n_params = list.length(params)
  let adam_m = list.repeat(0.0, n_params)
  let adam_v = list.repeat(0.0, n_params)

  // --- Training loop ---
  let #(t, _params, _adam_m, _adam_v) =
    list.fold(
      range(0, num_steps - 1),
      #(t, params, adam_m, adam_v),
      fn(acc, step) {
        let #(t, params, adam_m, adam_v) = acc
        train_step(
          t,
          state,
          params,
          adam_m,
          adam_v,
          docs,
          step,
          char_to_idx,
          bos,
        )
      },
    )

  // --- Inference ---
  io.println("")
  io.println("--- inference (new, hallucinated names) ---")
  inference(t, state, bos, idx_to_char, 20, 0)
}

// --- Parameter initialization ---

fn init_params(t: Tape, vocab_size: Int) -> #(Tape, StateDict, List(Int)) {
  let #(t, wte) = make_matrix(t, vocab_size, n_embd)
  let #(t, wpe) = make_matrix(t, block_size, n_embd)
  let #(t, lm_head) = make_matrix(t, vocab_size, n_embd)

  let state =
    dict.new()
    |> dict.insert("wte", wte)
    |> dict.insert("wpe", wpe)
    |> dict.insert("lm_head", lm_head)

  let #(t, state) =
    list.fold(range(0, n_layer - 1), #(t, state), fn(acc, i) {
      let #(t, state) = acc
      let prefix = "layer" <> int.to_string(i) <> "."
      let #(t, attn_wq) = make_matrix(t, n_embd, n_embd)
      let #(t, attn_wk) = make_matrix(t, n_embd, n_embd)
      let #(t, attn_wv) = make_matrix(t, n_embd, n_embd)
      let #(t, attn_wo) = make_matrix(t, n_embd, n_embd)
      let #(t, mlp_fc1) = make_matrix(t, 4 * n_embd, n_embd)
      let #(t, mlp_fc2) = make_matrix(t, n_embd, 4 * n_embd)
      let state =
        state
        |> dict.insert(prefix <> "attn_wq", attn_wq)
        |> dict.insert(prefix <> "attn_wk", attn_wk)
        |> dict.insert(prefix <> "attn_wv", attn_wv)
        |> dict.insert(prefix <> "attn_wo", attn_wo)
        |> dict.insert(prefix <> "mlp_fc1", mlp_fc1)
        |> dict.insert(prefix <> "mlp_fc2", mlp_fc2)
      #(t, state)
    })

  let params = flatten_state_dict(state)
  #(t, state, params)
}

fn make_matrix(t: Tape, nrows: Int, ncols: Int) -> #(Tape, List(List(Int))) {
  tape.map_tape(t, range(0, nrows - 1), fn(t, _row) {
    tape.map_tape(t, range(0, ncols - 1), fn(t, _col) {
      tape.var(t, gauss(0.0, gauss_std))
    })
  })
}

/// Flatten all parameter IDs from the state dict into a sorted list.
/// Sorting is critical: rebuild_tape creates new IDs 0, 1, 2, ... in the
/// order it receives them. Since state still references the original IDs
/// from init_params (which are sequential: 0, 1, 2, ...), we must return
/// them in ascending order so that rebuild_tape maps old ID i → new ID i.
fn flatten_state_dict(state: StateDict) -> List(Int) {
  dict.values(state)
  |> list.flat_map(fn(mat) { list.flat_map(mat, fn(row) { row }) })
  |> list.sort(int.compare)
}

// --- GPT forward pass ---

fn gpt_forward(
  t: Tape,
  state: StateDict,
  token_id: Int,
  pos_id: Int,
  keys: KVCache,
  values: KVCache,
) -> #(Tape, List(Int), KVCache, KVCache) {
  let assert Ok(wte) = dict.get(state, "wte")
  let assert Ok(wpe) = dict.get(state, "wpe")
  let assert Ok(lm_head) = dict.get(state, "lm_head")

  // Token + position embedding
  let assert [tok_emb, ..] = list.drop(wte, token_id)
  let assert [pos_emb, ..] = list.drop(wpe, pos_id)
  let #(t, x) =
    tape.zip_map_tape(t, tok_emb, pos_emb, fn(t, a, b) { tape.add(t, a, b) })

  // Initial RMSNorm
  let #(t, x) = tape.rmsnorm(t, x)

  // Transformer layers
  let #(t, x, keys, values) =
    list.fold(range(0, n_layer - 1), #(t, x, keys, values), fn(acc, li) {
      let #(t, x, keys, values) = acc
      transformer_layer(t, state, x, li, keys, values)
    })

  // Final logits
  let #(t, logits) = tape.linear(t, x, lm_head)
  #(t, logits, keys, values)
}

fn transformer_layer(
  t: Tape,
  state: StateDict,
  x: List(Int),
  layer_idx: Int,
  keys: KVCache,
  values: KVCache,
) -> #(Tape, List(Int), KVCache, KVCache) {
  let prefix = "layer" <> int.to_string(layer_idx) <> "."
  let assert Ok(attn_wq) = dict.get(state, prefix <> "attn_wq")
  let assert Ok(attn_wk) = dict.get(state, prefix <> "attn_wk")
  let assert Ok(attn_wv) = dict.get(state, prefix <> "attn_wv")
  let assert Ok(attn_wo) = dict.get(state, prefix <> "attn_wo")
  let assert Ok(mlp_fc1) = dict.get(state, prefix <> "mlp_fc1")
  let assert Ok(mlp_fc2) = dict.get(state, prefix <> "mlp_fc2")

  // --- Multi-head Attention ---
  let x_residual = x
  let #(t, xn) = tape.rmsnorm(t, x)
  let #(t, q) = tape.linear(t, xn, attn_wq)
  let #(t, k) = tape.linear(t, xn, attn_wk)
  let #(t, v) = tape.linear(t, xn, attn_wv)

  // Append k, v to cache (prepend + reverse when consumed would be more
  // efficient, but the cache is small — max block_size=16 entries)
  let layer_keys = result.unwrap(dict.get(keys, layer_idx), [])
  let layer_values = result.unwrap(dict.get(values, layer_idx), [])
  let layer_keys = list.append(layer_keys, [k])
  let layer_values = list.append(layer_values, [v])
  let keys = dict.insert(keys, layer_idx, layer_keys)
  let values = dict.insert(values, layer_idx, layer_values)

  // Multi-head attention
  let #(t, rev_x_attn) =
    list.fold(range(0, n_head - 1), #(t, []), fn(acc, h) {
      let #(t, collected) = acc
      let hs = h * head_dim
      let q_h = list.drop(q, hs) |> list.take(head_dim)
      let #(t, head_out) = attention_head(t, q_h, layer_keys, layer_values, hs)
      // Prepend reversed head_out, reverse all at the end
      #(t, list.append(list.reverse(head_out), collected))
    })
  let x_attn = list.reverse(rev_x_attn)

  let #(t, x_out) = tape.linear(t, x_attn, attn_wo)

  // Residual connection
  let #(t, x) =
    tape.zip_map_tape(t, x_out, x_residual, fn(t, a, b) { tape.add(t, a, b) })

  // --- MLP ---
  let x_residual = x
  let #(t, xn) = tape.rmsnorm(t, x)
  let #(t, h) = tape.linear(t, xn, mlp_fc1)
  let #(t, h) = tape.map_tape(t, h, fn(t, id) { tape.relu(t, id) })
  let #(t, h) = tape.linear(t, h, mlp_fc2)

  // Residual connection
  let #(t, x) =
    tape.zip_map_tape(t, h, x_residual, fn(t, a, b) { tape.add(t, a, b) })

  #(t, x, keys, values)
}

fn attention_head(
  t: Tape,
  q_h: List(Int),
  all_keys: List(List(Int)),
  all_values: List(List(Int)),
  head_start: Int,
) -> #(Tape, List(Int)) {
  let scale = tape.math_sqrt(int_to_float(head_dim))

  // Compute attention logits: dot(q_h, k_h[t]) / sqrt(head_dim) for each t
  let #(t, attn_logits) =
    tape.map_tape(t, all_keys, fn(t, k_full) {
      let k_h = list.drop(k_full, head_start) |> list.take(head_dim)
      let #(t, d) = tape.dot(t, q_h, k_h)
      tape.scale(t, d, 1.0 /. scale)
    })

  // Softmax
  let #(t, attn_weights) = tape.softmax(t, attn_logits)

  // Weighted sum of value vectors
  let v_heads =
    list.map(all_values, fn(v_full) {
      list.drop(v_full, head_start) |> list.take(head_dim)
    })

  // head_out[j] = sum_t(attn_weights[t] * v_heads[t][j])
  tape.map_tape(t, range(0, head_dim - 1), fn(t, j) {
    let #(t, terms) =
      tape.zip_map_tape(t, attn_weights, v_heads, fn(t, w, v_h) {
        let vj = nth(v_h, j)
        tape.mul(t, w, vj)
      })
    tape.sum(t, terms)
  })
}

// --- Training step ---

fn train_step(
  t: Tape,
  state: StateDict,
  params: List(Int),
  adam_m: List(Float),
  adam_v: List(Float),
  docs: List(String),
  step: Int,
  char_to_idx: Dict(String, Int),
  bos: Int,
) -> #(Tape, List(Int), List(Float), List(Float)) {
  let doc_idx = step % list.length(docs)
  let assert [doc, ..] = list.drop(docs, doc_idx)

  // Tokenize: [BOS] ++ chars ++ [BOS]
  let char_tokens =
    string.to_graphemes(doc)
    |> list.map(fn(ch) {
      let assert Ok(idx) = dict.get(char_to_idx, ch)
      idx
    })
  let tokens = list.flatten([[bos], char_tokens, [bos]])
  let n = int.min(block_size, list.length(tokens) - 1)

  // Reset the tape: delete computation graph nodes, keep params 0..n_params-1
  let t = tape.reset(t, list.length(params))
  let new_params = params

  // Forward pass: process each position
  let empty_cache = dict.new()
  let #(t, rev_losses, _keys, _values) =
    list.fold(
      range(0, n - 1),
      #(t, [], empty_cache, empty_cache),
      fn(acc, pos_id) {
        let #(t, losses, keys, values) = acc
        let token_id = nth(tokens, pos_id)
        let target_id = nth(tokens, pos_id + 1)
        let #(t, logits, keys, values) =
          gpt_forward(t, state, token_id, pos_id, keys, values)
        let #(t, probs) = tape.softmax(t, logits)
        let target_prob = nth(probs, target_id)
        let #(t, log_prob) = tape.log(t, target_prob)
        let #(t, loss_t) = tape.neg(t, log_prob)
        #(t, [loss_t, ..losses], keys, values)
      },
    )
  let losses = list.reverse(rev_losses)

  // Average loss
  let #(t, loss_sum) = tape.sum(t, losses)
  let #(t, loss) = tape.scale(t, loss_sum, 1.0 /. int_to_float(n))

  // Print progress
  let loss_val = tape.data(t, loss)
  io.print(
    "\rstep "
    <> string.pad_start(int.to_string(step + 1), 4, " ")
    <> " / "
    <> int.to_string(num_steps)
    <> " | loss "
    <> float.to_string(float_round(loss_val, 4)),
  )

  // Backward pass
  let grads = tape.backward(t, loss)

  // Adam update
  let lr_t =
    learning_rate *. { 1.0 -. int_to_float(step) /. int_to_float(num_steps) }
  let step_f = int_to_float(step + 1)

  let #(t, rev_params, rev_m, rev_v) =
    adam_update(t, grads, lr_t, step_f, new_params, adam_m, adam_v, [], [], [])

  #(t, list.reverse(rev_params), list.reverse(rev_m), list.reverse(rev_v))
}

fn adam_update(
  t: Tape,
  grads: tape.GradTable,
  lr_t: Float,
  step_f: Float,
  params: List(Int),
  ms: List(Float),
  vs: List(Float),
  acc_p: List(Int),
  acc_m: List(Float),
  acc_v: List(Float),
) -> #(Tape, List(Int), List(Float), List(Float)) {
  case params, ms, vs {
    [p, ..pr], [m_old, ..mr], [v_old, ..vr] -> {
      let g = tape.grad(grads, p)
      let m_new = beta1 *. m_old +. { 1.0 -. beta1 } *. g
      let v_new = beta2 *. v_old +. { 1.0 -. beta2 } *. g *. g
      let m_hat = m_new /. { 1.0 -. tape.math_pow(beta1, step_f) }
      let v_hat = v_new /. { 1.0 -. tape.math_pow(beta2, step_f) }
      let p_data = tape.data(t, p)
      let new_data =
        p_data -. lr_t *. m_hat /. { tape.math_sqrt(v_hat) +. eps_adam }
      let t = tape.set_data(t, p, new_data)
      adam_update(
        t,
        grads,
        lr_t,
        step_f,
        pr,
        mr,
        vr,
        [p, ..acc_p],
        [m_new, ..acc_m],
        [v_new, ..acc_v],
      )
    }
    _, _, _ -> #(t, acc_p, acc_m, acc_v)
  }
}

// --- Inference ---

fn inference(
  t: Tape,
  state: StateDict,
  bos: Int,
  idx_to_char: Dict(Int, String),
  num_samples: Int,
  sample_idx: Int,
) {
  case sample_idx >= num_samples {
    True -> Nil
    False -> {
      let #(t, sample) = generate_one(t, state, bos, idx_to_char)
      io.println(
        "sample "
        <> string.pad_start(int.to_string(sample_idx + 1), 2, " ")
        <> ": "
        <> sample,
      )
      inference(t, state, bos, idx_to_char, num_samples, sample_idx + 1)
    }
  }
}

fn generate_one(
  t: Tape,
  state: StateDict,
  bos: Int,
  idx_to_char: Dict(Int, String),
) -> #(Tape, String) {
  let n_params = list.length(flatten_state_dict(state))
  let t = tape.reset(t, n_params)
  generate_loop(t, state, bos, idx_to_char, bos, 0, dict.new(), dict.new(), "")
}

fn generate_loop(
  t: Tape,
  state: StateDict,
  bos: Int,
  idx_to_char: Dict(Int, String),
  token_id: Int,
  pos_id: Int,
  keys: KVCache,
  values: KVCache,
  acc: String,
) -> #(Tape, String) {
  case pos_id >= block_size {
    True -> #(t, acc)
    False -> {
      let #(t, logits, keys, values) =
        gpt_forward(t, state, token_id, pos_id, keys, values)
      // Temperature scaling
      let #(t, scaled) =
        tape.map_tape(t, logits, fn(t, l) {
          tape.scale(t, l, 1.0 /. temperature)
        })
      let #(t, probs) = tape.softmax(t, scaled)
      let weights = list.map(probs, fn(id) { tape.data(t, id) })
      let next_token = weighted_choice(weights)
      case next_token == bos {
        True -> #(t, acc)
        False -> {
          let ch = result.unwrap(dict.get(idx_to_char, next_token), "?")
          generate_loop(
            t,
            state,
            bos,
            idx_to_char,
            next_token,
            pos_id + 1,
            keys,
            values,
            acc <> ch,
          )
        }
      }
    }
  }
}
