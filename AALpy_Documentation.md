# AALpy Documentation: Automata Generation

AALpy is a versatile Python library designed for active automata learning and the manipulation of finite-state systems. It provides high-level utilities for generating random machines and low-level classes for manual construction.

## 1. Deterministic Finite Automata (DFA)

In AALpy, a **DFA** is composed of `DfaState` objects, where each state has a transition for every symbol in the input alphabet and a boolean flag indicating if it is an accepting state.

### Random Generation
To generate a random DFA, use the `generate_random_dfa` function from the `utils` module.

```python
from aalpy.utils import generate_random_dfa

# Parameters:
# - num_states: Total number of states in the DFA
# - input_alphabet_size: Number of unique symbols in the alphabet (e.g., 2 for ['0', '1'])
random_dfa = generate_random_dfa(num_states=10, input_alphabet_size=2)

# Accessing properties
print(f"Alphabet: {random_dfa.get_input_alphabet()}")
print(f"Number of states: {len(random_dfa.states)}")
```

### Manual Construction
For precise control, you can define states and transitions manually using `DfaState` and `Dfa`.

```python
from aalpy.automata import DfaState, Dfa

# 1. Create states
s0 = DfaState('s0', is_accepting=False)
s1 = DfaState('s1', is_accepting=True)

# 2. Define transitions
s0.transitions['0'] = s0
s0.transitions['1'] = s1
s1.transitions['0'] = s0
s1.transitions['1'] = s1

# 3. Initialize the DFA (initial_state, list_of_states)
my_dfa = Dfa(s0, [s0, s1])
```

---

## 2. Non-deterministic Finite Automata (NFA)

AALpy supports non-deterministic systems through the **Observable Non-deterministic Finite-state Machine (ONFSM)** model. In an ONFSM, an input can lead to multiple states, but the combination of (input, output) must uniquely identify the transition's behavior for it to remain "observable" by learning algorithms.

### Random Generation
Use `generate_random_ONFSM` to create random non-deterministic machines.

```python
from aalpy.utils import generate_random_ONFSM

# Parameters:
# - num_states: Number of states
# - input_size: Size of the input alphabet
# - output_size: Size of the output alphabet (NFA behavior is modeled via output/acceptance)
random_nfa = generate_random_ONFSM(num_states=5, input_size=2, output_size=2)
```

### Manual Construction
Use `OnfsmState` to define non-deterministic transitions. Transitions in an ONFSM are structured as a mapping from input to a list of possible (output, next_state) tuples.

```python
from aalpy.automata import OnfsmState, Onfsm

s0 = OnfsmState('s0')
s1 = OnfsmState('s1')

# s0 on input 'a' can go to s0 with output '0' OR s1 with output '1'
s0.transitions['a'] = [('0', s0), ('1', s1)]

nfa = Onfsm(s0, [s0, s1])
```

---

## 3. Epsilon-NFA ($\epsilon$-NFA)

AALpy is an **active learning** library, meaning it focuses on transitions triggered by observable inputs. It does not have a native `EpsilonNFA` class because $\epsilon$-transitions (internal transitions without input) are typically abstracted away during the learning process.

### Working with $\epsilon$-Transitions in AALpy
To represent an $\epsilon$-NFA for your dataset, you can use the following strategies:

1.  **Special Symbol Representation:**
    Manually treat a specific symbol (e.g., `'epsilon'` or `None`) as the empty transition.
    ```python
    s0.transitions[None] = [('accept', s1)] 
    # This requires you to manually trigger the transition using machine.step(None)
    ```

2.  **Conversion to NFA/DFA:**
    If you are using AALpy to learn from an $\epsilon$-NFA, the library will automatically learn an equivalent **DFA** or **ONFSM** that represents the same observable language but without the internal $\epsilon$-steps.

3.  **Custom SUL (System Under Learning):**
    If your "Black Box" contains epsilon transitions, you can wrap it in a `SUL` interface. AALpy will interact with it by sending inputs; the epsilon transitions will happen internally and will be reflected in the final output or state reachability observed by the learner.

---

## 4. Visualization and Verification

To verify your generated automata (DFA, NFA, or simulated $\epsilon$-NFA), AALpy provides a visualization tool that generates state diagrams.

```python
from aalpy.utils import visualize_automaton

# This will generate a .pdf or .png file (requires Graphviz installed on your system)
visualize_automaton(random_dfa)
```

### Summary of Key Classes
| Type | State Class | Machine Class | Random Utility |
| :--- | :--- | :--- | :--- |
| **DFA** | `DfaState` | `Dfa` | `generate_random_dfa` |
| **NFA** | `OnfsmState` | `Onfsm` | `generate_random_ONFSM` |
| **$\epsilon$-NFA** | N/A | Use `Onfsm` | Manual simulation required |
