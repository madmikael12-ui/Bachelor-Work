import random
from aalpy.utils import generate_random_dfa, generate_random_ONFSM

Alphabet_size = 2
Max_states = 100

State_tokens = []
Alphabet = []

for i in range (Alphabet_size):
    Alphabet.append(f'input_{i}')

for i in range(Max_states):
    State_tokens.append(f'STATE_{i}')

Special_tokens = ['PAD','epsilon', 'START', 'ACCEPT', 'TRANSITION', 'SEP', '<DETECT>', '<TRANSFORM>', '<CLASS_DFA>', '<CLASS_NFA>', '<CLASS_EPS_NFA>']
Token_combo = State_tokens + Alphabet + Special_tokens

Tokens = {}
for i, token in enumerate(Token_combo):
    Tokens[token] = i

PAD_INDEX = Tokens['PAD']
VOCAB_SIZE = len(Token_combo)

def Automata(type):
    num_states = random.randint(2, Max_states)
    input_alphabet= []
    
    for i in range (Alphabet_size):
        input_alphabet.append(str(i))
    
    if type == 'DFA':
        automaton = generate_random_dfa(num_states=num_states, alphabet=input_alphabet)
        path = 'DFA_Model'
    else:
        automaton = generate_random_ONFSM(num_states=num_states, num_inputs=Alphabet_size, num_outputs=2)
        for state in automaton.states:
            new_transition= {}
            for i in range (Alphabet_size):
                new_transition[str(i)] = state.transitions.get(f'i{i+1}', [])
            state.transitions = new_transition
    
        if type == 'EPS_NFA':
            for state in automaton.states:
                if random.random() < 0.3:
                    random_dest = random.choice(automaton.states)
                    state.transitions['epsilon'] = [('o1', random_dest)]
            path = 'EPS_NFA_Model'
        else:
            path = 'NFA_Model'
    
    state_to_token= {}
    for i, state in enumerate(automaton.states):
        state_to_token[state] = Tokens[f'STATE_{i}']
    
    sequence= [Tokens['START'], state_to_token[automaton.initial_state]]
    
    for state in automaton.states:
        sequence.append(state_to_token[state])
        
        if type == 'DFA':
            if state.is_accepting:
                sequence.append(Tokens['ACCEPT'])
            for i, input_val in enumerate(input_alphabet):
                dest_state = state.transitions[input_val]
                sequence.extend([Tokens['TRANSITION'], Tokens[f'input_{i}'], state_to_token[dest_state]])
        else:
            for i in range(Alphabet_size):
                for output, next_state in state.transitions.get(str(i), []):
                    if str(output) == 'o2':
                        sequence.append(Tokens['ACCEPT'])
                    sequence.extend([Tokens['TRANSITION'], Tokens[f'input_{i}'], state_to_token[next_state]])
            
            if type == 'EPS_NFA' and 'epsilon' in state.transitions:
                epsilon_transition= state.transitions['epsilon']
                first_trans = epsilon_transition[0]
                next_state = first_trans[1]
                
                sequence.extend([Tokens['TRANSITION'], Tokens['epsilon'], state_to_token[next_state]])
        sequence.append(Tokens['SEP'])
        
    return sequence

def sample():
    types = ['DFA', 'NFA', 'EPS_NFA']
    selected_type = random.choice(types)
    return Automata(selected_type), selected_type
