import random
from aalpy.utils import generate_random_dfa, generate_random_ONFSM

Alphabet_size = 6
Max_states = 100

Alphabet = []

for i in range (Alphabet_size):
    Alphabet.append(f'input_{i}')

State_ID_tokens = [f'ID_{i}' for i in range(Max_states)]
State_Targets_tokens = [f'Target_{i}' for i in range(Max_states)]

Special_tokens = ['PAD','epsilon', 'START', 'ACCEPT', 'TRANSITION', 'SEP']
Token_combo = State_ID_tokens + State_Targets_tokens + Alphabet + Special_tokens

Tokens = {}
for i, token in enumerate(Token_combo):
    Tokens[token] = i

PAD_INDEX = Tokens['PAD']
VOCAB_SIZE = len(Token_combo)

def Automata(type):
    num_states = random.randint(10, Max_states)
    input_alphabet= [str(i) for i in range(Alphabet_size)]
    
    automaton = generate_random_dfa(num_states, input_alphabet)
    
    available_indices = list(range(Max_states))
    random.shuffle(available_indices)
    
    state_to_id_val = {}
    state_to_target_val = {}
    
    for i, state in enumerate(automaton.states):
        state_to_id_val[state] = Tokens[f'ID_{available_indices[i]}']
        state_to_target_val[state] = Tokens[f'Target_{available_indices[i]}']
    
    adj = {}
    for state in automaton.states:
        adj[state] = {inp: [dest] for inp, dest in state.transitions.items()}
    
    if type in ['NFA', 'EPS_NFA']:
        num_extra = random.randint(num_states // 2, num_states)
        for _ in range(num_extra):
            src = random.choice(automaton.states)
            inp = random.choice(input_alphabet)
            dest = random.choice(automaton.states)
            if dest not in adj[src][inp]:
                adj[src][inp].append(dest)

        if type == 'EPS_NFA':
            for _ in range(random.randint(1,10)):
                src = random.choice(automaton.states)
                dest = random.choice(automaton.states)
                if 'epsilon' not in adj[src]:
                    adj[src]['epsilon'] = []
                if dest not in adj[src]['epsilon']:
                    adj[src]['epsilon'].append(dest)
        
    sequence = [Tokens['START'], state_to_id_val[automaton.initial_state]]
    
    shuffled_states = list(automaton.states)
    random.shuffle(shuffled_states)
    
    for state in shuffled_states:
        sequence.append(state_to_id_val[state])
        
        if state.is_accepting:
            sequence.append(Tokens['ACCEPT'])
        
        for inp, targets in adj[state].items():
            token_inp = Tokens['epsilon'] if inp == 'epsilon' else Tokens[f'input_{inp}']
            sequence.append(Tokens['TRANSITION'])
            sequence.append(token_inp)
            for dest in targets:
                sequence.append(state_to_target_val[dest])
        
        sequence.append(Tokens['SEP'])

    return sequence

def sample():
    types = ['DFA', 'NFA', 'EPS_NFA']
    selected_type = random.choice(types)
    sequence = Automata(selected_type)
    
    return sequence, selected_type
