import pdb

import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import argparse
import os

parser = argparse.ArgumentParser(description="Tic Tac Toe player")
parser.add_argument("--clean", action='store_true', default=False, help="regenerate data and weights")
args = parser.parse_args()

def safe_remove(fn):
  try:
    os.remove(fn)
  except FileNotFoundError:
    0

if args.clean:
  safe_remove("data")
  safe_remove("model")
  
N = 0
X = 1
O = 2
TIE = 3

OUTCOME_WIN = 0
OUTCOME_LOSS = 1
OUTCOME_TIE = 2

EPOCHS = 20

def game_get(game, i, j):
  game[i*3+j]

def calculate_winner(state):
  lines = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
  ];
  for i in range(len(lines)):
    a, b, c = lines[i]
    if state[a] is not N and state[a] == state[b] and state[a] == state[c]:
      return state[a]
  
  if all(sq is not N for sq in state):
    return TIE;

  return None

def other_player(player):
  if player == X:
    return O
  else:
    return X

def state_to_one_hot(state):
  one_hot = []
  for s in state:
    oh = [0, 0, 0]
    oh[s] = 1
    one_hot += oh
  return one_hot

# games state -> next states
def get_games(player, games, game = [[0 for _ in range(9)]], cells = [0,1,2,3,4,5,6,7,8]):
#def get_games(player, games, game = [[0 for _ in range(9)]], cells = [0,1]):

  if calculate_winner(game[-1]) is not None:
    return

  next_states = []
  for cell in cells:
    next_state = game[-1].copy()
    next_state[cell] = player
    next_states.append(next_state)

    next_cells = cells.copy()
    next_cells.remove(cell)
    next_game = game.copy()
    next_game += [next_state]

    get_games(other_player(player), games, next_game, next_cells)
  games += [(game[-1], player, next_states)]
    
def get_move_result(player, state):
  winner = calculate_winner(state)
  if winner == O or winner == X:
    if player == winner:
      return 1.0
    else:
      return 0.0
  elif winner == TIE:
      return 0.5
  else:
      return 0.5
  
  #return {X: 1.0, O: 0, TIE: 0.5, None: 0.5}[result]
  #return {OUTCOME_WIN: 1.0, OUTCOME_LOSS: 0, OUTCOME_TIE: 0.5, None: 0.5}[result]

init_state = [0]*9

# x: list of state+next_states
# y: one hot 1.0 if win, 0 if loose, 0.5 if tie
def generate_data(model):
  x = []
  y = []
  games = []
  get_games(X, games)

  for (state, player, next_states) in games:
    choices = []
    for next_state in next_states:
      choices += [state, next_state]
    choices += [init_state] * (18 - len(choices))
    x.append(choices);
    assert len(choices) == 18
    outcomes = [get_move_result(player, next_state) for next_state in next_states]
    outcomes += [0.0] * (9 - len(next_states))
    assert len(outcomes) == 9
    y.append(outcomes)

  return np.array(x), np.array(y)

try:
  with open('data', 'rb') as data_file:
    data = pickle.load(data_file)
    training_moves = data['training_moves']
    training_outcomes = data['training_outcomes']
except:
  training_moves, training_outcomes = generate_data(None)

  with open('data', 'wb') as data_file:
    pickle.dump({'training_moves': training_moves, 'training_outcomes': training_outcomes}, data_file)

'''
def data_stats(moves, outcomes):
  win, loss, tie = 0, 0, 0
  for outcome in outcomes:
    if outcome == OUTCOME_WIN:
      win += 1
    elif outcome == OUTCOME_LOSS:
      loss += 1
    else:
      tie += 1 
  total = win + loss + tie
  print("Data: {0}/{1}/{2}\n".format(win/total*100, loss/total*100, tie/total*100))

data_stats(training_moves, training_outcomes)
'''

'''
  S1+S2 -> X O Tie

  What about input the difference?
'''

try: 
  model = keras.models.load_model('model')
except:
  model = keras.Sequential()

  N_FEATURES = 64
  VOCAB_SIZE = 4 
  EMBEDDING_SIZE = 16

  model.add(keras.layers.Reshape((9*2*9,), input_shape=(18,9)))
  model.add(keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_SIZE))
  model.add(keras.layers.Reshape((9*18, EMBEDDING_SIZE)))
  model.add(keras.layers.Conv1D(N_FEATURES, (18,), strides=(18,)))
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(9, activation='softmax'))

  #model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  model.compile(optimizer=tf.train.AdamOptimizer(), loss='categorical_crossentropy', metrics=['accuracy'])
  print("training_moves: {0} training_outcomes {1}".format(training_moves.shape, training_outcomes.shape))
  model.fit(training_moves, training_outcomes, epochs=EPOCHS, batch_size=20)

  model.save("model")

def find_outcome(state, next_state):
  win = 0
  loss = 0
  tie = 0

  expected_state = state+next_state
  for i, move in enumerate(training_moves):
    if np.array_equal(move, expected_state):
      if training_outcomes[i] == OUTCOME_WIN:
        win += 1
      elif training_outcomes[i] == OUTCOME_LOSS:
        loss += 1
      else:
        tie += 1
  #if win == 0 and loss == 0 and tie == 0:
    #pdb.set_trace()
  return (win, loss, tie)
        
def pick_move(state, player):
  moves = []
  for i in range(9):
    if state[i] == N:
      next_state = state.copy()
      next_state[i] = player
      moves.append(state)
      moves.append(next_state)
  moves += [init_state] * (18 - len(moves))

  choices = model.predict(np.array([moves]))
  move = np.argmax(choices)
  return moves[move*2+1]

def state_to_char(state):
  if state == X:
    return "X"
  elif state == O:
    return "O"
  else:
    return " "

def state_to_line(state):
  state = [state_to_char(s) for s in state]
  return "{0}{1}{2}".format(state[0], state[1], state[2])

def print_state(current_player, state):
  print("{0}: {1}\n   {2}\n   {3}\n".format(current_player, state_to_line(state[0:3]), state_to_line(state[3:6]), state_to_line(state[6:9])))

def play_game(first_move_is_rand = False):
  state = [N]*9

  current_player = X
  while calculate_winner(state) is None:
    if first_move_is_rand:
      move = random.randint(1, len(state))-1
      next_state = [s for s in state]
      next_state[move] = current_player
      prev_state = state
      state = next_state
       
      first_move_is_rand = False
    else:
      next_state  = pick_move(state, current_player)
      prev_state = state
      state = next_state

    print_state(current_player, state) 
    ''' 
    print("{0}: {1} {4}/{5}/{6} {7}/_/{8}\n   {2}\n   {3}\n".format(current_player, state_to_line(state[0:3]), state_to_line(state[3:6]), state_to_line(state[6:9]), win, loss, tie, win_pred, tie_pred))
   
    choices = {} 
    for i in range(len(prev_state)):
      tstate = prev_state.copy()
      if tstate[i] == N:
        tstate[i] = current_player
        choices[str(tstate)] = find_outcome(prev_state, tstate)

    print(choices)
    '''
    
    current_player = other_player(current_player)


play_game(True)
