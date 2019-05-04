import pdb

import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import argparse
import os
import sys

if "../" not in sys.path:
  sys.path.append("../lib")
from helpers import splits_by_percentages

parser = argparse.ArgumentParser(description="Tic Tac Toe player")
parser.add_argument("--batch_size", type=int, default=200, help="batch size")
parser.add_argument("--epochs", type=int, default=1, help="epochs")
parser.add_argument("--show_games", action='store_true', default=False, help="print the games played")
parser.add_argument("--retrain", action='store_true', default=False, help="retrain the nn")
parser.add_argument("--clean", action='store_true', default=False, help="regenerate data and weights")
args = parser.parse_args()
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs

def safe_remove(fn):
  try:
    os.remove(fn)
  except FileNotFoundError:
    0

if args.retrain:
  safe_remove("model")

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

  next_states = [init_state]*9
  for cell in cells:
    next_state = game[-1].copy()
    next_state[cell] = player
    next_states[cell] = next_state

    next_cells = cells.copy()
    next_cells.remove(cell)
    next_game = game.copy()
    next_game += [next_state]

    get_games(other_player(player), games, next_game, next_cells)
  assert len(next_states) == 9
  games += [(game[-1], player, next_states)]
   
def switch_move_to_opponent(state, next_state):
  opp_state = [0]*9
  for i in range(9):
    if state[i] == next_state[i]:
      opp_state[i] = state[i]
    else:
      opp_state[i] = other_player(next_state[i]) 
  return opp_state

def could_win(player, state):
  opp_state = [0]*9
  for i in range(9):
    if state[i] == N:
      next_state = state.copy();
      next_state[i] = player
      if calculate_winner(next_state) == player:
        return True
  return False

def get_move_result(player, state, next_state):
  opp_state = switch_move_to_opponent(state, next_state)
  winner = calculate_winner(next_state)
  opp_winner = calculate_winner(opp_state)
  if winner == O or winner == X:
    if player == winner:
      return 1.0
  if opp_winner == O or opp_winner == X:
    if other_player(player) == opp_winner:
      return 1.0

  return 0.0
  
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
    #choices += [init_state] * (18 - len(choices))
    assert len(choices) == 18
    x.append(choices);
    outcomes = [get_move_result(player, state, next_state) for next_state in next_states]
    outcomes += [0.0] * (9 - len(next_states))
    assert len(outcomes) == 9
    y.append(outcomes)

  return np.array(x), np.array(y)

try:
  with open('data', 'rb') as data_file:
    data = pickle.load(data_file)
    data_moves = data['data_moves']
    data_outcomes = data['data_outcomes']

except:    
  data_moves, data_outcomes = generate_data(None)

  with open('data', 'wb') as data_file:
    pickle.dump({'data_moves': data_moves, 'data_outcomes': data_outcomes}, data_file)

data_moves_splits, data_outcomes_splits = splits_by_percentages([data_moves, data_outcomes], [80,10,10])
data_moves_training, data_moves_validation, data_moves_test = data_moves_splits
data_outcomes_training, data_outcomes_validation, data_outcomes_test = data_outcomes_splits

'''''
  S1+S2 -> X O Tie

  What about input the difference?
'''

try: 
  model = keras.models.load_model('model')
except:
  model = keras.Sequential()

  N_FEATURES = 128
  VOCAB_SIZE = 4 
  EMBEDDING_SIZE = 32

  model.add(keras.layers.Reshape((9*2*9,), input_shape=(18,9)))
  model.add(keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_SIZE))
  model.add(keras.layers.Reshape((9*18, EMBEDDING_SIZE)))
  model.add(keras.layers.Conv1D(N_FEATURES, (18,), strides=(18,)))
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(9, activation='softmax'))

  #model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
  #model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
  model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
  #print("training_moves: {0} training_outcomes {1}".format(training_moves.shape, training_outcomes.shape))
  model.fit(data_moves_training, data_outcomes_training, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(data_moves_validation, data_outcomes_validation))
  evaluation = model.evaluate(data_moves_test, data_outcomes_test)
  print("Test Accuracy = {0}".format(evaluation[1]))
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
    else:
      moves.append(init_state)
      moves.append(init_state)
  assert len(moves) == 18

  choices = model.predict(np.array([moves]))[0]
  move = np.argmax(choices)

  if args.show_games:
    l1 = ""
    l2 = ""
    l3 = ""
    l4 = ""
    l5 = ""
    gap = " | "
    for i in range(9):
      c = choices[i]
      s = moves[i*2+1]
      if i == move:
        l1 += "+++++++" + gap
      else:
        l1 += "       " + gap
      l2 += "{0}".format(c)[0:7] + gap
      l3 += "  " + state_to_line(s[0:3]) + "  " + gap
      l4 += "  " + state_to_line(s[3:6]) + "  " + gap
      l5 += "  " + state_to_line(s[6:9]) + "  " + gap
    print(l1)
    print(l2)
    print(l3)
    print(l4)
    print(l5)
    print("")

  return moves[move*2+1]

def state_to_char(state):
  if state == X:
    return "X"
  elif state == O:
    return "O"
  else:
    return "."

def state_to_line(state):
  state = [state_to_char(s) for s in state]
  return "{0}{1}{2}".format(state[0], state[1], state[2])

def print_state(current_player, state):
  print("{0}: {1}\n   {2}\n   {3}\n".format(current_player, state_to_line(state[0:3]), state_to_line(state[3:6]), state_to_line(state[6:9])))

def play_game(first_move_is_position = None):
  state = [N]*9

  missed_winning_move = False
  missed_block_move = False
  current_player = X
  if args.show_games:
    print("")
    print("-"*50)
    print("")
  while calculate_winner(state) is None:
    has_winning_move = could_win(current_player, state)
    if first_move_is_position:
      next_state  = state.copy()
      next_state[first_move_is_position] = current_player
      first_move_is_position = None
      prev_state = state
      state = next_state
    else:
      next_state  = pick_move(state, current_player)
      prev_state = state
      state = next_state

    has_block_move = could_win(other_player(current_player), state)

    if has_winning_move and calculate_winner(state) != current_player:
      missed_winning_move = True
    if has_block_move and calculate_winner(state) != current_player:
      missed_block_move = True
    current_player = other_player(current_player)
  # return true iff tie
  return calculate_winner(state), missed_winning_move, missed_block_move


ties = 0
x = 0
o = 0
misses_win = 0
misses_block = 0
for i in range(9):
  winner, missed_winning_move, missed_block_move = play_game(i)
  if missed_winning_move:
    misses_win += 1
  if missed_block_move:
    misses_block += 1
  if winner == TIE:
    ties += 1
  elif winner == X:
    x += 1
  elif winner == O:
    o += 1
  else:
    assert False

print("epochs: {6} batch_size {5}: {0},{1},{2} missing_winning_move: {4} missed_blocking_move: {7}".format(x, o, ties, 0, misses_win, BATCH_SIZE, EPOCHS, misses_block))
print("nn needs more structure to compare choices not just chocie to start state");
print("next move compare input data to game play data");
print("learn more about structure");
