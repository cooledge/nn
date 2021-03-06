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
from layers.ttt import TTTLayer

parser = argparse.ArgumentParser(description="Tic Tac Toe player")
parser.add_argument("--batch_size", type=int, default=200, help="batch size")
parser.add_argument("--epochs", type=int, default=1, help="epochs")
parser.add_argument("--show-games", action='store_true', default=False, help="print the games played")
parser.add_argument("--retrain", action='store_true', default=False, help="retrain the nn")
parser.add_argument("--incremental", action='store_true', default=False, help="retrain the nn by playing games only")
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
  # if all zeros no move switch
  if sum(next_state) == 0:
    return next_state
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
      return 0.5

  return 0.0

''' 
state = [1, 2, 1, 2, 1, 2, 0, 1, 0] 
next_state = [0]*9
player = O
pdb.set_trace()
result = get_move_result(player, state, next_state)
pdb.set_trace()
'''

assert get_move_result(O, [N, N, O, X, O, N, X, N, X], [O, N, O, X, O, N, X, N, X]) == 0.5
assert get_move_result(O, [N, N, O, X, O, N, X, N, X], [X, N, O, X, O, N, X, O, X]) == 0.5
assert get_move_result(O, [N, N, O, X, O, N, X, N, X], [N, X, O, X, O, N, X, O, X]) == 0.5

init_state = [0]*9

def player_to_one_hot(player):
  if player == X:
    return np.array([1.0, 0.0])
  else:
    return np.array([0.0, 1.0])

# x: list of state+next_states
# y: one hot 1.0 if win, 0 if loose, 0.5 if tie
def generate_data(model):
  x_choices = []
  x_player = []
  y = []
  games = []
  get_games(X, games)
  #games = games[1:50]

  for (state, player, next_states) in games:
    outcomes = [get_move_result(player, state, next_state) for next_state in next_states]
    outcomes += [0.0] * (9 - len(next_states))
    # skip the non-wins
    if sum(outcomes) == 0:
      continue
    if sum(outcomes) > 3:
      pdb.set_trace()
      pdb.set_trace()
    choices = []
    for next_state in next_states:
      choices.append(next_state)
    assert len(choices) == 9
    x_choices.append(choices);
    x_player.append(player_to_one_hot(player))
    assert len(outcomes) == 9
    y.append(outcomes)

  return np.array(x_choices), np.array(x_player), np.array(y)

try:
  with open('data', 'rb') as data_file:
    data = pickle.load(data_file)
    data_moves = data['data_moves']
    data_player = data['data_player']
    data_outcomes = data['data_outcomes']
except:    
  data_moves, data_player, data_outcomes = generate_data(None)
  with open('data', 'wb') as data_file:
    pickle.dump({'data_moves': data_moves, 'data_player': data_player, 'data_outcomes': data_outcomes}, data_file)

data_moves_splits, data_player_splits, data_outcomes_splits = splits_by_percentages([data_moves, data_player, data_outcomes], [80,10,10])
data_moves_training, data_moves_validation, data_moves_test = data_moves_splits
data_player_training, data_player_validation, data_player_test = data_player_splits
data_outcomes_training, data_outcomes_validation, data_outcomes_test = data_outcomes_splits


'''''
  S1+S2 -> X O Tie

  What about input the difference?
'''
'''
class TTTLayer(tf.keras.layers.Layer):

  def __init__(self, n_embedding, n_features):
    super(TTTLayer, self).__init__()
    self.n_features = n_features
    self.n_embedding = n_embedding

  def get_config(self):
    config = {'n_embedding': self.n_embedding,
              'n_features': self.n_features,
             }
    base_config = super(TTTLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
    
  # (N, 9)
  def build(self, input_shape):
    assert input_shape[-1] == self.n_embedding
    assert input_shape[-2] == 9
    self.kernel = []
    self.kernel = self.add_variable("TTTLayer_kernel", shape=[3*self.n_embedding, self.n_features])

  def f(self, t):
    return tf.reshape(t, (1, self.n_embedding))

  def board_to_features(self, board):

    row1 = tf.concat([TTTLayer.f(self, board[0]), TTTLayer.f(self, board[1]), TTTLayer.f(self, board[2])], 0)
    row2 = tf.concat([TTTLayer.f(self, board[3]), TTTLayer.f(self, board[4]), TTTLayer.f(self, board[5])], 0)
    row3 = tf.concat([TTTLayer.f(self, board[6]), TTTLayer.f(self, board[7]), TTTLayer.f(self, board[8])], 0)

    col1 = tf.concat([TTTLayer.f(self, board[0]), TTTLayer.f(self, board[3]), TTTLayer.f(self, board[6])], 0)
    col2 = tf.concat([TTTLayer.f(self, board[1]), TTTLayer.f(self, board[4]), TTTLayer.f(self, board[7])], 0)
    col3 = tf.concat([TTTLayer.f(self, board[2]), TTTLayer.f(self, board[5]), TTTLayer.f(self, board[8])], 0)

    diag1 = tf.concat([TTTLayer.f(self, board[0]), TTTLayer.f(self, board[4]), TTTLayer.f(self, board[8])], 0)
    diag2 = tf.concat([TTTLayer.f(self, board[2]), TTTLayer.f(self, board[4]), TTTLayer.f(self, board[6])], 0)

    components = [row1, row2, row3, col1, col2, col3, diag1, diag2]
    feature_layer = [tf.linalg.matmul([tf.reshape(component, (3*self.n_embedding,))], self.kernel)[0] for component in components]
    feature_layer = tf.concat(feature_layer, 0)
    feature_layer = tf.reshape(feature_layer, (len(components)*self.n_features,))

    return feature_layer

  # 9, 9
  def sample_to_features(self, sample):
    mapped = tf.map_fn(lambda board: TTTLayer.board_to_features(self, board), sample, dtype=tf.float32)
    return mapped

  # input (?, Choices, BoaGamesrd)
  def call(self, samples):
    output = tf.map_fn(lambda sample: TTTLayer.sample_to_features(self, sample), samples)
    return output
'''
def build_model():
  model = keras.Sequential()

  N_FEATURES = 32  # hope for winx/winy/nowin
  VOCAB_SIZE = 4  
  EMBEDDING_SIZE = 16

  inputChoices = keras.layers.Input(shape=(9,9))
  # one hot for current player
  inputPlayer = keras.layers.Input(shape=(2,))

  #model = keras.layers.Reshape((9, 9), input_shape=(9,9))(inputChoices)
  winLossModel = keras.layers.Reshape((9, 9))(inputChoices)
  winLossModel = keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)(winLossModel)
  winLossModel = TTTLayer(EMBEDDING_SIZE, N_FEATURES)(winLossModel)
  winLossModel = keras.layers.Flatten()(winLossModel)
  winLossModel = keras.layers.ReLU()(winLossModel)
  winLossModel = keras.Model(inputs=inputChoices, outputs=winLossModel)

  combined = keras.layers.concatenate([winLossModel.output, inputPlayer])
  combined = keras.layers.Dense(9, activation='softmax')(combined)
  combined = keras.Model(inputs=[winLossModel.input, inputPlayer], outputs=combined)

  return combined

model = build_model()

try: 
  model.load_weights('model')
  model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

  '''
  pdb.set_trace()
  # hand code the weights
  one_hot_weights = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])
  model.layers[1].set_weights([one_hot_weights])

  ttt_weights = np.array([
       [ 1.0, 0.0, 0.0],
       [ 0.0, 1.0, 0.0],
       [ 0.0, 0.0, 0.0],
       [ 0.0, 0.0, 0.0],
       [ 1.0, 0.0, 0.0],
       [ 0.0, 1.0, 0.0 ],
       [ 0.0, 0.0, 0.0],
       [ 0.0, 0.0, 0.0],
       [ 1.0, 0.0, 0.0],
       [ 0.0, 1.0, 0.0 ],
       [ 0.0, 0.0, 0.0],
       [ 0.0, 0.0, 0.0]])
  pdb.set_trace()
  model.layers[2].set_weights([ttt_weights])
  '''

except:
  #model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
  model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
  if args.incremental:
      print("Not training the model off the test data")
  else:
      model.fit([data_moves_training, data_player_training], data_outcomes_training, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=([data_moves_validation, data_player_validation], data_outcomes_validation))
      evaluation = model.evaluate([data_moves_test, data_player_test], data_outcomes_test)
      print("Test Accuracy = {0}".format(evaluation[1]))
      model.save("model")

def inverse_loss(y_true, y_pred, from_logits=False, axis=-1):
  return -keras.backend.categorical_crossentropy(y_true, y_pred, from_logits=from_logits, axis=axis)

imodel = keras.Model(inputs=model.inputs, outputs=model.outputs)
imodel.compile(optimizer=tf.keras.optimizers.Adam(), loss=inverse_loss, metrics=['accuracy'])

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
      moves.append(next_state)
    else:
      moves.append(init_state)
  assert len(moves) == 9

  #pdb.set_trace()
  inputs = [np.array([moves]), np.array([player_to_one_hot(player)])]
  choices = model.predict([np.array([moves]), np.array([player_to_one_hot(player)])])[0]
  move = np.argmax(choices)

  choice_input_moves = np.array(moves)
  choice_input_players = player_to_one_hot(player)
  choice_output = np.array([0]*9)
  choice_output[move] = 1.0

  #if moves[move] == [0]*9:
    # picking a move with no choice
    #pdb.set_trace()

  if args.show_games:
    l1 = ""
    l2 = ""
    l3 = ""
    l4 = ""
    l5 = ""
    gap = " | "
    for i in range(9):
      c = choices[i]
      s = moves[i]
      if i == move:
        l1 += "+++++++" + gap
      elif get_move_result(player, state, moves[i]) == 1.0:
        l1 += "WWWWWWW" + gap
      elif get_move_result(player, state, moves[i]) == 0.5:
        l1 += "LLLLLLL" + gap
      else:
        l1 += "       " + gap
      l2 += "{:7.5f}".format(c if c > 0.00001 else 0.0)[0:7] + gap
      l3 += "  " + state_to_line(s[0:3]) + "  " + gap
      l4 += "  " + state_to_line(s[3:6]) + "  " + gap
      l5 += "  " + state_to_line(s[6:9]) + "  " + gap
    print(l1)
    print(l2)
    print(l3)
    print(l4)
    print(l5)
    print("")

  return [moves[move], choice_input_moves, choice_input_players, choice_output]


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
  # X or O lists
  choice_inputs_moves = [[], []]
  choice_inputs_players = [[], []]
  choice_outputs = [[], []]
  counter = 0
  while calculate_winner(state) is None:
    has_winning_move = could_win(current_player, state)
    if first_move_is_position:
      next_state  = state.copy()
      next_state[first_move_is_position] = current_player
      first_move_is_position = None
      prev_state = state
      state = next_state
    else:
      next_state, choice_input_moves, choice_input_players, choice_output  = pick_move(state, current_player)
      choice_inputs_moves[current_player-1].append(choice_input_moves)
      choice_inputs_players[current_player-1].append(choice_input_players)
      choice_outputs[current_player-1].append(choice_output)
      # invalid moves count as loss
      if args.incremental and sum(next_state) == 0:
        return other_player(current_player), False, False, choice_inputs_moves, choice_inputs_players, choice_outputs
      prev_state = state
      state = next_state

    has_block_move = could_win(other_player(current_player), state)

    if has_winning_move and calculate_winner(state) != current_player:
      missed_winning_move = True
    if has_block_move and calculate_winner(state) != current_player:
      missed_block_move = True
    current_player = other_player(current_player)
    counter += 1
    if counter > 9:
      break
  # return true iff tie
  return calculate_winner(state), missed_winning_move, missed_block_move, choice_inputs_moves, choice_inputs_players, choice_outputs

def move_equal(m1, m2):
  for i in range(len(m1)):
    if m1[i] != m2[i]:
      return False
  return True

def moves_matches(moves, move, position):
  return move_equal(moves[position], move)

def matches(moves, outcomes, move, position):
  print(move)
  #pdb.set_trace()
  return [(moves[i], outcomes[i]) for i in range(len(moves)) if moves_matches(moves[i], move, position)]

#print("Test Moves");
#print(matches(data_moves, [1, 2, 1, 2, 1, 2, 2, 1, 1], 8))

'''
state = [N, N, N, N, N, O, N, X, X]
next_state = [N, N, N, N, N, O, O, X, X]
pdb.set_trace()
result = get_move_result(O, state, next_state)
print("Result is " + result)
'''
'''
print("Desired move");
print(matches(data_moves, data_outcomes, [N, N, N, N, N, O, O, X, X], 6))
print("Actual move");
print(matches(data_moves, data_outcomes, [N, N, N, O, N, O, N, X, X], 3))
'''
#play_game(3)

def run_games():
  ties = 0
  x = 0
  o = 0
  no_winner = 0
  misses_win = 0
  misses_block = 0
  for i in range(9):
    winner, missed_winning_move, missed_block_move, choice_inputs_moves, choice_inputs_players, choice_outputs = play_game(i)
   
    # incremental re-training 

    if winner != TIE:
      if len(choice_inputs_moves[0]) != 0:
        mx = model if winner == X else imodel
        mx.fit([np.array(choice_inputs_moves[0]), np.array(choice_inputs_players[0])], np.array(choice_outputs[0])) # , validation_data=([data_moves_validation, data_player_validation], data_outcomes_validation))
      if len(choice_inputs_moves[1]) != 0:
        mo = model if winner == O else imodel
        mo.fit([np.array(choice_inputs_moves[1]), np.array(choice_inputs_players[1])], np.array(choice_outputs[1])) # , validation_data=([data_moves_validation, data_player_validation], data_outcomes_validation))
    
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
      no_winner += 1

  print("epochs: {6} batch_size {5}: {0},{1},{2},{8} missing_winning_move: {4} missed_blocking_move: {7}".format(x, o, ties, 0, misses_win, BATCH_SIZE, EPOCHS, misses_block, no_winner))

for i in range(1000):
  run_games()

